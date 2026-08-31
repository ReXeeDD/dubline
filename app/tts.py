"""Edge-TTS synthesis and sample-accurate placement onto the dub track."""
from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import edge_tts
import numpy as np

from .media import SR, decode_pcm, duration_of, fade, place, write_wav

# Parallel edge-tts streams. Measured on 60 real dub lines: 12 gives ~7 lines/s,
# 24 gives ~10, and 40 gives nothing more than 24 - the service stops rewarding
# extra connections. Network variance between runs is large, so this is the
# level two separate runs agreed on rather than the best single reading.
CONCURRENCY = 24
DECODE_WORKERS = 8       # parallel ffmpeg decodes when assembling the track
NATIVE_RATE_CAP = 1.35   # how far edge-tts's own prosody rate is pushed first
HARD_TEMPO_CEIL = 1.55   # ceiling on extra time-compression at assembly time
MIN_GAP = 0.06           # smallest gap ever left before the next line starts

# Whisper's segments run edge to edge - in a real transcript 98% of lines start
# exactly where the previous one ended. Dubbed literally that becomes an unbroken
# wall of speech, which is the single thing that stops it sounding like a person.
# A human pauses by roughly this much, and how long depends on the punctuation
# they just spoke. Measured on the reference recording: 320 ms median, 420 ms at
# the 90th percentile.
BREATH_SENTENCE = 0.30   # after . ! ? … - a full stop for breath
BREATH_CLAUSE = 0.17     # after , ; : - a shorter beat
BREATH_RUNON = 0.09      # mid-thought, where a speaker barely lifts

# A breath is only worth taking if the line is not already sprinting to fit. Past
# this the gap is given back rather than compressing speech further, because fast
# speech sounds far less human than a missing pause.
BREATH_TEMPO_LIMIT = 1.25

# What to do when a line still will not fit even at the speed-up the viewer
# allowed. Until now the answer was to let it run on top of the next line, and
# on real material that happened constantly: measured across seven finished
# videos, 562 lines overlapped the next speaker, the worst by 3.4 seconds. In
# this genre 98% of lines have no gap after them at all, so an overrun always
# lands on speech rather than on silence.
#
# Two voices at once is worse than fast speech, so a line that is about to
# collide is compressed past the viewer's cap - but only that line, only by as
# much as the collision needs, and never past a rate that stops being words.
RESCUE_TEMPO_CEIL = 1.9    # assembly-time compression allowed to avoid a clash
RESCUE_TOTAL_RATE = 2.15   # total of voice rate and assembly rate, never passed
MIN_SEPARATION = 0.04      # silence guaranteed before the next line starts


def _fit(clip: np.ndarray, limit: float) -> np.ndarray:
    """Cut a clip that would still run into the next speaker, and fade it out.

    Only reached when even the rescue tempo was not enough - roughly one line in
    a thousand. The fade is far longer than the usual butt-joint ramp so the
    voice sounds like it trails off rather than being sliced mid-word.
    """
    # The final line is given an unbounded limit - it has nothing to collide
    # with, and its tail past the end of the video is worth keeping.
    if limit <= 0 or limit == float("inf") or len(clip) <= int(limit * SR):
        return fade(clip)
    return fade(clip[:int(limit * SR)], ms=55.0)


# ------------------------------------------------------------------ voices ---
_VOICE_CACHE: list[dict] | None = None


async def _fetch_voices() -> list[dict]:
    voices = await edge_tts.list_voices()
    out = []
    for v in voices:
        if not v["Locale"].startswith("en-"):
            continue
        tags = v.get("VoiceTag", {}) or {}
        out.append({
            "id": v["ShortName"],
            "name": v["ShortName"].split("-")[-1].replace("Neural", ""),
            "locale": v["Locale"],
            "gender": v["Gender"],
            "multilingual": "Multilingual" in v["ShortName"],
            "personalities": tags.get("VoicePersonalities", []),
        })
    order = {"en-US": 0, "en-GB": 1, "en-AU": 2, "en-CA": 3, "en-IE": 4}
    out.sort(key=lambda v: (order.get(v["locale"], 9), not v["multilingual"], v["name"]))
    return out


def list_voices() -> list[dict]:
    """Cloned voices first, then the standard edge-tts catalogue."""
    global _VOICE_CACHE
    if _VOICE_CACHE is None:
        _VOICE_CACHE = asyncio.run(_fetch_voices())

    from . import voiceclone
    clones = [{
        "id": c["id"],
        "name": c["name"],
        "locale": "clone",
        "gender": c.get("gender", "Unknown"),
        "multilingual": False,
        "personalities": [],
        "clone": True,
    } for c in voiceclone.list_clones()]
    return clones + _VOICE_CACHE


# --------------------------------------------------------------- synthesis ---
# Characters that reach the voice looking like text but are not spoken as text.
# The non-breaking hyphen is what LLMs reach for in "god-tier" and "max-level";
# it is not the ASCII hyphen and an English voice has no rule for it. The rest
# are quotation and dash forms that only exist to be read on a page.
_TYPOGRAPHY = {
    "‑": "-", "‒": "-", "–": "-", "—": " - ",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "…": " ", "­": "",
}
# Vietnamese vowels carry marks that an English voice cannot read: "Loan Phuong"
# is a name it can say, "Loan Phượng" is one it stumbles over. Measured across
# the library, 159 lines - 3% of the Vietnamese ones - carry a name like this.
# Only the SPOKEN text is folded down; the subtitle keeps the correct spelling,
# which is what the viewer reads.
_VN_LETTERS = str.maketrans({"đ": "d", "Đ": "D", "ơ": "o", "Ơ": "O",
                             "ư": "u", "Ư": "U", "ă": "a", "Ă": "A"})


def _clean_for_speech(text: str) -> str:
    """The line as the voice should receive it, not as the viewer reads it."""
    text = re.sub("[一-鿿　-〿＀-￯]+", " ", text)  # stray CJK
    for bad, good in _TYPOGRAPHY.items():
        text = text.replace(bad, good)
    # Nothing is said by a line that opens mid-thought, and the voice reads the
    # dots as a long unexplained pause at the very moment it should be talking.
    text = re.sub(r"^\s*(?:\.\.\.|…)\s*", "", text)
    text = text.translate(_VN_LETTERS)
    # Strip the tone marks that are left, keeping the base letter. Latin only -
    # a decomposed CJK character must not lose anything here.
    text = "".join(c for c in unicodedata.normalize("NFD", text)
                   if not (unicodedata.combining(c) and c.isascii() is False
                           and ord(c) < 0x0370))
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def _say(text: str, voice: str, dst: Path, rate_pct: int = 0,
               pitch_hz: int = 0, volume_pct: int = 0) -> float:
    """Synthesize to `dst` and return the spoken length in seconds.

    The stream carries WordBoundary events alongside the audio, so the duration
    comes back for free. Probing the file instead would mean spawning ffprobe
    twice per subtitle line - thousands of blocking subprocess launches that
    stall the event loop and serialise the whole stage.
    """
    comm = edge_tts.Communicate(
        text,
        voice,
        rate=f"{rate_pct:+d}%",
        pitch=f"{pitch_hz:+d}Hz",
        volume=f"{volume_pct:+d}%",
    )

    audio = bytearray()
    end_ticks = 0          # 100-nanosecond units, as edge-tts reports them
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            end_ticks = max(end_ticks, int(chunk["offset"]) + int(chunk["duration"]))

    dst.write_bytes(bytes(audio))
    if end_ticks:
        return end_ticks / 1e7
    # no word boundaries (punctuation-only line): fall back to probing off-thread
    return await asyncio.to_thread(duration_of, dst)


def _fingerprint(text: str, voice: str, pitch: int, speed: int, volume: int) -> str:
    """Identity of a rendered line: everything that changes how it sounds.

    It goes in the filename, so a re-run reuses a line only when the text and
    every voice setting still match, and an edited line is simply a different
    file rather than something that has to be invalidated by hand.
    """
    raw = f"{voice}|{pitch}|{speed}|{volume}|{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _usable(path: Path) -> bool:
    """A file left by a run that died mid-write is short - treat it as absent."""
    try:
        return path.is_file() and path.stat().st_size > 512
    except OSError:
        return False


async def _render_one(seg: dict, voice: str, workdir: Path, window: float,
                      max_speedup: float, sem: asyncio.Semaphore,
                      pitch: int = 0, speed: int = 0, volume: int = 0) -> dict:
    """Synthesize one line and fit it into `window` seconds.

    `speed` is the voice's own baseline pace, chosen by the user. Any extra
    compression needed to fit the window multiplies on top of it rather than
    replacing it, so a line that runs long is still spoken at the user's chosen
    pace plus whatever the timing demands.
    """
    base = 1.0 + speed / 100.0
    text = _clean_for_speech(seg.get("en", ""))
    if not text:
        return {"id": seg["id"], "file": None, "duration": 0.0, "speedup": 1.0}

    fp = _fingerprint(text, voice, pitch, speed, volume)
    raw = workdir / f"tts_{seg['id']:05d}_{fp}.mp3"
    natural = 0.0
    if _usable(raw):
        # Already rendered with these exact settings on an earlier run.
        natural = await asyncio.to_thread(duration_of, raw)
    else:
        async with sem:
            for attempt in range(3):
                try:
                    natural = await _say(text, voice, raw, rate_pct=speed,
                                         pitch_hz=pitch, volume_pct=volume)
                    if _usable(raw):
                        break
                    raise RuntimeError("empty audio returned")
                except Exception:
                    if attempt == 2:
                        return {"id": seg["id"], "file": None, "duration": 0.0,
                                "speedup": 1.0, "error": "tts failed"}
                    await asyncio.sleep(1.2 * (attempt + 1))

    if natural <= 0:
        return {"id": seg["id"], "file": None, "duration": 0.0, "speedup": 1.0}

    needed = natural / window if window > 0 else 1.0

    # Comfortably inside the window: leave it completely untouched.
    if needed <= 1.02:
        return {"id": seg["id"], "file": raw, "duration": natural,
                "speedup": 1.0, "natural": natural, "window": window}

    # Re-synthesise at a faster speaking rate. edge-tts's own prosody rate
    # sounds markedly better than time-compressing already-rendered audio, so
    # it absorbs as much of the overrun as it can. Anything left over is handled
    # at assembly time, where the true available gap is known.
    native = min(needed, NATIVE_RATE_CAP)
    rate_pct = int(round((base * native - 1.0) * 100))
    fast = workdir / f"tts_{seg['id']:05d}_{fp}_r{rate_pct}.mp3"
    actual = natural
    if _usable(fast):
        actual = await asyncio.to_thread(duration_of, fast)
    else:
        async with sem:
            try:
                actual = await _say(text, voice, fast, rate_pct=rate_pct,
                                    pitch_hz=pitch, volume_pct=volume)
            except Exception:
                fast = raw

    src = fast if _usable(fast) else raw
    if src is raw:
        actual = natural

    return {"id": seg["id"], "file": src, "duration": actual,
            "native_rate": native, "natural": natural, "window": window}


def breath_after(text: str) -> float:
    """How long a speaker would pause after saying this line."""
    t = (text or "").strip()
    if not t:
        return MIN_GAP
    if t[-1] in ".!?…":
        return BREATH_SENTENCE
    if t[-1] in ",;:—-":
        return BREATH_CLAUSE
    return BREATH_RUNON


def _windows(segments: list[dict], total: float) -> list[float]:
    """Time each line may occupy: its own slot, less the breath that follows.

    The breath is requested, not guaranteed. A line that would have to sprint to
    fit keeps its time instead - see the placement step, which hands the gap back
    rather than compressing speech past the point where it stops sounding human.
    """
    out = []
    for i, s in enumerate(segments):
        nxt = segments[i + 1]["start"] if i + 1 < len(segments) else total
        want = breath_after(s.get("en", "")) if i + 1 < len(segments) else MIN_GAP
        out.append(max(0.35, (nxt - want) - s["start"]))
    return out


async def _render_all(segments, voice, workdir, windows, max_speedup, progress,
                      pitch=0, speed=0, volume=0):
    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0
    results: dict[int, dict] = {}

    async def one(seg, window):
        nonlocal done
        r = await _render_one(seg, voice, workdir, window, max_speedup, sem,
                              pitch, speed, volume)
        results[seg["id"]] = r
        done += 1
        if progress and done % 5 == 0:
            progress(f"Generating speech {done}/{len(segments)}",
                     72 + int(20 * done / max(1, len(segments))))
        return r

    await asyncio.gather(*(one(s, w) for s, w in zip(segments, windows)))
    return results


def build_dub_track(segments: list[dict], voice: str, workdir: Path,
                    total_duration: float, max_speedup: float = 1.7,
                    progress=None, pitch: int = 0, speed: int = 0,
                    volume: int = 0) -> tuple[Path, dict]:
    """Synthesize every line and lay it down at its exact original timestamp.

    A cloned voice is spoken by its base edge-tts voice here; the conversion to
    the cloned timbre happens once on the finished track, in the pipeline.
    """
    from . import voiceclone
    voice = voiceclone.base_voice_for(voice)

    spoken = [s for s in segments if s.get("en", "").strip()]
    windows = _windows(segments, total_duration)
    win_by_id = {s["id"]: w for s, w in zip(segments, windows)}

    if progress:
        progress(f"Generating speech for {len(spoken)} lines", 72)

    results = asyncio.run(_render_all(
        spoken, voice, workdir, [win_by_id[s["id"]] for s in spoken],
        max_speedup, progress, int(pitch), int(speed), int(volume)))

    if progress:
        progress("Aligning dubbed audio to the video timeline", 93)

    # Each line is anchored to its own transcript timestamp and compressed to
    # fit the real gap before the next one. Letting a long line push the next
    # line later instead makes the delay accumulate: over 1500 dense lines that
    # compounds into minutes of desync by the end of the video.
    #
    # The decisions themselves live in _plan, which the streaming assembler also
    # calls. This used to be a second copy of that logic, which meant a line
    # could be timed one way when a video was dubbed in one pass and another way
    # when it was streamed - and only one of the copies got fixes.
    stats = {"lines": 0, "compressed": 0, "drift": 0.0, "failed": 0,
             "max_speedup_used": 1.0, "overlapped": 0, "crowded": 0, "clipped": 0}
    plan = _plan(segments, results, total_duration, max_speedup,
                 0, len(segments), stats)

    # Decoding is one ffmpeg launch per line; run them across threads so a long
    # video does not spend minutes waiting on serial subprocess startup.
    clips: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
        futures = {pool.submit(decode_pcm, r["file"], tempo): seg["id"]
                   for seg, r, tempo, _lim in plan}
        for fut in as_completed(futures):
            try:
                clips[futures[fut]] = fut.result()
            except Exception:
                pass

    track = np.zeros(int(round(total_duration * SR)) + SR, dtype=np.float32)

    placed_end = 0.0
    for seg, r, tempo, limit in plan:
        clip = clips.get(seg["id"])
        if clip is None:
            stats["failed"] += 1
            continue

        clip = _fit(clip, limit)
        start = seg["start"]              # hard anchor - never shifted
        place(track, clip, start)

        actual = len(clip) / SR
        total_rate = tempo * r.get("native_rate", 1.0)
        seg["dub_start"] = round(start, 3)
        seg["dub_end"] = round(start + actual, 3)
        seg["dub_speedup"] = round(total_rate, 3)

        stats["lines"] += 1
        if start < placed_end - 0.02:
            stats["overlapped"] += 1
        placed_end = start + actual

    for seg in segments:
        if seg.get("en", "").strip() and seg["id"] not in clips:
            stats["failed"] += 1

    out = write_wav(track, workdir / "dub.wav")
    stats["max_speedup_used"] = round(stats["max_speedup_used"], 2)
    return out, stats


class StreamMixer:
    """Renders the dub a time window at a time, for live playback.

    build_dub_track lays the whole video down in one pass over one big array.
    That cannot start playing until it finishes, so this does the same work in
    order, handing back exactly the samples covering each window as it goes.

    The one thing it has to get right is the seam. A line that starts near the
    end of a window carries on past it, and the samples that spill over are
    kept and mixed into the front of the next window rather than being cut - a
    dropped tail would be an audible click on every window boundary, and a
    dropped *sample* would desync everything after it.
    """

    # Room beyond the window for a line that starts inside it and runs on. The
    # longest a single dub line can run is its own slot plus the speed-up
    # allowance, comfortably inside this.
    TAIL_SECONDS = 30.0

    def __init__(self, segments: list[dict], total_duration: float, voice: str,
                 workdir: Path, max_speedup: float = 1.7, pitch: int = 0,
                 speed: int = 0, volume: int = 0):
        from . import voiceclone
        self.segments = segments
        self.total = total_duration
        self.voice = voiceclone.base_voice_for(voice)
        self.workdir = workdir
        self.max_speedup = max_speedup
        self.pitch, self.speed, self.volume = int(pitch), int(speed), int(volume)

        # Slot widths depend on the following line, so they are worked out once
        # over the whole transcript - not per window, where the last line would
        # have no successor to measure against.
        widths = _windows(segments, total_duration)
        self.win_by_id = {s["id"]: w for s, w in zip(segments, widths)}

        self.pos = 0.0                                    # start of next window
        self.carry = np.zeros(0, dtype=np.float32)        # spill from last one
        self.stats = {"lines": 0, "compressed": 0, "drift": 0.0, "failed": 0,
                      "max_speedup_used": 1.0, "overlapped": 0,
                      "crowded": 0, "clipped": 0}

    def render(self, lo: int, hi: int, until: float, progress=None) -> np.ndarray:
        """Speak segments[lo:hi] and return the audio covering [pos, until)."""
        until = min(max(until, self.pos), self.total)
        n = int(round((until - self.pos) * SR))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        spoken = [s for s in self.segments[lo:hi] if s.get("en", "").strip()]
        results: dict[int, dict] = {}
        if spoken:
            results = asyncio.run(_render_all(
                spoken, self.voice, self.workdir,
                [self.win_by_id[s["id"]] for s in spoken],
                self.max_speedup, progress, self.pitch, self.speed, self.volume))

        plan = _plan(self.segments, results, self.total, self.max_speedup,
                     lo, hi, self.stats)

        buf = np.zeros(n + int(self.TAIL_SECONDS * SR), dtype=np.float32)
        buf[:len(self.carry)] += self.carry[:len(buf)]

        clips: dict[int, np.ndarray] = {}
        if plan:
            with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                futures = {pool.submit(decode_pcm, r["file"], tempo): seg["id"]
                           for seg, r, tempo, _lim in plan}
                for fut in as_completed(futures):
                    try:
                        clips[futures[fut]] = fut.result()
                    except Exception:
                        pass

        for seg, r, tempo, limit in plan:
            clip = clips.get(seg["id"])
            if clip is None:
                self.stats["failed"] += 1
                continue
            clip = _fit(clip, limit)
            place(buf, clip, seg["start"] - self.pos)     # offset within window
            actual = len(clip) / SR
            seg["dub_start"] = round(seg["start"], 3)
            seg["dub_end"] = round(seg["start"] + actual, 3)
            seg["dub_speedup"] = round(tempo * r.get("native_rate", 1.0), 3)
            self.stats["lines"] += 1

        out = buf[:n].copy()
        self.carry = buf[n:]
        self.pos = until
        return out

    # A line may run a little past the end of the video, and that much is worth
    # keeping. Beyond it the carry buffer is only the unused scratch space, and
    # feeding that to the stream would leave the audio longer than the picture.
    OVERRUN_LIMIT = 1.0

    def drain(self) -> np.ndarray:
        """The last line's tail, if it runs past the end of the final window."""
        tail, self.carry = self.carry, np.zeros(0, dtype=np.float32)
        tail = tail[:int(self.OVERRUN_LIMIT * SR)]
        voiced = np.nonzero(np.abs(tail) > 1e-5)[0]
        return tail[:int(voiced[-1]) + 1] if voiced.size else tail[:0]


def _plan(segments: list[dict], results: dict, total_duration: float,
          max_speedup: float, lo: int, hi: int, stats: dict) -> list:
    """Decide how fast each rendered line has to be spoken to fit its gap.

    Shared by the one-pass and the streaming assembler so both make exactly the
    same timing decisions - a line must not land differently depending on which
    of the two produced it.
    """
    plan = []
    for i in range(lo, hi):
        seg = segments[i]
        r = results.get(seg["id"])
        if not r or not r.get("file"):
            continue
        nxt = segments[i + 1]["start"] if i + 1 < len(segments) else total_duration
        # Cleared before they can be set again below. These are saved into
        # segments.json and the transcript draws a "cut short" badge from them,
        # so a line that was rushed once and has since been shortened by hand
        # would otherwise keep the badge for the life of the video.
        seg.pop("crowded", None)
        seg.pop("clipped", None)

        want = breath_after(seg.get("en", "")) if i + 1 < len(segments) else MIN_GAP
        room = nxt - seg["start"]
        gap = max(0.30, room - want)
        if r["duration"] / max(gap, 1e-6) > BREATH_TEMPO_LIMIT:
            relaxed = max(MIN_GAP, r["duration"] / BREATH_TEMPO_LIMIT)
            gap = max(0.30, min(room - MIN_GAP, relaxed))
        tempo = r["duration"] / gap if gap > 0 else 1.0
        ceiling = max(1.0, min(HARD_TEMPO_CEIL,
                               max_speedup / r.get("native_rate", 1.0)))
        tempo = min(max(1.0, tempo), ceiling)

        # Fitting the slot is preferred; not talking over the next speaker is
        # required. The last line has nothing to collide with, so it keeps the
        # tail that runs past the end of the video.
        if i + 1 < len(segments):
            limit = max(0.20, nxt - seg["start"] - MIN_SEPARATION)
            if r["duration"] / tempo > limit:
                rescue = min(RESCUE_TEMPO_CEIL,
                             RESCUE_TOTAL_RATE / r.get("native_rate", 1.0))
                tempo = min(max(tempo, r["duration"] / limit), max(tempo, rescue))
                stats["crowded"] += 1
                # Recorded on the line itself, not only in the totals, so the
                # transcript can point at the handful of lines that had to be
                # rushed or cut and they can be shortened by hand.
                seg["crowded"] = True
                if r["duration"] / tempo > limit + 0.01:
                    stats["clipped"] += 1
                    seg["clipped"] = True
        else:
            limit = float("inf")

        if tempo > 1.01:
            stats["compressed"] += 1
        stats["max_speedup_used"] = max(
            stats["max_speedup_used"], round(tempo * r.get("native_rate", 1.0), 3))
        plan.append((seg, r, tempo, limit))
    return plan


def preview(text: str, voice: str, dst: Path, pitch: int = 0,
            rate_pct: int = 0, volume: int = 0) -> Path:
    """Render a sample line. A cloned voice renders as its base voice here -
    the caller converts the result to the cloned timbre."""
    from . import voiceclone
    asyncio.run(_say(text, voiceclone.base_voice_for(voice), dst,
                     rate_pct=rate_pct, pitch_hz=pitch, volume_pct=volume))
    return dst
