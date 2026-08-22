"""Edge-TTS synthesis and sample-accurate placement onto the dub track."""
from __future__ import annotations

import asyncio
import hashlib
import re
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
def _clean_for_speech(text: str) -> str:
    text = re.sub("[一-鿿　-〿＀-￯]+", " ", text)  # stray CJK
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
    plan: list[tuple[dict, dict, float]] = []
    for i, seg in enumerate(segments):
        r = results.get(seg["id"])
        if not r or not r.get("file"):
            continue
        nxt = segments[i + 1]["start"] if i + 1 < len(segments) else total_duration

        # Ask for a breath, then check what it costs. If leaving the pause would
        # push this line past a comfortable speaking rate, take a shorter breath
        # instead - a rushed line is worse than a short pause.
        want = breath_after(seg.get("en", "")) if i + 1 < len(segments) else MIN_GAP
        room = nxt - seg["start"]
        gap = max(0.30, room - want)
        if r["duration"] / max(gap, 1e-6) > BREATH_TEMPO_LIMIT:
            relaxed = max(MIN_GAP, r["duration"] / BREATH_TEMPO_LIMIT)
            gap = max(0.30, min(room - MIN_GAP, relaxed))
        tempo = r["duration"] / gap if gap > 0 else 1.0
        # The Settings slider caps the *total* rate, and part of it was already
        # spent on the faster speaking rate, so only the remainder is available
        # here. Beyond that speech stops being intelligible, and those few lines
        # are allowed to run into the next one rather than being mangled.
        ceiling = max(1.0, min(HARD_TEMPO_CEIL,
                               max_speedup / r.get("native_rate", 1.0)))
        tempo = min(max(1.0, tempo), ceiling)
        plan.append((seg, r, tempo))

    # Decoding is one ffmpeg launch per line; run them across threads so a long
    # video does not spend minutes waiting on serial subprocess startup.
    clips: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
        futures = {pool.submit(decode_pcm, r["file"], tempo): seg["id"]
                   for seg, r, tempo in plan}
        for fut in as_completed(futures):
            try:
                clips[futures[fut]] = fut.result()
            except Exception:
                pass

    track = np.zeros(int(round(total_duration * SR)) + SR, dtype=np.float32)
    stats = {"lines": 0, "compressed": 0, "drift": 0.0, "failed": 0,
             "max_speedup_used": 1.0, "overlapped": 0}

    placed_end = 0.0
    for seg, r, tempo in plan:
        clip = clips.get(seg["id"])
        if clip is None:
            stats["failed"] += 1
            continue

        clip = fade(clip)
        start = seg["start"]              # hard anchor - never shifted
        place(track, clip, start)

        actual = len(clip) / SR
        total_rate = tempo * r.get("native_rate", 1.0)
        seg["dub_start"] = round(start, 3)
        seg["dub_end"] = round(start + actual, 3)
        seg["dub_speedup"] = round(total_rate, 3)

        stats["lines"] += 1
        if total_rate > 1.03:
            stats["compressed"] += 1
        if start < placed_end - 0.02:
            stats["overlapped"] += 1
        stats["max_speedup_used"] = max(stats["max_speedup_used"], total_rate)
        placed_end = start + actual

    for seg in segments:
        if seg.get("en", "").strip() and seg["id"] not in clips:
            stats["failed"] += 1

    out = write_wav(track, workdir / "dub.wav")
    stats["max_speedup_used"] = round(stats["max_speedup_used"], 2)
    return out, stats


def preview(text: str, voice: str, dst: Path, pitch: int = 0,
            rate_pct: int = 0, volume: int = 0) -> Path:
    """Render a sample line. A cloned voice renders as its base voice here -
    the caller converts the result to the cloned timbre."""
    from . import voiceclone
    asyncio.run(_say(text, voiceclone.base_voice_for(voice), dst,
                     rate_pct=rate_pct, pitch_hz=pitch, volume_pct=volume))
    return dst
