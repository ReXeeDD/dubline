"""Speech recognition via Groq-hosted Whisper, with silence-aware chunking."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from .config import ffmpeg
from .media import ASR_EXT, _NOWIN, media_info, slice_audio

# Groq caps uploads at 25 MB. 16 kHz mono Opus runs ~4 KB/s, so a 15 minute
# chunk lands near 3.5 MB - roughly a seventh of the limit. Anything that still
# comes out oversized is split in half automatically rather than failing.
CHUNK_SECONDS = 900.0
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
MIN_CHUNK_SECONDS = 30.0
# Every chunk after the first starts this far early. Video with continuous
# music has no silence to cut at, so a hard cut would slice a word in half;
# the lead-in gives Whisper the run-up, and the duplicated span is discarded
# when the results are merged.
OVERLAP = 2.5


def _silences(audio: Path, noise_db: int = -32, min_dur: float = 0.35) -> list[float]:
    """Midpoints of detected silences - safe places to cut without clipping speech."""
    p = subprocess.run(
        [ffmpeg(), "-i", str(audio), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_dur}", "-f", "null", "-"],
        capture_output=True, **_NOWIN,
    )
    log = p.stderr.decode("utf-8", "ignore")
    starts = [float(m) for m in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", log)]
    return [(s + e) / 2.0 for s, e in zip(starts, ends) if e > s]


def _quiet_points(audio: Path) -> list[float]:
    """Loosen the threshold until something turns up - a track with constant
    background music never dips below -32 dB."""
    for db in (-32, -26, -20):
        found = _silences(audio, noise_db=db)
        if found:
            return found
    return []


def _boundaries(duration: float, audio: Path) -> list[tuple[float, float]]:
    if duration <= CHUNK_SECONDS:
        return [(0.0, duration)]

    quiet = _quiet_points(audio)
    cuts: list[float] = []
    target = CHUNK_SECONDS
    while target < duration - 30.0:
        # snap to the quietest moment within +/-45s of the target, else hard cut
        near = [q for q in quiet if abs(q - target) < 45.0 and q > (cuts[-1] if cuts else 0) + 60]
        cut = min(near, key=lambda q: abs(q - target)) if near else target
        cuts.append(cut)
        target = cut + CHUNK_SECONDS

    edges = [0.0] + cuts + [duration]
    return [(edges[i], edges[i + 1] - edges[i]) for i in range(len(edges) - 1)]


# Groq's verbose_json sometimes names the language in full rather than by code.
_LANG_CODE = {
    "chinese": "zh", "mandarin": "zh", "vietnamese": "vi", "japanese": "ja",
    "korean": "ko", "thai": "th", "english": "en", "spanish": "es",
    "french": "fr", "german": "de", "russian": "ru", "portuguese": "pt",
    "indonesian": "id", "hindi": "hi", "arabic": "ar", "italian": "it",
}


def _part_key(model: str, language: str, offset: float, span: float) -> str:
    raw = f"{model}|{language}|{offset:.3f}|{span:.3f}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def transcribe(client, audio: Path, model: str, language: str,
               workdir: Path, progress=None) -> tuple[list[dict], str]:
    """Return ([{id, start, end, text}], language) with whole-file timestamps.

    The language is what Whisper actually heard, which with source_language on
    "auto" is the only place it is ever named.

    Every chunk's result is written to work/asr_parts as soon as it comes back.
    A long video is several uploads, and losing the whole transcript because the
    last one timed out means paying for and waiting on all of them again - so a
    re-run picks up from the first chunk that never finished.
    """
    duration = media_info(audio)["duration"]
    chunks = _boundaries(duration, audio)
    segments: list[dict] = []

    parts = workdir / "asr_parts"
    parts.mkdir(parents=True, exist_ok=True)
    heard: dict[str, int] = {}      # language votes, one per chunk

    # worklist so an oversized piece can be replaced by two smaller ones
    pending = list(chunks)
    done = 0
    seq = 0

    while pending:
        start, length = pending.pop(0)
        if progress:
            total = done + len(pending) + 1
            progress(f"Transcribing audio {done + 1}/{total}",
                     10 + int(30 * done / max(1, total)))

        # start a little early so a cut cannot land mid-word
        lead = OVERLAP if start > 0.05 else 0.0
        offset = start - lead

        # _boundaries is deterministic for a given audio file, so a chunk keeps
        # the same key across runs and an earlier result is still valid.
        cached = parts / f"{_part_key(model, language, offset, length + lead)}.json"
        if cached.is_file():
            try:
                blob = json.loads(cached.read_text(encoding="utf-8"))
                segments.extend(blob["segments"])
                if blob.get("language"):
                    heard[blob["language"]] = heard.get(blob["language"], 0) + 1
                done += 1
                continue
            except Exception:
                cached.unlink(missing_ok=True)

        if len(chunks) == 1 and not done and not pending:
            part = audio
        else:
            part = slice_audio(audio, workdir / f"asr_{seq:03d}{ASR_EXT}",
                               offset, length + lead)
            seq += 1

        size = part.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            if length / 2 < MIN_CHUNK_SECONDS:
                raise RuntimeError(
                    f"A {length:.0f}s piece of audio still encodes to "
                    f"{size / 1e6:.1f} MB, over Groq's 25 MB limit.")
            part.unlink(missing_ok=True)
            half = length / 2
            pending.insert(0, (start + half, length - half))
            pending.insert(0, (start, half))
            continue

        kwargs = dict(
            file=(part.name, part.read_bytes()),
            model=model,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            temperature=0.0,
        )
        if language and language != "auto":
            kwargs["language"] = language

        resp = client.audio.transcriptions.create(**kwargs)
        raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

        # Whisper reports what it actually heard. With source_language on auto
        # that is the only place the real language is ever named, and the
        # translator needs it: its pronoun handling is language-specific.
        if raw.get("language"):
            heard[str(raw["language"]).strip().lower()] = \
                heard.get(str(raw["language"]).strip().lower(), 0) + 1

        got: list[dict] = []
        for s in raw.get("segments") or []:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            a = float(s["start"]) + offset
            b = float(s["end"]) + offset
            # a line sitting mostly in the lead-in already came from the
            # previous chunk, so drop it rather than transcribing it twice
            if lead and (a + b) / 2.0 < start:
                continue
            got.append({"start": round(a, 3), "end": round(b, 3), "text": text})

        segments.extend(got)
        try:
            cached.write_text(
                json.dumps({"language": (raw.get("language") or "").strip().lower(),
                            "segments": got}, ensure_ascii=False),
                encoding="utf-8")
        except Exception:
            pass    # a cache that cannot be written must not fail the run
        done += 1

    detected = max(heard, key=heard.get) if heard else (language or "")
    detected = _LANG_CODE.get(detected, detected)

    segments = _repair(client, audio, model, detected, segments, parts, progress)
    segments.sort(key=lambda s: s["start"])
    segments = _merge(_clean(segments, duration))
    for i, s in enumerate(segments):
        s["id"] = i
    return segments, detected


# A segment this long is already an outlier - real lines sit near two seconds
# and the 99th percentile of a clean transcript is under four.
DEGENERATE_SPAN = 8.0
# Whisper's failure mode is not silence but collapse: it emits a stream of
# one- and two-letter stubs with the accents stripped off ("Kh bi ng t T gia
# c vi g Th"). Measured over a real 1400-line transcript, normal lines sit at
# 0.17 by this ratio and collapsed ones at 0.71 to 0.90, so the two groups do
# not overlap anywhere near this threshold.
DEGENERATE_FRAG = 0.55
MIN_WORDS = 6            # below this the ratio is too noisy to judge
MAX_REPAIRS = 40         # a wholly broken transcript must not cost a fortune


def _fragment_ratio(text: str) -> float:
    words = text.split()
    if len(words) < MIN_WORDS:
        return 0.0
    return sum(1 for w in words if len(w) <= 2) / len(words)


def _degenerate(seg: dict) -> bool:
    """Did Whisper lose the thread on this segment rather than transcribe it?"""
    return (seg["end"] - seg["start"] >= DEGENERATE_SPAN
            and _fragment_ratio(seg["text"]) > DEGENERATE_FRAG)


def _repair(client, audio: Path, model: str, language: str,
            segments: list[dict], parts: Path, progress=None) -> list[dict]:
    """Re-transcribe the spans Whisper collapsed on, one at a time.

    A collapsed segment is a stretch of real dialogue that ends up untranslated
    and therefore silent in the dub - ten seconds of a character visibly
    speaking with no voice. Fed that span on its own, and told the language it
    already detected, Whisper recovers it: in testing each collapsed 11 second
    line came back as five to eight clean ones.
    """
    broken = [s for s in segments if _degenerate(s)][:MAX_REPAIRS]
    if not broken:
        return segments

    fixed: list[dict] = [s for s in segments if not _degenerate(s)]
    for n, seg in enumerate(broken, 1):
        if progress:
            progress(f"Re-transcribing unclear audio {n}/{len(broken)}", 40)
        # a little air either side so the first and last word are not clipped
        start = max(0.0, seg["start"] - 0.3)
        span = (seg["end"] - seg["start"]) + 0.6
        cached = parts / f"fix_{_part_key(model, language, start, span)}.json"

        got: list[dict] | None = None
        if cached.is_file():
            try:
                got = json.loads(cached.read_text(encoding="utf-8"))["segments"]
            except Exception:
                cached.unlink(missing_ok=True)

        if got is None:
            try:
                part = slice_audio(audio, parts / f"fix_{n:03d}{ASR_EXT}", start, span)
                kwargs = dict(file=(part.name, part.read_bytes()), model=model,
                              response_format="verbose_json",
                              timestamp_granularities=["segment"], temperature=0.0)
                if language and language != "auto":
                    kwargs["language"] = language
                resp = client.audio.transcriptions.create(**kwargs)
                raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
                got = [{"start": round(float(x["start"]) + start, 3),
                        "end": round(float(x["end"]) + start, 3),
                        "text": (x.get("text") or "").strip()}
                       for x in (raw.get("segments") or [])
                       if (x.get("text") or "").strip()]
                part.unlink(missing_ok=True)
                cached.write_text(json.dumps({"segments": got}, ensure_ascii=False),
                                  encoding="utf-8")
            except Exception:
                got = None

        # Recovering nothing is not a reason to lose the slot - a bad line is
        # still better than a silent one, and the translator can drop it.
        fixed.extend(got if got else [seg])

    return fixed


_HALLUCINATIONS = {
    "字幕由amara.org社群提供", "字幕由Amara.org社群提供", "由 Amara.org 社群提供的字幕",
    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目", "謝謝觀看", "谢谢观看",
    "MING PAO CANADA", "Thanks for watching!", "感謝您的收看",
}


MERGE_MAX_SPAN = 7.0     # longest merged segment, seconds
MERGE_MAX_GAP = 0.40     # only join across pauses shorter than this
MERGE_MAX_CHARS = 44     # keep subtitles readable in two lines
_CJK = re.compile(r"[㐀-鿿぀-ヿ가-힯]")
_SENT_END = re.compile(r"[。！？!?…]\s*$")


def _merge(segs: list[dict]) -> list[dict]:
    """Join Whisper's short fragments into sentence-sized lines.

    Whisper splits dense speech into one- and two-second pieces. Those make
    impossible dubbing targets - English simply cannot say a whole clause in
    1.3 seconds - so adjacent fragments of the same sentence are combined into
    one line with a realistic window. It also makes the subtitles readable.
    """
    out: list[dict] = []
    for s in segs:
        if out:
            prev = out[-1]
            gap = s["start"] - prev["end"]
            span = s["end"] - prev["start"]
            if (gap <= MERGE_MAX_GAP
                    and span <= MERGE_MAX_SPAN
                    and len(prev["text"]) + len(s["text"]) <= MERGE_MAX_CHARS
                    and not _SENT_END.search(prev["text"])):
                joiner = "" if (_CJK.search(prev["text"]) or _CJK.search(s["text"])) else " "
                prev["text"] = prev["text"] + joiner + s["text"]
                prev["end"] = s["end"]
                continue
        out.append(dict(s))
    return out


def _clean(segs: list[dict], duration: float) -> list[dict]:
    """Drop Whisper's known filler artefacts and fix overlapping timestamps."""
    out: list[dict] = []
    for s in segs:
        t = s["text"].strip()
        flat = re.sub(r"[\s,.。，、!！?？]", "", t)
        if not flat or flat in {re.sub(r"[\s,.。，、!！?？]", "", h) for h in _HALLUCINATIONS}:
            continue
        # a single character repeated for the whole line is never real speech
        if len(set(flat)) == 1 and len(flat) > 6:
            continue
        s["text"] = t
        s["start"] = max(0.0, min(s["start"], duration))
        s["end"] = max(s["start"] + 0.2, min(s["end"], duration))
        if out and s["start"] < out[-1]["end"]:
            s["start"] = out[-1]["end"]
            if s["end"] <= s["start"] + 0.15:
                s["end"] = s["start"] + 0.4
        out.append(s)
    return out
