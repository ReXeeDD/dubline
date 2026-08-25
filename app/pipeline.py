"""End-to-end dubbing pipeline: video in, English-dubbed video out.

Every stage checkpoints to the video's work/ folder, so a run that fails - a
dropped connection, an expired key, a full disk - resumes from the last thing
that finished instead of starting again from the upload. See `_state`.
"""
from __future__ import annotations

import hashlib
import json
import queue
import shutil
import threading
import time
import traceback
import wave
from pathlib import Path

import numpy as np
from groq import Groq

from . import asr, hlsout, library, llm, subtitles, translate, tts, voiceclone
from .media import SR, ASR_EXT, extract_audio, media_info, mux, thumbnail

STATE_FILE = "stage.json"


def _state(work: Path) -> dict:
    """What the previous run got through, as recorded in work/stage.json."""
    try:
        return json.loads((work / STATE_FILE).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _mark(work: Path, stage: str, stamp: dict) -> None:
    """Record that `stage` finished, along with the inputs it finished for."""
    data = _state(work)
    data[stage] = stamp
    try:
        (work / STATE_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass    # bookkeeping must never be the thing that fails a run


def _clear(work: Path, *stages: str) -> None:
    data = _state(work)
    for s in stages:
        data.pop(s, None)
    try:
        (work / STATE_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _digest(*parts) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _audio_ready(work: Path) -> Path | None:
    """The extracted audio from a previous run, if it is complete and readable."""
    p = work / f"audio{ASR_EXT}"
    try:
        if p.is_file() and p.stat().st_size > 4096 and media_info(p)["duration"] > 0.5:
            return p
    except Exception:
        pass
    return None


def _checkpointer(vid: str, every: float = 8.0):
    """Save translated lines as they arrive, but not on every single batch.

    Writing 1500 segments to disk costs more than a batch takes to translate, so
    the saves are spaced out. The worst case is losing the last few seconds.
    """
    last = [0.0]

    def save(segments: list[dict], force: bool = False) -> None:
        now = time.monotonic()
        if force or now - last[0] >= every:
            last[0] = now
            library.save_segments(vid, segments)

    return save


def make_asr_client(api_key: str) -> Groq:
    """Transcription is Groq-only - no local Whisper is installed."""
    if not api_key:
        raise RuntimeError(
            "No Groq API key set. Open Settings in the app and paste your key "
            "(get one free at https://console.groq.com/keys).")
    return Groq(api_key=api_key)


def _reporter(vid: str):
    def report(stage: str, pct: int) -> None:
        library.update(vid, stage=stage, progress=max(0, min(100, int(pct))))
    return report


def process(vid: str, opts: dict) -> None:
    """Run the full pipeline for one library entry. Safe to call in a thread."""
    folder = library.vdir(vid)
    work = folder / "work"
    work.mkdir(exist_ok=True)
    progress = _reporter(vid)

    try:
        library.update(vid, status="processing", error=None)
        src = library.source_file(vid)
        if src is None:
            raise RuntimeError("Source video file is missing from the library.")

        # ---------------------------------------------------------- probe ---
        progress("Reading video", 3)
        info = media_info(src)
        if not info["has_audio"]:
            raise RuntimeError("This video has no audio track to translate.")
        duration = info["duration"]
        library.update(vid, duration=duration)

        thumb = folder / "thumb.jpg"
        if not thumb.exists():
            thumbnail(src, thumb, at=min(5.0, duration * 0.15))

        # -------------------------------------------------------- extract ---
        audio = _audio_ready(work)
        if audio is None:
            progress("Extracting audio", 7)
            audio = extract_audio(src, work / f"audio{ASR_EXT}")

        # ------------------------------------------------------------ asr ---
        # The transcript is reused only if it was made from this audio with the
        # same model and language - change either and it has to be redone.
        stamp = _digest(opts["asr_model"], opts["source_language"],
                        audio.stat().st_size)
        segments = library.load_segments(vid) if _state(work).get("asr") == stamp else []
        segments = [s for s in segments if s.get("text", "").strip()]

        if not segments:
            progress("Transcribing audio", 10)
            segments, heard = asr.transcribe(
                make_asr_client(opts["groq_api_key"]), audio,
                opts["asr_model"], opts["source_language"], work, progress)
            if not segments:
                raise RuntimeError(
                    "No speech was detected in the audio. If the video is quiet or "
                    "music-only there is nothing to dub.")
            library.save_segments(vid, segments)
            _mark(work, "asr", stamp)
            _mark(work, "lang", heard)
        # With source_language on "auto" this is the only place the real
        # language is ever named, and the translator's pronoun handling is
        # language-specific - so it must reach translate(), not "auto".
        lang = _state(work).get("lang") or opts["source_language"]
        library.update(vid, line_count=len(segments), source_lang=lang)

        # --------------------------------------------- translate + speak ---
        # Both stages run a window at a time and publish as they go, so the
        # video becomes watchable from the start long before the end of it has
        # been translated. Lines already carrying an `en` value are skipped, so
        # an interrupted run only pays for what is still missing.
        clients = llm.make_translation_clients(opts)
        dub, stats = _stream_dub(vid, src, segments, duration, opts, work,
                                 folder, progress, lang, clients)
        library.save_segments(vid, segments)

        _finish(vid, src, segments, duration, opts, work, folder, progress,
                dub=dub, stats=stats)

    except Exception as exc:
        library.update(vid, status="failed", error=f"{exc}",
                       stage="Failed: " + str(exc)[:180])
        (work / "error.log").write_text(traceback.format_exc(), encoding="utf-8")


def revoice(vid: str, opts: dict) -> None:
    """Re-run only speech synthesis and muxing - reuses existing translations.

    Used when the voice is changed or subtitle text is edited in the UI.
    """
    folder = library.vdir(vid)
    work = folder / "work"
    work.mkdir(exist_ok=True)
    progress = _reporter(vid)

    try:
        library.update(vid, status="processing", error=None)
        src = library.source_file(vid)
        segments = library.load_segments(vid)
        if src is None or not segments:
            raise RuntimeError("Nothing to re-voice - run the full pipeline first.")

        # Rendered lines are named after their text and voice settings, so
        # anything left from an interrupted run is reused where it still
        # applies and ignored where it does not. Only the finished track has
        # to be thrown away.
        _clear(work, "dub")
        duration = media_info(src)["duration"]
        progress("Preparing new voice", 70)
        _finish(vid, src, segments, duration, opts, work, folder, progress)

    except Exception as exc:
        library.update(vid, status="failed", error=f"{exc}",
                       stage="Failed: " + str(exc)[:180])


def retranslate(vid: str, opts: dict) -> None:
    """Translate again from the saved transcript, then re-voice and re-mux.

    Skips upload, extraction and transcription, so improvements to the
    translation prompt can be applied to an existing video without paying for
    Whisper again or waiting for it.
    """
    folder = library.vdir(vid)
    work = folder / "work"
    work.mkdir(exist_ok=True)
    progress = _reporter(vid)

    try:
        library.update(vid, status="processing", error=None)
        src = library.source_file(vid)
        segments = library.load_segments(vid)
        if src is None or not segments:
            raise RuntimeError("No transcript to re-translate - run the full "
                               "pipeline first.")

        # This is an explicit "do it again", so the existing English is dropped
        # rather than resumed, and the cast list is rebuilt in case the prompt
        # that produces it has changed.
        for s in segments:
            s["en"] = ""
        (work / "glossary.txt").unlink(missing_ok=True)
        _clear(work, "dub")

        progress(f"Translating {len(segments)} lines to English", 20)
        save = _checkpointer(vid)
        clients = llm.make_translation_clients(opts)
        segments = translate.translate(
            clients[0], segments,
            llm.translation_model(opts),
            _state(work).get("lang") or opts["source_language"], progress,
            models=(opts.get("llm_helpers") or [])
            if opts.get("llm_provider") != "local" else None,
            clients=clients, checkpoint=save, cache_dir=work)
        lang = _state(work).get("lang") or opts["source_language"]
        helpers = (opts.get("llm_helpers") or []) \
            if opts.get("llm_provider") != "local" else []
        translate.polish(clients, segments,
                         [llm.translation_model(opts)] + helpers, lang,
                         translate.cast_list(clients[0], segments,
                                             llm.helper_model(opts), lang, work),
                         progress)
        save(segments, force=True)

        duration = media_info(src)["duration"]
        _finish(vid, src, segments, duration, opts, work, folder, progress)

    except Exception as exc:
        library.update(vid, status="failed", error=f"{exc}",
                       stage="Failed: " + str(exc)[:180])


def remix(vid: str, opts: dict) -> None:
    """Re-mux the existing dub track onto the video with new audio options.

    Reuses work/dub.wav, so changing the background-audio setting costs one
    encode instead of re-synthesising every line.
    """
    folder = library.vdir(vid)
    work = folder / "work"
    progress = _reporter(vid)

    try:
        src = library.source_file(vid)
        dub = work / "dub.wav"
        if src is None or not dub.exists():
            raise FileNotFoundError("no existing dub track")

        library.update(vid, status="processing", error=None)
        progress("Re-mixing audio", 40)

        out = folder / "dubbed.mp4"
        tmp = folder / "dubbed.tmp.mp4"
        mux(src, dub, tmp,
            original=src if opts.get("keep_original_audio") else None,
            bed_gain=float(opts.get("original_audio_gain", 0.10)),
            subs=(folder / "english.srt") if opts.get("burn_subtitles") else None)
        out.unlink(missing_ok=True)
        tmp.replace(out)

        stats = dict(library.get(vid).get("stats") or {})
        stats["keep_original_audio"] = bool(opts.get("keep_original_audio"))
        library.update(vid, status="ready", progress=100, stage="Ready", stats=stats)

    except FileNotFoundError:
        revoice(vid, opts)          # no dub track cached - rebuild it
    except Exception as exc:
        library.update(vid, status="failed", error=f"{exc}",
                       stage="Failed: " + str(exc)[:180])


def _window_bounds(segments: list[dict], total: float,
                   first: float = 45.0, cap: float = 300.0) -> list[tuple]:
    """Split the video into (lo, hi, until) windows to dub one at a time.

    Windows start short and grow. The first one only has to be long enough to
    give the viewer somewhere to start, and every second spent on it is a second
    they are still waiting; later windows are long because per-window overhead
    is paid once each and by then playback is already running ahead of them.
    """
    out: list[tuple[int, int, float]] = []
    t, size, i = 0.0, first, 0
    while t < total - 1e-3:
        until = min(total, t + size)
        j = i
        while j < len(segments) and segments[j]["start"] < until:
            j += 1
        out.append((i, j, until))
        i, t, size = j, until, min(cap, size * 2)
    return out


def _stream_dub(vid: str, src: Path, segments: list[dict], duration: float,
                opts: dict, work: Path, folder: Path, progress, lang: str,
                clients: list) -> tuple[Path, dict]:
    """Translate, speak and publish the dub window by window, in order.

    Translation runs one window ahead of speech, so the two stages overlap
    instead of queueing. Everything is fed to a live HLS stream as it is
    finished, which is what lets the video be watched from the start while the
    end of it has not been translated yet.
    """
    stream = hlsout.LiveStream(folder, src, duration)
    progress("Preparing the stream", 42)
    stream.start()          # picture is segmented here - about a second

    # One cast list for the whole video, settled before the first line is
    # translated. Building it per window would let the same character be
    # spelled differently in different parts of the same episode.
    glossary = translate.cast_list(clients[0], segments,
                                   llm.helper_model(opts), lang, work)
    system = translate.build_system(lang, glossary)

    bounds = _window_bounds(segments, duration)
    mixer = tts.StreamMixer(segments, duration, opts["voice"], work,
                            max_speedup=float(opts.get("max_speedup", 1.7)),
                            pitch=int(opts.get("pitch", 0)),
                            speed=int(opts.get("speed", 0)),
                            volume=int(opts.get("volume", 0)))

    ready: queue.Queue = queue.Queue(maxsize=1)
    failure: list[BaseException] = []

    def early(stage: str, pct: int) -> None:
        """Report translation only until there is something to watch.

        Once playback can start, the consumer's "watchable up to" message is
        the useful one and this would fight it for the same field. Before then
        this is the only sign of life there is - and a rate-limit backoff can
        hold the first window for minutes, which without this looks frozen.
        """
        if stream.ready_seconds <= 0:
            progress(stage, min(pct, 44))

    # Shared by every window. These carry what each model's remaining budget
    # actually is, and that knowledge has to outlive a single window.
    budgets: dict = {}

    def translate_ahead() -> None:
        try:
            for lo, hi, until in bounds:
                if hi > lo:
                    helpers = (opts.get("llm_helpers") or []) \
                        if opts.get("llm_provider") != "local" else []
                    translate.translate(
                        clients[0], segments, llm.translation_model(opts), lang,
                        early, models=helpers or None,
                        clients=clients, window=(lo, hi), system=system,
                        budgets=budgets)
                    # Read back what came out and re-ask for the lines that are
                    # plainly wrong, while this window is still ahead of the
                    # speech stage and nothing has been spoken yet.
                    translate.polish(
                        clients, segments,
                        [llm.translation_model(opts)] + helpers, lang,
                        glossary, early, budgets=budgets, window=(lo, hi))
                ready.put((lo, hi, until))
        except BaseException as exc:       # hand it to the consumer to raise
            failure.append(exc)
        finally:
            ready.put(None)

    worker = threading.Thread(target=translate_ahead, daemon=True)
    worker.start()

    wav = _WavAppender(work / "dub.wav")
    clone = voiceclone.is_clone(opts.get("voice", ""))
    try:
        while True:
            item = ready.get()
            if item is None:
                break
            lo, hi, until = item
            pcm = mixer.render(lo, hi, until)
            if clone:
                pcm = _convert_window(pcm, opts, work, progress)
            stream.feed(pcm)
            wav.write(pcm)
            library.save_segments(vid, segments)
            done, pct = until, 45 + int(50 * until / max(1.0, duration))
            progress(f"Watchable up to {_clock(done)} of {_clock(duration)}", pct)

        if failure:
            raise failure[0]

        tail = mixer.drain()
        if tail.size:
            stream.feed(tail)
            wav.write(tail)
        stream.finish()
    except BaseException:
        stream.abandon()
        raise
    finally:
        wav.close()

    stats = dict(mixer.stats)
    stats["max_speedup_used"] = round(stats["max_speedup_used"], 2)
    if clone:
        stats["voice_clone"] = opts["voice"]
        stats["soften"] = int(opts.get("soften", 0))
    return work / "dub.wav", stats


def _convert_window(pcm, opts: dict, work: Path, progress):
    """Re-timbre one window with the cloned voice, keeping its length exact."""
    from .media import decode_pcm, write_wav
    raw, out = work / "win_raw.wav", work / "win_clone.wav"
    write_wav(pcm, raw)
    try:
        voiceclone.convert(raw, out, opts["voice"], opts, None)
        got = decode_pcm(out)
        # Conversion is length-preserving, but a sample either way here would
        # shift everything after it, so the window is pinned to its own size.
        if len(got) < len(pcm):
            got = np.pad(got, (0, len(pcm) - len(got)))
        return got[:len(pcm)]
    except Exception as exc:
        progress(f"Voice conversion skipped: {exc}", 90)
        return pcm
    finally:
        raw.unlink(missing_ok=True)
        out.unlink(missing_ok=True)


class _WavAppender:
    """Writes the dub track out as it is produced, for the downloadable mp4."""

    def __init__(self, path: Path):
        self.w = wave.open(str(path), "wb")
        self.w.setnchannels(1)
        self.w.setsampwidth(2)
        self.w.setframerate(SR)

    def write(self, pcm) -> None:
        self.w.writeframes((np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())

    def close(self) -> None:
        try:
            self.w.close()
        except Exception:
            pass


def _clock(seconds: float) -> str:
    m, s = divmod(int(max(0.0, seconds)), 60)
    return f"{m}:{s:02d}"


def _dub_stamp(segments: list[dict], opts: dict) -> str:
    """Everything the finished dub track depends on, in one value."""
    lines = "\x1f".join(f"{s['id']}:{s['start']}:{s.get('en', '')}" for s in segments)
    return _digest(opts.get("voice"), opts.get("pitch", 0), opts.get("speed", 0),
                   opts.get("volume", 0), opts.get("max_speedup", 1.7),
                   opts.get("soften", 0), opts.get("clone_index_rate"),
                   opts.get("clone_protect"), _digest(lines))


def _finish(vid: str, src: Path, segments: list[dict], duration: float,
            opts: dict, work: Path, folder: Path, progress,
            dub: Path | None = None, stats: dict | None = None) -> None:
    """Shared tail of both pipelines: TTS, alignment, subtitles, mux.

    `dub` and `stats` are passed in by the streaming pipeline, which has
    already built the track window by window; without them the track is built
    here in one pass, which is what re-voicing an existing video does.
    """
    # ------------------------------------------------------------- speech ---
    # A finished dub track is worth a lot: for a cloned voice it carries the
    # GPU conversion pass, which is the single most expensive step in the whole
    # pipeline. If the last run got that far and nothing it depends on has
    # changed, the encode is the only thing left to redo.
    stamp = _dub_stamp(segments, opts)
    if dub is not None:
        _mark(work, "dub", {"stamp": stamp, "size": dub.stat().st_size,
                            "stats": stats or {}})
        stats = dict(stats or {})
        reuse = True
    else:
        dub = work / "dub.wav"
        prev = _state(work).get("dub") or {}
        reuse = (prev.get("stamp") == stamp and dub.is_file()
                 and prev.get("size") == dub.stat().st_size)
        if reuse:
            stats = dict(prev.get("stats") or {})

    if reuse:
        progress("Preparing the download", 93)
    else:
        dub, stats = tts.build_dub_track(
            segments, opts["voice"], work, duration,
            max_speedup=float(opts.get("max_speedup", 1.7)), progress=progress,
            pitch=int(opts.get("pitch", 0)), speed=int(opts.get("speed", 0)),
            volume=int(opts.get("volume", 0)))
        library.save_segments(vid, segments)

        # --------------------------------------------------- cloned voice ---
        # Conversion returns audio of exactly the same length, so this cannot
        # disturb the alignment established above. work/dub.wav keeps the
        # converted track, letting the audio mix be changed later without
        # re-converting.
        if voiceclone.is_clone(opts.get("voice", "")):
            converted = work / "dub_clone.wav"
            try:
                voiceclone.convert(dub, converted, opts["voice"], opts, progress)
                converted.replace(dub)
                stats["voice_clone"] = opts["voice"]
                stats["soften"] = int(opts.get("soften", 0))
            except Exception as exc:
                # A missing toolchain must not throw away a finished dub - ship
                # the base voice and say so.
                stats["voice_clone_error"] = str(exc)[:200]
                progress(f"Voice conversion skipped: {exc}", 94)

        _mark(work, "dub", {"stamp": stamp, "size": dub.stat().st_size,
                            "stats": stats})

    # ---------------------------------------------------------- subtitles ---
    progress("Writing subtitle files", 94)
    subtitles.write_srt(segments, folder / "english.srt", "en")
    subtitles.write_vtt(segments, folder / "english.vtt", "en")
    subtitles.write_srt(segments, folder / "original.srt", "text")
    subtitles.write_vtt(segments, folder / "original.vtt", "text")

    # --------------------------------------------------------------- mux ---
    progress("Encoding the dubbed video", 96)
    out = folder / "dubbed.mp4"
    tmp = folder / "dubbed.tmp.mp4"
    mux(src, dub, tmp,
        original=src if opts.get("keep_original_audio") else None,
        bed_gain=float(opts.get("original_audio_gain", 0.10)),
        subs=(folder / "english.srt") if opts.get("burn_subtitles") else None)
    out.unlink(missing_ok=True)
    tmp.replace(out)

    # ------------------------------------------------------------ tidy up ---
    for f in work.glob("tts_*.mp3"):
        f.unlink(missing_ok=True)
    for f in work.glob(f"asr_*{ASR_EXT}"):
        f.unlink(missing_ok=True)
    # The per-chunk transcripts only exist to survive a failure. The finished
    # transcript is in the library now, so they have nothing left to protect.
    shutil.rmtree(work / "asr_parts", ignore_errors=True)
    # Same for the stream: it exists so the video can be watched before it is
    # finished, and dubbed.mp4 now does that job better. Keeping it would cost
    # roughly another copy of the video per title - about 140 MB an hour.
    shutil.rmtree(folder / "hls", ignore_errors=True)
    # work/dub.wav is deliberately kept so the audio mix can be changed later
    # without re-synthesising every line
    shutil.rmtree(work / "__pycache__", ignore_errors=True)

    stats["translated_lines"] = sum(1 for s in segments if s.get("en", "").strip())
    stats["pitch"] = int(opts.get("pitch", 0))
    stats["speed"] = int(opts.get("speed", 0))
    stats["volume"] = int(opts.get("volume", 0))
    stats["keep_original_audio"] = bool(opts.get("keep_original_audio"))
    library.update(vid, status="ready", progress=100, stage="Ready",
                   voice=opts["voice"], line_count=len(segments), stats=stats)
