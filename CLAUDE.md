# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Dubline: a local FastAPI + vanilla-JS web app that turns a foreign-language
video into an English-dubbed one. Whisper and the translation LLM are hosted on
Groq; speech is Microsoft Edge TTS; everything else (ffmpeg, sqlite, optional
RVC voice conversion) runs on the machine. `README.md` is unusually detailed and
carries the measurements behind most design decisions — read it before changing
sync, batching or streaming behaviour.

## Commands

```bash
python setup.py          # downloads ffmpeg/ffprobe + yt-dlp into bin/, verifies deps
python run.py --open     # serve on http://localhost:8000 and open a browser
python run.py --port 8010 --lan   # other port; --lan also serves the network
python selftest.py       # full offline test suite
```

Windows one-click: `start.bat` builds `.venv\`, installs `requirements.txt`,
runs `setup.py`, then `run.py --open`. `.claude/launch.json` defines a `dubline`
preview config on port 8010.

### Tests

`selftest.py` is a single hand-rolled script, not pytest — there is no test
runner, no discovery and no `-k`. It builds a synthetic clip, dubs scripted
lines, and asserts timing/resume/streaming/library invariants. It needs internet
for Edge TTS but **no API key**. `check(label, passed, detail)` accumulates into
a module-level `ok`; `main()` exits non-zero if anything failed.

Each numbered section is a self-contained module-level function, so a single
group can be run on its own:

```bash
python -c "import selftest; selftest._stream_checks()"
```

Sections: `_resume_checks` (7), `_quality_checks` (8), `_stream_checks` (9),
`_dub_quality_checks` (10), `_library_checks` (11), `_download_checks` (12).
Sections 1–6 (synthetic video, voices, synthesis, alignment, subtitles, mux)
only exist inline in `main()`.

`tools/` holds one-off measurement scripts (`bench_translate.py`,
`concurrency.py`, `keytest.py`, `paralleltest.py`) that produced the constants
in `translate.py` and `tts.py`. They hit the real API and cost tokens.

## Architecture

Request path: `run.py` → uvicorn → `app/main.py` (FastAPI, serves `static/` and
the JSON API) → `POOL.submit(pipeline.process, ...)`.

**One dubbing job at a time.** `main.py` uses a `ThreadPoolExecutor(max_workers=1)`
so ffmpeg and Whisper uploads do not fight each other; downloads get a separate
pool. On startup, anything left `processing`/`queued` is re-queued rather than
marked failed — resumption is cheap because every stage is checkpointed.

Module roles (`app/`):

| module | role |
|---|---|
| `config.py` | on-disk layout, settings + API-key resolution, ffmpeg/ffprobe lookup |
| `media.py` | every ffmpeg/ffprobe invocation, PCM decode, sample-accurate `place()`, `mux()` |
| `asr.py` | Groq Whisper with silence-aware chunking, overlap merge, degenerate-segment repair |
| `llm.py` | one call surface over Groq and any OpenAI-compatible local server (LM Studio) |
| `translate.py` | batching, `TokenBudget` pacing, cast list, `polish()` repair pass |
| `tts.py` | edge-tts synthesis, rate fitting, `build_dub_track()` and `StreamMixer` |
| `hlsout.py` | `LiveStream` — watchable output while the dub is still being made |
| `voiceclone.py` | optional RVC re-timbre via a warm subprocess in a separate venv |
| `subtitles.py` | SRT/VTT writers |
| `library.py` | sqlite metadata + `segments.json` I/O |
| `pipeline.py` | orchestrates all of the above, owns the checkpoint format |
| `download.py` | yt-dlp probe/fetch; downloading never starts a dub |

### The segment dict

One list of dicts is the spine of the whole pipeline, saved to
`data/library/<id>/segments.json` and mutated in place by each stage:

```python
{"id": int, "start": float, "end": float, "text": str, "en": str}
```

`text` is the source transcript, `en` the translation. **`en` being non-empty is
the resume signal** — `translate()` skips any line that already has one, which
is why re-translating requires explicitly clearing it (see `retranslate()`).

### Entry points into pipeline.py

`process` (full run) · `revoice` (reuse translations, re-synthesise) ·
`retranslate` (reuse transcript, clear `en`, translate again) · `remix` (reuse
`work/dub.wav`, one encode). Each is a whole job submitted to `POOL`; they all
converge on `_finish()` for TTS-if-needed, subtitles and mux.

### Checkpointing

`work/stage.json` records which stages completed, each against a `_digest()`
**stamp of the inputs it completed for** (`_state` / `_mark` / `_clear`).
A stage is reused only when its stamp still matches — so changing the ASR model
invalidates the transcript, and changing voice/pitch/speed/max_speedup or any
line's text invalidates the dub track (`_dub_stamp`). When adding an option that
affects rendered audio, add it to `_dub_stamp` or stale audio will be served.

Per-stage granularity: ASR resumes per chunk (`work/asr_parts`), translation per
line (via `en`), TTS per line (cached by a fingerprint of text **and** voice
settings), the dub track as a whole.

### Streaming (the non-obvious part)

`_stream_dub()` runs translation one window ahead of speech on a background
thread, handing windows over a `queue.Queue(maxsize=1)`. Windows start at 45 s
and double to a 300 s cap (`_window_bounds`).

Two invariants that the self-test enforces and that are easy to break:

1. **One long-running audio encoder.** `hlsout.LiveStream` pipes every window
   into a single ffmpeg process. Encoding windows independently accumulates AAC
   priming delay (+0.427 s over 20 min — audible desync). Video is segmented
   once with `-c:v copy` and never re-touched.
2. **Batches sit on one grid across the whole transcript**, not per window.
   Otherwise a window boundary creates a stub batch that pays a full system
   prompt for two lines. Streaming must cost exactly the same request count as
   dubbing in one pass.

### Rate limits

Groq meters tokens per minute **per model** and also per day. `TokenBudget`
(one per model, shared across that model's streams, lock-protected) reserves
before sending and reconciles from `x-ratelimit-*` headers. A per-minute limit
is waited out; a per-day limit raises `Exhausted` and that model is dropped from
the pool so the others carry on — the app must never appear to freeze. Extra
models multiply throughput; extra keys from the *same* Groq account do not
(limits are per organization).

### Voice cloning invariant

RVC conversion is length-preserving by contract — it runs *after* all timing
work and must never change sample count. `_convert_window()` pads/truncates to
be certain, and the self-test compares sample counts and fails on any
difference. A missing or broken Applio toolchain must degrade to the base voice
and record the reason in stats, never fail the run.

## Conventions

- **Nothing is written outside the project folder.** `config.py` repoints
  `tempfile.tempdir`, `TMP`/`TEMP`/`TMPDIR`, `HF_HOME` and `XDG_CACHE_HOME` into
  `data/`; `start.bat` does the same for pip. The user's C: drive is full — new
  code that caches or spools must keep to `data/`.
- **Binaries resolve through `config.ffmpeg()` / `ffprobe()`**, which prefer
  `bin/` over PATH. Never call `"ffmpeg"` directly; on Windows also pass
  `media._NOWIN` to `subprocess` so no console window flashes.
- **All ffmpeg work belongs in `media.py`.** Other modules call its helpers.
- Comments in this codebase explain *why*, usually with the measurement that
  settled it. Match that when touching tuned constants — don't change
  `CONCURRENCY`, `BATCH`, `STREAMS_PER_MODEL`, the breath timings or the window
  sizes without a reason of the same kind.
- New settings go in `config.DEFAULTS`; `save_settings()` ignores unknown keys,
  so a key missing there is silently dropped.
- A pipeline stage failing writes `status="failed"` plus the traceback to
  `work/error.log`; bookkeeping (`_mark`) is best-effort and must never itself
  fail a run.
- Frontend is dependency-free vanilla JS (`static/app.js`, hash router near the
  bottom) with `hls.min.js` vendored. `main.py` serves `static/` through
  `FreshStatic`, which forces `Cache-Control: no-cache` so an updated `app.js`
  is never skipped by the browser.
- `data/`, `bin/`, `.venv/` and `settings.json`/`credentials.json` are
  gitignored. The Groq key lives in `data/settings.json` with a backup in
  `data/cache/credentials.json`; lookup order is settings → `GROQ_API_KEY` env →
  cache.
- `library.set_position()` deliberately bypasses `update()` because `update()`
  stamps `updated_at`, which the player uses as its cache-buster — writing it
  during playback would keep changing the URL of the file being watched.
- New sqlite columns must be appended to `library.MIGRATIONS`;
  `CREATE TABLE IF NOT EXISTS` will not add them to an existing library.

## Source material

Typical input is Vietnamese/Chinese web series with first-person narration.
`translate.py` carries language-specific pronoun notes and builds a cast list
(names, genders, who narrates) before translating, because naive translation
turns inner monologue from "I" into "you". Leave `source_language` on `auto` —
detection is also what selects those pronoun notes.
