# Dubline

A local, YouTube-style web app that turns a foreign-language video into an
English-dubbed one — transcript, translation, new voice track, and a re-synced
video, all from a browser tab on your own machine.

```
video → audio → Chinese transcript with timestamps → English translation
      → English speech → time-aligned dub track → re-muxed video
```

Everything is written inside this folder on `D:`. Nothing touches `C:`.

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python setup.py
```

```bash
python run.py --open
```

Then open <http://localhost:8000>, click the gear icon, and paste a Groq API key
(free at <https://console.groq.com/keys>). Upload a video and it processes on its own.

On Windows you can just double-click `start.bat`.

---

## How the sync works

Losing lip/timing sync is the usual failure of DIY dubbing, because English
translations of Chinese are typically 20–40 % longer to say. Dubline handles it
per line rather than globally:

1. Whisper returns each line with a real start and end timestamp.
2. The translator is told the **character budget** for each line (≈15 chars per
   second of screen time) and asked to stay inside it.
3. Each line's window is its own slot **plus the silence that follows it**, so a
   long line borrows from the pause instead of colliding with the next speaker.
4. If it still will not fit, the line is spoken faster — first by re-synthesising
   at a higher speaking rate (natural, pitch-correct), then with `atempo` for the
   remainder, never exceeding the cap you set in Settings.
5. Every clip is mixed into a silent track at its **exact sample offset**, and
   the video stream is copied, not re-encoded, so the picture never shifts.

The self-test measures the result: alignment drift stays around 100–150 ms.

---

## What runs where

| Stage | Engine | Cost |
|---|---|---|
| Transcription | Groq `whisper-large-v3` | free tier |
| Translation | Groq LLM (Qwen etc.) **or** your local LM Studio | free tier / free |
| Speech | Microsoft Edge TTS, 47 English voices | free, no key |
| Cloned voice | local RVC model on your GPU | free, offline |
| Audio + video | bundled ffmpeg in `bin/` | free |

Only transcription strictly requires the Groq key. Translation can be pointed at
LM Studio in Settings (`http://localhost:1234/v1`), though small local models
translate noticeably worse than the hosted ones.

> Settings loads the **live model list** from your Groq account rather than
> hard-coding ids, so retired models never linger in the dropdown. The default is
> `qwen/qwen3.6-27b`.

## Cloned voices

The default voice, **Rei**, is a locally trained RVC model rather than a stock
Edge voice. It is voice *conversion*, not synthesis, and the distinction matters:

1. edge-tts speaks every line as usual, so the rate fitting, hard-anchored
   placement, pitch, pace and loudness settings all still apply;
2. the finished dub track is then passed through the model, which replaces the
   timbre and returns **audio of exactly the same length**.

Because step 2 cannot re-time anything, adding a cloned voice cannot introduce
drift. The self-test asserts this - it compares sample counts before and after
conversion and fails on any difference.

Conversion runs on the GPU under a separate virtualenv (torch + CUDA + Applio's
pinned dependencies) so none of that lands in this app's environment. Paths are
`clone_applio_dir` and `clone_python` in Settings. If the toolchain is missing
the dub still completes in the base voice and the reason is recorded in the
video's stats rather than failing the run.

Loading a model costs ~14 s, far more than converting with it, so the worker
process is kept warm between jobs and shuts down after 10 minutes idle to
release GPU memory. A first preview takes ~15 s; later ones take ~3 s.

Installed voices live in `data/voices/<id>/` as `model.pth`, `model.index` and
`voice.json`. Drop another folder in to add a second cloned voice - it appears
in the voice list automatically, under "Your cloned voices".

`voice.json` fields:

| field | meaning |
|---|---|
| `name` | shown in the UI |
| `base_voice` | the Edge voice that speaks before conversion |
| `index_rate` | retrieval strength, 0-1. Higher locks timbre, smears consonants |
| `protect` | shields unvoiced consonants from conversion artifacts |

### Softness

A model trained on little audio converts with broadband roughness - measurable
as spectral flatness, which roughly doubles from the edge-tts input to the
converted output. The **Softness** slider (cloned voices only) drives the
spectral cleaner inside the conversion.

It is calibrated, not guessed. Measured on the Rei model, against the reference
speaker:

| softness | roughness | speaker match | |
|---|---|---|---|
| 0 | 0.030 | 0.985 | raw model output |
| 35 | 0.025 | 0.983 | default - as clean as plain edge-tts |
| 70 | 0.018 | 0.980 | smoother, drifting from the voice |
| 100 | 0.015 | 0.977 | smooth but no longer really him |

There is a genuine trade here, so the UI states which band you are in rather
than hiding it. The default of 35 was chosen because it brings roughness down to
the level of ordinary edge-tts speech while speaker match is still 0.983.

Softness acts *inside* the conversion, so changing it re-converts: about three
seconds on a preview with the worker warm, a few minutes for a whole episode.

> A hand-written spectral post-filter was tried first - smoothing across time and
> frequency, noise-floor subtraction, a high shelf - and discarded. Measured at
> every setting it *raised* flatness rather than lowering it, because modifying
> STFT magnitudes against unmodified phase breaks reconstruction consistency and
> the overlap-add artifacts land as exactly the roughness being chased.

Quality is bounded by the training audio, not the training run. A model trained
on ~2 minutes is recognisable but strains on sounds the source never contained;
more clean speech is the single biggest improvement available.

### Rate limits shape the translation stage
>
> Groq's on-demand tier allows **8,000 tokens per minute**, and `max_tokens` is
> reserved against that budget when the request is made — so an over-large
> `max_tokens` alone can trigger a 413. Translation therefore runs in 20-line
> batches, paced against the live `x-ratelimit-*` headers, and any batch that is
> still refused is split in half and retried. Sustained throughput is about
> **150 subtitle lines per minute**.
>
> Reasoning is explicitly disabled (`reasoning_effort="none"`). Qwen3.6 otherwise
> spends the budget thinking *and* its chain of thought breaks strict JSON
> validation, failing the request outright.

---

## In the app

- **Library** — grid of every video, with live progress on anything processing.
- **Watch** — player with English/original subtitle toggle, and a switch between
  the English dub and the original audio that keeps your playback position.
- **Transcript** — follows along as it plays; click a line to jump there.
- **Edit** — fix any English line, save, and optionally re-dub with the fix.
- **Voice & audio** — change the dub voice or the background-audio mix. Both reuse
  the existing translation, so neither makes an API call. Changing only the mix
  re-muxes from the cached dub track and finishes in seconds.
- **Downloads** — dubbed MP4, English `.srt`, original-language `.srt`.

## Where your API key is stored

Settings shows both paths inline, with a tick next to whichever currently holds a key:

```
data\settings.json                 primary
data\cache\credentials.json        cached backup
```

Saving writes both. If `settings.json` is ever deleted or corrupted, the key is
restored from the cache automatically — you never have to paste it twice. Lookup
order is `settings.json` → `GROQ_API_KEY` environment variable → cached copy.

**Test key** validates against Groq before saving, so a mistyped key is caught
immediately instead of failing halfway through your first video. Both files stay
on D: inside this project folder.

## Settings worth knowing

- **Maximum speed-up** (default 1.7×) — the cap on compressing an over-long line.
  Lower it for calmer speech and slightly looser sync; raise it for tighter sync.
- **Keep original audio** (default **off**) — when enabled, the source is mixed
  quietly under the dub to preserve music and ambience. On a dialogue-heavy video
  that means hearing both languages at once, so it is off unless you turn it on.
  Changing it uses the **remix** path: the rendered dub track is reused, so it is
  one encode (well under a minute) rather than a full re-dub.
- **Burn subtitles** — draws subtitles permanently into the picture. Requires a
  full video re-encode, so it is much slower.

---

## Verifying it works

```bash
python selftest.py
```

Builds a synthetic clip, dubs scripted lines, and asserts that the audio lands on
the right timestamps, that over-long lines get compressed within the cap, that the
subtitle files are correct, and that the mux preserves duration and video stream.
Needs internet for Edge TTS, but no API key.

## Layout

```
app/       config, media (ffmpeg), asr, translate, llm, tts, subtitles,
           library (sqlite), pipeline, main (FastAPI)
static/    index.html, app.js, style.css
bin/       ffmpeg.exe, ffprobe.exe        (downloaded by setup.py)
data/      library/<id>/  source, dubbed.mp4, segments.json, srt/vtt, thumb
           library.db, settings.json, tmp/
```

## Troubleshooting

**"ffmpeg is not installed"** — run `python setup.py`.

**Processing fails right away** — no API key, or an expired model id. Open
Settings; if the model dropdown is empty your key is not being accepted.

**Rate limited on long videos** — Groq's free tier has per-minute limits. Audio is
chunked into 15-minute pieces; retry from the card's ⋮ menu, which restarts from
scratch but is safe to repeat.

**Upload size** — audio is sent as 16 kHz mono Opus at 32 kbps (~3.5 MB per
15-minute chunk, about a seventh of Groq's 25 MB cap). Whisper resamples to
16 kHz mono log-mel regardless, so lossless audio would cost ~10x the bytes for
no accuracy gain. Anything that still comes out oversized is split in half
automatically instead of failing.

**Dub sounds rushed** — lower Maximum speed-up in Settings and re-dub from the
watch page. The translator is already told to be concise, but some passages are
simply dense.

**Video has no speech** — music-only clips produce no segments and stop with a
clear message.
