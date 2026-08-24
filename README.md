# Dubline

A local, YouTube-style web app that turns a foreign-language video into an
English-dubbed one — transcript, translation, new voice track, and a re-synced
video, all from a browser tab on your own machine.

```
video → audio → transcript with timestamps → English translation
      → English speech → time-aligned dub track → re-muxed video
```

**You do not have to wait for it to finish.** The picture is published the
moment processing starts and the dub is streamed onto it as it is made, so a
one-hour episode is watchable from about 45 seconds in while the rest is still
being translated behind you.

Nothing is installed system-wide. Everything — the Python environment, ffmpeg,
your videos, temp files — stays inside this folder.

---

## Quick start

**Windows:** download or clone the repo and double-click **`start.bat`**.

The first run builds a private Python environment in `.venv\`, installs the
packages, downloads ffmpeg into `bin\`, and opens the app. It takes a few
minutes. Every run after that starts in a couple of seconds.

**Anything else:**

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
(free at <https://console.groq.com/keys>). Upload a video and it processes on
its own.

Add `--lan` to reach it from a phone or TV on the same network.

---

## What you need to download

Almost nothing — the two AI models that do the heavy lifting run on Groq's
servers, not yours.

| Thing | Size | How you get it |
|---|---|---|
| **ffmpeg + ffprobe** | ~180 MB | **Automatic.** `setup.py` downloads them into `bin\`. If it fails, grab [`ffmpeg-master-latest-win64-gpl.zip`](https://github.com/BtbN/FFmpeg-Builds/releases) and copy `ffmpeg.exe` and `ffprobe.exe` out of its `bin\` folder into this project's `bin\`. |
| **Groq API key** | — | Free at <https://console.groq.com/keys>. Paste it into Settings. This is the only thing you must supply by hand. |
| **Whisper + the translation LLM** | — | Hosted by Groq. Nothing to download. |
| **Speech voices** | — | Microsoft Edge TTS. 47 English voices, streamed, no key and no download. |
| **hls.js** | 414 KB | Already in the repo at `static/vendor/`, so live playback works offline. |
| **A cloned voice** *(optional)* | ~55 MB each | See below. Skip this and everything still works. |

### Optional: cloned voices

Dubline can re-timbre the finished dub with an [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
v2 model so the narration sounds like a specific voice instead of a stock one.
This needs an NVIDIA GPU and a separate toolchain, and is entirely opt-in.

1. Install **[Applio](https://github.com/IAHispano/Applio)** anywhere on your
   disk. It provides the conversion runtime and, if you want, trains a model
   from your own recordings.
2. Put the model where Dubline looks for it:

   ```
   data/voices/<id>/model.pth      the RVC v2 model
   data/voices/<id>/model.index    its retrieval index
   data/voices/<id>/voice.json     the description below
   ```

3. `voice.json`:

   ```json
   {
     "name": "Rei",
     "gender": "Male",
     "base_voice": "en-US-AndrewMultilingualNeural",
     "index_rate": 0.5,
     "protect": 0.33
   }
   ```

4. In Settings, point **Applio folder** and **Applio Python** at the install —
   e.g. `D:/voice-clone/applio` and `D:/voice-clone/venv/Scripts/python.exe`.

The folder appears in the voice list under "Your cloned voices" on the next
page load. Drop in more folders for more voices. If the toolchain is missing or
broken the dub still completes in the base voice, and the reason is recorded in
the video's stats rather than failing the run.

> Please only clone a voice you have the right to use.

---

## How the sync works

Losing lip/timing sync is the usual failure of DIY dubbing, because English
translations are typically 20–40 % longer to say than the original. Dubline
handles it per line rather than globally:

1. Whisper returns each line with a real start and end timestamp.
2. The translator is told the **character budget** for each line (≈15 chars per
   second of screen time) and asked to stay inside it — but never at the cost of
   grammar, which outranks the budget in the prompt.
3. Each line's window is its own slot **plus the silence that follows it**, so a
   long line borrows from the pause instead of colliding with the next speaker.
4. If it still will not fit, the line is spoken faster — first by re-synthesising
   at a higher speaking rate (natural, pitch-correct), then with `atempo` for the
   remainder, never exceeding the cap you set in Settings.
5. Every clip is mixed into a silent track at its **exact sample offset**, and
   the video stream is copied, not re-encoded, so the picture never shifts.

The self-test measures the result: alignment drift stays around 100–150 ms.

### Who is speaking

Video with a narrator breaks naively-translated dubs in a specific way: the
main character's inner monologue comes out addressed to *you* instead of spoken
as *I*. Before the first line is translated, Dubline asks the model for a cast
list — every named character, their gender, and which of them is the narrator —
and hands that to every batch along with a short note on the source language's
pronoun system. Vietnamese and Chinese wuxia pronouns (`ta`, `ngươi`, `nàng`,
`hắn`, `chàng`) carry information English has no direct equivalent for, so the
note spells out the mapping with a worked example.

Measured on 30 lines of first-person narration, that took wrong-person errors
from 7 down to 1.

---

## Watching while it works

Processing publishes an HLS stream as it goes, and the library card grows a
**Watch now** button as soon as there is anything to see.

* **The picture** never changes, so it is segmented once, straight from the
  source with `-c:v copy` — 0.94 s for a 54-minute episode. The whole video is
  seekable before any dubbing has happened.
* **The dub** is produced a window at a time (45 s, then 90, then doubling to a
  5-minute cap) and fed to **one long-running encoder**, which appends segments
  to a growing playlist.

Feeding a single continuous encoder is the part that matters. AAC carries
encoder priming at the start of every independent encode, so per-window encodes
accumulate — measured at +0.427 s over twenty minutes, which is visible
lip-sync error by the end of an hour. One stream measured +0.064 s over ten
minutes, all of it in the final flush.

When the run finishes, the HLS copy is deleted and the page reloads onto the
finished MP4, so live playback costs no permanent disk.

## Nothing is paid for twice

Every expensive stage is checkpointed, so a crash, a closed laptop or a rate
limit costs only the work that was actually in flight:

| Stage | Resumes at |
|---|---|
| Transcription | the chunk — finished chunks are read back from `work/asr_parts` |
| Translation | the batch — lines that already carry a translation are skipped |
| Speech | the line — cached by a fingerprint of its text *and* its voice settings |
| Dub track | the whole track, if the subtitles and voice settings are unchanged |

Retrying a failed video from the card's ⋮ menu picks up where it stopped. A
transcript that is already complete costs zero requests.

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
> hard-coding ids, so retired models never linger in the dropdown.

### What an hour of video costs

Groq's free tier meters requests *and* tokens, per model, per day. Measured on a
54-minute episode of 1,430 subtitle lines:

| | per hour of video | free-tier limit |
|---|---|---|
| LLM requests | ~49 (1 cast list + 48 batches) | 1,000 per model per day |
| LLM tokens | ~125,000 | 200,000 per model per day |
| Whisper requests | 4 chunks + repairs | 2,000 per day |
| Whisper audio | ~4,500 seconds | 28,800 seconds per day |

Tokens are what run out first. Adding helper models in Settings gives you a
separate 200 k bucket each — with the default three, roughly **4–5 hours of
video a day**.

A second API key helps only if it belongs to a *different* account: limits are
metered per organization, so a second key of your own shares the same buckets.

> **Batching, and why it is what it is.** Groq allows 8,000 tokens per minute,
> and `max_tokens` is reserved against that budget when the request is made — so
> an over-large `max_tokens` alone can trigger a 413. Translation runs in
> 30-line batches paced against the live `x-ratelimit-*` headers, and any batch
> still refused is split in half and retried.
>
> Those batches sit on **one grid across the whole transcript**, not one per
> streaming window. A window boundary almost never lands on a multiple of 30, and
> a leftover stub would become a whole extra request paying a full system prompt
> for two lines. With the grid, streaming costs exactly the same 48 requests as
> dubbing in one pass — the self-test asserts the two plans are identical.
>
> A daily limit is also told apart from a per-minute one. Per-minute is waited
> out; per-day cannot be, so that model is dropped from the pool and the others
> carry the work, rather than the app appearing to freeze.

---

## Cloned voices

This is voice *conversion*, not synthesis, and the distinction matters:

1. edge-tts speaks every line as usual, so the rate fitting, hard-anchored
   placement, pitch, pace and loudness settings all still apply;
2. the finished dub track is then passed through the model, which replaces the
   timbre and returns **audio of exactly the same length**.

Because step 2 cannot re-time anything, adding a cloned voice cannot introduce
drift. The self-test asserts this — it compares sample counts before and after
conversion and fails on any difference.

Conversion runs on the GPU under a separate virtualenv (torch + CUDA + Applio's
pinned dependencies) so none of that lands in this app's environment. Loading a
model costs ~14 s, far more than converting with it, so the worker process is
kept warm between jobs and shuts down after 10 minutes idle to release GPU
memory. A first preview takes ~15 s; later ones ~3 s.

`voice.json` fields:

| field | meaning |
|---|---|
| `name` | shown in the UI |
| `base_voice` | the Edge voice that speaks before conversion |
| `index_rate` | retrieval strength, 0–1. Higher locks timbre, smears consonants |
| `protect` | shields unvoiced consonants from conversion artifacts |

### Softness

A model trained on little audio converts with broadband roughness — measurable
as spectral flatness, which roughly doubles from the edge-tts input to the
converted output. The **Softness** slider (cloned voices only) drives the
spectral cleaner inside the conversion.

It is calibrated, not guessed. Measured on a model trained from ~2 minutes of
speech, against the reference speaker:

| softness | roughness | speaker match | |
|---|---|---|---|
| 0 | 0.030 | 0.985 | raw model output |
| 35 | 0.025 | 0.983 | default — as clean as plain edge-tts |
| 70 | 0.018 | 0.980 | smoother, drifting from the voice |
| 100 | 0.015 | 0.977 | smooth but no longer really them |

There is a genuine trade here, so the UI states which band you are in rather
than hiding it. The default of 35 brings roughness down to the level of ordinary
edge-tts speech while speaker match is still 0.983.

Softness acts *inside* the conversion, so changing it re-converts: about three
seconds on a preview with the worker warm, a few minutes for a whole episode.

> A hand-written spectral post-filter was tried first — smoothing across time and
> frequency, noise-floor subtraction, a high shelf — and discarded. Measured at
> every setting it *raised* flatness rather than lowering it, because modifying
> STFT magnitudes against unmodified phase breaks reconstruction consistency and
> the overlap-add artifacts land as exactly the roughness being chased.

Quality is bounded by the training audio, not the training run. More clean
speech is the single biggest improvement available.

---

## In the app

- **Library** — grid of every video, with live progress on anything processing
  and a **Watch now** button on anything already partly dubbed.
- **Watch** — player with English/original subtitle toggle, and a switch between
  the English dub and the original audio that keeps your playback position.
- **Transcript** — follows along as it plays; click a line to jump there.
- **Edit** — fix any English line, save, and optionally re-dub with the fix.
- **Voice & audio** — change the dub voice or the background-audio mix. Both reuse
  the existing translation, so neither makes an API call. Changing only the mix
  re-muxes from the cached dub track and finishes in seconds.
- **Downloads** — dubbed MP4, English `.srt`, original-language `.srt`.

## Where your API key is stored

Settings shows both paths inline, with a tick next to whichever currently holds
a key:

```
data\settings.json                 primary
data\cache\credentials.json        cached backup
```

Saving writes both. If `settings.json` is ever deleted or corrupted, the key is
restored from the cache automatically. Lookup order is `settings.json` →
`GROQ_API_KEY` environment variable → cached copy.

Both files are inside `data/`, which `.gitignore` excludes — along with
`settings.json` and `credentials.json` by name, wherever they appear — so a
`git add -A` cannot publish your key. **Test key** validates against Groq before
saving, so a mistyped key is caught immediately rather than halfway through your
first video.

## Settings worth knowing

- **Maximum speed-up** (default 1.7×) — the cap on compressing an over-long line.
  Lower it for calmer speech and slightly looser sync; raise it for tighter sync.
- **Keep original audio** (default **off**) — when enabled, the source is mixed
  quietly under the dub to preserve music and ambience. On a dialogue-heavy video
  that means hearing both languages at once. Changing it uses the **remix** path:
  the rendered dub track is reused, so it is one encode rather than a full re-dub.
- **Burn subtitles** — draws subtitles permanently into the picture. Requires a
  full video re-encode, so it is much slower.
- **Source language** (default auto) — detection also decides which pronoun notes
  the translator gets, so set it explicitly only if detection guesses wrong.

---

## Verifying it works

```bash
python selftest.py
```

Builds a synthetic clip, dubs scripted lines, and asserts that the audio lands
on the right timestamps, that over-long lines are compressed within the cap,
that a failed run resumes instead of restarting, that the streaming windows tile
the video without dropping a sample, and that streaming costs no more API
requests than dubbing in one pass. Needs internet for Edge TTS, but no API key.

## Layout

```
app/       config, media (ffmpeg), asr, translate, llm, tts, subtitles,
           hlsout (live streaming), voiceclone, library (sqlite),
           pipeline, main (FastAPI)
static/    index.html, app.js, style.css, vendor/hls.min.js
bin/       ffmpeg.exe, ffprobe.exe            (downloaded by setup.py)
data/      library/<id>/  source, dubbed.mp4, segments.json, srt/vtt, thumb
           voices/<id>/   optional cloned-voice models
           library.db, settings.json, cache/, tmp/
```

`bin/`, `data/` and `.venv/` are all created on first run and none are tracked
by git.

## Troubleshooting

**"ffmpeg is not installed"** — run `python setup.py`, or use `start.bat` which
does it for you.

**Processing fails right away** — no API key, or an expired model id. Open
Settings; if the model dropdown is empty your key is not being accepted.

**"out of tokens for the day"** — a per-day limit, not a per-minute one, so it
cannot be waited out. Add helper models in Settings to spread the load, or
retry after it resets. Retrying resumes rather than starting over.

**Upload size** — audio is sent as 16 kHz mono Opus at 32 kbps (~3.5 MB per
15-minute chunk, about a seventh of Groq's 25 MB cap). Whisper resamples to
16 kHz mono log-mel regardless, so lossless audio would cost ~10× the bytes for
no accuracy gain. Anything still oversized is split in half automatically.

**Dub sounds rushed** — lower Maximum speed-up in Settings and re-dub from the
watch page. The translator is already told to be concise, but some passages are
simply dense.

**A character is visibly speaking with no dub over them** — Whisper occasionally
collapses on a long span and returns vowel-stripped fragments, which then
translate to nothing. Those segments are detected and re-transcribed
automatically; on a real 1,400-line episode this recovered 12 collapsed segments
covering 138 seconds.

**Video has no speech** — music-only clips produce no segments and stop with a
clear message.
