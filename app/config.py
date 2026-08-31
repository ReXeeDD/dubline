"""Configuration, on-disk layout and binary resolution.

Everything this app writes - temp files, ffmpeg binaries, uploads, outputs -
stays inside the project folder on D:. Nothing is written to C:.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LIBRARY = DATA / "library"
TMP = DATA / "tmp"
CACHE = DATA / "cache"
BIN = ROOT / "bin"
SETTINGS_FILE = DATA / "settings.json"
# Second copy of the API key. If settings.json is ever deleted or corrupted the
# key is recovered from here instead of having to be pasted again.
CRED_CACHE = CACHE / "credentials.json"

for _d in (DATA, LIBRARY, TMP, CACHE, BIN):
    _d.mkdir(parents=True, exist_ok=True)

# Keep every temporary file on D:. Starlette spools large uploads through
# tempfile, so without this a multi-GB upload would land on a nearly full C:.
tempfile.tempdir = str(TMP)
os.environ["TMP"] = os.environ["TEMP"] = os.environ["TMPDIR"] = str(TMP)
# Any library that caches models does so here too, never in the user profile.
os.environ.setdefault("HF_HOME", str(DATA / "hf"))
os.environ.setdefault("XDG_CACHE_HOME", str(DATA / "cache"))


# ---------------------------------------------------------------- defaults ---
DEFAULTS = {
    "groq_api_key": "",
    # Extra Groq keys used alongside the one above. Worth adding only if a key
    # belongs to a DIFFERENT account: limits are metered per organization, so a
    # second key of your own shares the same buckets and changes nothing.
    "groq_api_keys": [],
    # Groq hosted Whisper. turbo is ~2x faster, large-v3 is the most accurate
    # for Chinese. Overridable from the Settings panel.
    "asr_model": "whisper-large-v3",
    # Translation LLM. Settings loads the live /models list from the account,
    # so this is only the starting point.
    "llm_model": "qwen/qwen3.6-27b",
    # "groq" (default, far better quality) or "local" for an OpenAI-compatible
    # server such as LM Studio. Transcription always goes through Groq.
    # Groq meters tokens per minute PER MODEL, so running several at once
    # multiplies throughput on one API key. Empty means use llm_model alone.
    "llm_helpers": [],
    "llm_provider": "groq",
    "local_base_url": "http://localhost:1234/v1",
    "local_model": "",
    # "auto" lets Whisper name the language itself, which is also what tells
    # the translator which pronoun system it is reading. Set it explicitly
    # only if detection keeps guessing wrong on quiet or accented audio.
    "source_language": "auto",
    # A stock edge-tts voice, so a fresh install dubs with no extra downloads.
    # Cloned voices installed under data/voices/ appear in the same list and
    # are spoken by their base edge-tts voice first, then re-timbred, so every
    # timing and prosody setting below still applies to them too.
    "voice": "en-US-BrianNeural",
    # Retrieval strength against the training set. Higher locks timbre harder
    # but can smear consonants; 0.5 is the usual balance.
    "clone_index_rate": 0.5,
    # Protects unvoiced consonants from conversion artifacts.
    "clone_protect": 0.33,
    # Softness, 0-100, driving the spectral cleaner on a cloned voice. A model
    # trained on little audio converts with broadband roughness; cleaning it up
    # trades against how much the result still sounds like the target speaker.
    # Measured on the Rei model: 30 -> speaker match 0.983, 50 -> 0.980,
    # 90 -> 0.967. The UI marks where that trade starts to bite.
    "soften": 35,
    # Where the RVC toolchain lives. Both stay on D:.
    "clone_applio_dir": "D:/zt/voice-clone/applio",
    "clone_python": "D:/zt/voice-clone/venv/Scripts/python.exe",
    # Prosody pitch shift in Hz applied to every line. Negative is deeper:
    # -40 Hz takes a typical male voice from about 140 Hz down to about 103 Hz.
    "pitch": 0,
    # Baseline speaking pace as a percentage, -40 to +50. Lines that overrun
    # their slot are still compressed further on top of this.
    "speed": 0,
    # Output loudness in percent, -50 to +50. Useful when the dub sits too
    # quietly against the background bed.
    "volume": 0,
    # Off by default: mixing the source under the dub means two voices speaking
    # at once. Only worth enabling for music-led video with little dialogue.
    "keep_original_audio": False,
    "original_audio_gain": 0.06,   # background bed level when it is enabled
    "max_speedup": 1.7,            # hard cap on time-compressing a dub line
    "burn_subtitles": False,
    # Which browser yt-dlp should borrow cookies from, or "" for none.
    #
    # YouTube now answers an unauthenticated request for many videos with
    # "Sign in to confirm you're not a bot", and no yt-dlp version gets past
    # that on its own - the copy here is already the current release. Reading
    # the cookies of a browser you are signed in with is yt-dlp's own answer to
    # it, and it is what turns the bot check back into a normal download:
    # measured on a blocked link, 0 usable formats without it and 47 with.
    #
    # The cookie database is locked while that browser is running, so pick one
    # that is closed - Chrome open in the foreground cannot be read.
    "ytdlp_browser": "",
}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_settings() -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(_read_json(SETTINGS_FILE))

    # Key lookup order: settings.json -> GROQ_API_KEY env var -> cached copy.
    if not cfg.get("groq_api_key"):
        cfg["groq_api_key"] = (os.environ.get("GROQ_API_KEY", "")
                               or _read_json(CRED_CACHE).get("groq_api_key", ""))
    return cfg


def save_settings(patch: dict) -> dict:
    cfg = load_settings()
    for k, v in patch.items():
        if k in DEFAULTS:
            cfg[k] = v
    SETTINGS_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    cache_credentials(cfg.get("groq_api_key", ""))
    return cfg


def cache_credentials(api_key: str) -> None:
    """Keep a recoverable copy of the key outside settings.json."""
    if not api_key:
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    CRED_CACHE.write_text(
        json.dumps({"groq_api_key": api_key, "saved_at": time.time()}, indent=2),
        encoding="utf-8")
    try:  # best effort on Windows; owner-only where the OS supports it
        CRED_CACHE.chmod(0o600)
    except Exception:
        pass


def key_locations() -> dict:
    """Where the key is stored, for display in the Settings panel."""
    return {
        "settings_file": str(SETTINGS_FILE),
        "cache_file": str(CRED_CACHE),
        "settings_saved": bool(_read_json(SETTINGS_FILE).get("groq_api_key")),
        "cache_saved": bool(_read_json(CRED_CACHE).get("groq_api_key")),
        "from_env": bool(os.environ.get("GROQ_API_KEY")),
    }


# ------------------------------------------------------------------ ffmpeg ---
_BIN_CACHE: dict[str, str] = {}
_EXE = ".exe" if os.name == "nt" else ""

# BtbN's static Windows builds: one zip with both binaries, no installer.
FFMPEG_ZIP = ("https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
              "ffmpeg-master-latest-win64-gpl.zip")


def _probe(path: str) -> bool:
    try:
        subprocess.run([path, "-version"], capture_output=True, timeout=20)
        return True
    except Exception:
        return False


def _resolve(name: str) -> str:
    """Prefer our own bin/ on D:, then PATH. Never installs into C:."""
    if name in _BIN_CACHE:
        return _BIN_CACHE[name]

    local = BIN / f"{name}{_EXE}"
    if local.exists() and _probe(str(local)):
        _BIN_CACHE[name] = str(local)
        return str(local)

    found = shutil.which(name)
    if found and _probe(found):
        _BIN_CACHE[name] = found
        return found

    raise RuntimeError(
        f"{name} is not installed. Run:  python setup.py\n"
        f"That downloads ffmpeg into {BIN} (about 80 MB, stays on D:)."
    )


def ffmpeg() -> str:
    return _resolve("ffmpeg")


def ffprobe() -> str:
    return _resolve("ffprobe")


def ffmpeg_ready() -> bool:
    try:
        _resolve("ffmpeg")
        _resolve("ffprobe")
        return True
    except Exception:
        return False
