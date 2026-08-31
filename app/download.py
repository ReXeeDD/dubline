"""Fetching a source video from a URL with yt-dlp.

Two jobs: list what qualities a link offers, and pull one of them down into a
library folder so the dubbing pipeline can pick it up later.

Nothing here starts a dub. A downloaded video sits in its own section until it
is explicitly sent for translation - grabbing a link and deciding to spend an
hour of Groq budget on it are separate decisions.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .config import BIN, ffmpeg
from .media import _NOWIN

EXE = ".exe" if os.name == "nt" else ""

# Video codecs, best first. avc1 is preferred well past the point where it is
# the smallest: it is the only one every player takes, and the dubbing pipeline
# stream-copies the picture rather than re-encoding it, so whatever is
# downloaded is exactly what comes out the other end.
_VCODEC_RANK = ("avc1", "av01", "vp09", "vp9")

# How good an audio track to pair with each picture size. The original audio is
# only ever heard by Whisper, which downmixes it to 16 kHz mono - so paying for
# the top track everywhere buys nothing. 129 kbps m4a is the sensible pairing
# for anything worth watching; the two smallest sizes get the light one because
# at 144p the audio would otherwise be three quarters of the download.
_AUDIO_FOR_HEIGHT = ((360, 128), (0, 48))


class NotInstalled(RuntimeError):
    pass


def ytdlp() -> str:
    """The yt-dlp binary: the bundled one first, then whatever is on PATH."""
    local = BIN / f"yt-dlp{EXE}"
    if local.exists():
        return str(local)
    found = shutil.which("yt-dlp")
    if found:
        return found
    raise NotInstalled(
        "yt-dlp is not installed. Run  python setup.py  to fetch it, or "
        "install it yourself from https://github.com/yt-dlp/yt-dlp/releases")


def available() -> bool:
    try:
        ytdlp()
        return True
    except NotInstalled:
        return False


# Browsers yt-dlp can read a cookie store from. Offered in Settings rather than
# guessed at, because the one to use is whichever is signed in AND closed.
BROWSERS = ["chrome", "brave", "edge", "firefox", "opera", "vivaldi", "safari",
            "chromium", "whale"]


def _cookie_args(cfg: dict | None = None) -> list[str]:
    """--cookies-from-browser, when Settings names one."""
    if cfg is None:
        from .config import load_settings
        cfg = load_settings()
    browser = (cfg.get("ytdlp_browser") or "").strip().lower()
    return ["--cookies-from-browser", browser] if browser in BROWSERS else []


# YouTube's bot wall, and the two ways it fails. The first is the wall itself;
# the second is what happens when a browser was named but is currently running,
# because its cookie database is locked while it is open.
_BOT_WALL = re.compile(r"confirm you.{0,3}re not a bot|sign in to confirm", re.I)
_LOCKED = re.compile(r"could not copy|cookie database|could not find .* cookies",
                     re.I)


def _explain(detail: str, cfg: dict | None = None) -> str:
    """Turn yt-dlp's message into one that says what to do about it."""
    if _LOCKED.search(detail):
        return (detail + "  -  that browser is open, and its cookies cannot be "
                "read while it is running. Close it, or pick a different "
                "browser under Settings > Downloads.")
    if _BOT_WALL.search(detail):
        if _cookie_args(cfg):
            return (detail + "  -  the cookies were read but YouTube still "
                    "refused. Sign in to YouTube in that browser, or choose "
                    "another one under Settings > Downloads.")
        return ("YouTube is asking this download to prove it is not a bot. "
                "Open Settings > Downloads and pick a browser you are signed "
                "in to YouTube with - it must be closed at the time, because "
                "its cookies cannot be read while it is running.")
    return detail


def _run(args: list[str], timeout: int = 180, cfg: dict | None = None) -> str:
    out = subprocess.run([ytdlp(), *_cookie_args(cfg), *args],
                         capture_output=True, timeout=timeout, **_NOWIN)
    if out.returncode != 0:
        msg = out.stderr.decode("utf-8", "ignore").strip().splitlines()
        detail = next((m for m in reversed(msg) if m.strip()), "")
        detail = re.sub(r"^ERROR:\s*", "", detail)
        raise RuntimeError(_explain(detail, cfg)[:400] or "yt-dlp failed")
    return out.stdout.decode("utf-8", "ignore")


def _best_audio(formats: list[dict], height: int) -> dict | None:
    """The audio track to pair with a picture this size.

    m4a is chosen over opus even when opus is better per bit, because an mp4
    picture and a webm sound track cannot be put in an mp4 - the merge would
    silently become an mkv.
    """
    want = next(kbps for at, kbps in _AUDIO_FOR_HEIGHT if height >= at)
    audio = [f for f in formats
             if f.get("vcodec") == "none" and f.get("acodec") not in (None, "none")
             and f.get("abr") and not f["format_id"].endswith("-drc")]
    if not audio:
        return None
    m4a = [f for f in audio if f.get("ext") == "m4a"] or audio
    # closest at or above the target, else simply the best there is
    at_least = [f for f in m4a if f["abr"] >= want - 1]
    return min(at_least, key=lambda f: f["abr"]) if at_least else max(
        m4a, key=lambda f: f["abr"])


def _rank(f: dict) -> tuple:
    codec = str(f.get("vcodec") or "")
    for i, name in enumerate(_VCODEC_RANK):
        if codec.startswith(name):
            return (i, -(f.get("tbr") or 0))
    return (len(_VCODEC_RANK), -(f.get("tbr") or 0))


def probe(url: str) -> dict:
    """What this link offers: title, length, and one entry per picture size."""
    raw = json.loads(_run(["-J", "--no-playlist", "--no-warnings", url]))
    formats = raw.get("formats") or []

    # One row per height. Several codecs are offered at each size and they are
    # not alternatives worth showing - a viewer wants "720p", not a choice
    # between four encodings of it.
    by_height: dict[int, dict] = {}
    for f in formats:
        h = f.get("height") or 0
        if not h or f.get("acodec") not in (None, "none"):
            continue          # audio-only, or a combined stream we do not need
        if f.get("ext") not in ("mp4", "webm") or f["format_id"].endswith("-drc"):
            continue
        if not (f.get("filesize") or f.get("filesize_approx")):
            continue          # no size means a stream yt-dlp cannot cost up front
        if h not in by_height or _rank(f) < _rank(by_height[h]):
            by_height[h] = f

    out = []
    for h, v in sorted(by_height.items(), reverse=True):
        a = _best_audio(formats, h)
        size = ((v.get("filesize") or v.get("filesize_approx") or 0)
                + ((a.get("filesize") or a.get("filesize_approx") or 0) if a else 0))
        out.append({
            "height": h,
            "label": f"{h}p" + (f"{int(v['fps'])}" if (v.get("fps") or 0) > 35 else ""),
            "format": f"{v['format_id']}+{a['format_id']}" if a else v["format_id"],
            "vcodec": str(v.get("vcodec") or "").split(".")[0],
            "abr": round(a["abr"]) if a else 0,
            "ext": "mp4" if v.get("ext") == "mp4" and (not a or a.get("ext") == "m4a")
                   else "mkv",
            "size": size,
        })
    return {
        "title": raw.get("title") or "Video",
        "duration": raw.get("duration") or 0,
        "uploader": raw.get("uploader") or "",
        "thumbnail": raw.get("thumbnail") or "",
        "webpage_url": raw.get("webpage_url") or url,
        "qualities": out,
    }


_PCT = re.compile(r"^DL\s+([\d.]+)%\s+(\S*)\s*(\S*)")


def fetch(url: str, fmt: str, dst_dir: Path, progress=None) -> Path:
    """Download one format into `dst_dir` as source.<ext>. Returns the file.

    yt-dlp is told where ffmpeg is rather than left to find it: the app ships
    its own copy in bin/, and on a machine with no system ffmpeg the merge of
    the separate picture and sound tracks would otherwise fail at the last step
    of a long download.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for old in dst_dir.glob("source.*"):
        old.unlink(missing_ok=True)

    args = [
        *_cookie_args(),
        "-f", fmt, "--no-playlist", "--no-warnings", "--newline",
        "--no-part", "--retries", "10", "--fragment-retries", "10",
        "--merge-output-format", "mp4/mkv",
        "--ffmpeg-location", str(Path(ffmpeg()).parent),
        "--progress-template", "DL %(progress._percent_str)s "
                               "%(progress._speed_str)s %(progress._eta_str)s",
        "-o", str(dst_dir / "source.%(ext)s"), url,
    ]
    proc = subprocess.Popen([ytdlp(), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            errors="ignore", **_NOWIN)
    last = ""
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        last = line
        m = _PCT.match(line.replace("%", "% ", 1))
        if m and progress:
            pct, speed, eta = m.group(1), m.group(2), m.group(3)
            progress(f"Downloading {pct}% at {speed or '-'}"
                     + (f", {eta} left" if eta and eta != "Unknown" else ""),
                     min(97, int(float(pct))))
        elif progress and line.startswith("[Merger]"):
            progress("Joining picture and sound", 98)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            _explain(re.sub(r"^ERROR:\s*", "", last))[:400] or "download failed")

    got = sorted(dst_dir.glob("source.*"), key=lambda p: p.stat().st_size)
    if not got:
        raise RuntimeError("yt-dlp finished but produced no file")
    return got[-1]
