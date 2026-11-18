"""Verify the install and fetch missing executables into ./bin on Windows."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from app.config import BIN, INSTALL_DIR, TMP, ffmpeg, ffmpeg_executable_works, ffmpeg_ready, ffprobe

EXE = ".exe" if os.name == "nt" else ""

URL = ("https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
       "ffmpeg-master-latest-win64-gpl.zip")

WANTED = {f"ffmpeg{EXE}", f"ffprobe{EXE}"}

# yt-dlp ships as a single self-contained executable, so fetching it is one
# request and no unpacking. It is optional - the app only needs it to download
# a video from a link - so a failure here is a warning, not a dead install.
YTDLP_URL = {
    "nt": "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe",
}.get(os.name, "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp")


def _mb(n: int) -> str:
    return f"{n / 1048576:.1f} MB"


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=30) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def download_ffmpeg() -> None:
    print(f"Downloading ffmpeg and ffprobe -> {BIN}", flush=True)
    with tempfile.TemporaryDirectory(prefix="ffmpeg-", dir=TMP) as work:
        stage = Path(work)
        archive = stage / "ffmpeg.zip"
        _download(URL, archive)

        print("  Extracting and verifying ffmpeg and ffprobe...", flush=True)
        with zipfile.ZipFile(archive) as z:
            members = {Path(member.filename).name: member for member in z.infolist()
                       if not member.is_dir()}
            for name in sorted(WANTED):
                if name not in members:
                    raise RuntimeError(f"The archive is missing {name}.")
                candidate = stage / name
                with z.open(members[name]) as src, candidate.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if not ffmpeg_executable_works(candidate):
                    raise RuntimeError(f"The downloaded {name} failed its version check.")

        for name in sorted(WANTED):
            (stage / name).replace(BIN / name)


def ensure_ffmpeg() -> bool:
    if ffmpeg_ready():
        return True
    if os.name != "nt":
        print("Install ffmpeg and ffprobe with your system package manager and put "
              "them on PATH.", file=sys.stderr)
        return False

    try:
        download_ffmpeg()
        if not ffmpeg_ready():
            raise RuntimeError("The installed ffmpeg and ffprobe could not be verified.")
    except (OSError, RuntimeError, zipfile.BadZipFile) as e:
        print(f"  Could not install ffmpeg and ffprobe: {e}", file=sys.stderr)
        print(f"  Download {URL}\n"
              f"  and copy ffmpeg.exe and ffprobe.exe from its bin folder into {BIN}.",
              file=sys.stderr)
        return False
    return True


def download_ytdlp() -> bool:
    """Fetch yt-dlp into ./bin. Optional: only the URL downloader needs it."""
    dst = BIN / f"yt-dlp{EXE}"
    if dst.exists() or shutil.which("yt-dlp"):
        return True
    print(f"Downloading yt-dlp -> {BIN}", flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="yt-dlp-", dir=TMP) as work:
            tmp = Path(work) / dst.name
            _download(YTDLP_URL, tmp)
            tmp.chmod(0o755)
            tmp.replace(dst)
        print(f"  yt-dlp  ({_mb(dst.stat().st_size)})")
        return True
    except Exception as e:
        print(f"  Could not fetch yt-dlp: {e}")
        print("  Downloading videos from a link will be unavailable until you")
        print(f"  put yt-dlp{EXE} in {BIN} yourself.")
        return False


def check_python_deps() -> list[str]:
    missing = []
    for mod, pkg in [("fastapi", "fastapi"), ("uvicorn", "uvicorn[standard]"),
                     ("groq", "groq"), ("edge_tts", "edge-tts"),
                     ("numpy", "numpy"), ("multipart", "python-multipart"),
                     ("soundfile", "soundfile")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up Dubline")
    parser.add_argument("--ffmpeg-only", action="store_true",
                        help="verify or install ffmpeg and ffprobe for startup")
    if parser.parse_args().ffmpeg_only:
        return 0 if ensure_ffmpeg() else 1

    print("=" * 58)
    print("  Dubline setup")
    print("=" * 58)
    print(f"  Project : {INSTALL_DIR}")
    print(f"  Binaries: {BIN}")
    print()

    missing = check_python_deps()
    if missing:
        print("  Missing Python packages: " + ", ".join(missing))
        print("  Install them with:")
        print("    pip install " + " ".join(missing))
        print()

    if not ensure_ffmpeg():
        return 1
    print(f"  ffmpeg  verified: {ffmpeg()}")
    print(f"  ffprobe verified: {ffprobe()}")

    print()
    download_ytdlp()

    print()
    if missing:
        print("  Install the packages listed above, then start the app:")
    else:
        print("  Setup complete. Start the app with:")
    print("    python run.py")
    print()
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
