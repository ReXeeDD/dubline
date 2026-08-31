"""One-time setup: fetch ffmpeg into ./bin and verify the install.

Everything lands inside this project folder on D: - nothing touches C:.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "bin"
TMP = ROOT / "data" / "tmp"
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


def have(name: str) -> str | None:
    p = BIN / f"{name}{EXE}"
    if p.exists():
        return str(p)
    return shutil.which(name)


def download_ffmpeg() -> None:
    BIN.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ffmpeg -> {BIN}")
    print(f"  source: {URL}")

    zip_path = TMP / "ffmpeg.zip"
    last = [0]

    def hook(blocks: int, bsize: int, total: int) -> None:
        got = blocks * bsize
        pct = int(got * 100 / total) if total > 0 else 0
        if pct >= last[0] + 5 or got >= total:
            last[0] = pct
            bar = "#" * (pct // 3)
            sys.stdout.write(f"\r  [{bar:<33}] {pct:3d}%  {_mb(got)}")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(URL, zip_path, reporthook=hook)
    except Exception as e:
        print(f"\n  Download failed: {e}")
        print("\n  Alternatives:")
        print("   1. winget install Gyan.FFmpeg   (installs system-wide)")
        print(f"   2. Download {URL}")
        print(f"      and copy ffmpeg{EXE} + ffprobe{EXE} into {BIN}")
        sys.exit(1)

    print("\n  Extracting...")
    extracted = 0
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            name = Path(member.filename).name
            if name in WANTED:
                with z.open(member) as srcf, (BIN / name).open("wb") as dstf:
                    shutil.copyfileobj(srcf, dstf)
                (BIN / name).chmod(0o755)
                extracted += 1
                print(f"    {name}  ({_mb(member.file_size)})")

    zip_path.unlink(missing_ok=True)

    if extracted < 2:
        print(f"  Expected 2 binaries, extracted {extracted}.")
        sys.exit(1)


def download_ytdlp() -> bool:
    """Fetch yt-dlp into ./bin. Optional: only the URL downloader needs it."""
    dst = BIN / f"yt-dlp{EXE}"
    if dst.exists() or shutil.which("yt-dlp"):
        return True
    BIN.mkdir(parents=True, exist_ok=True)
    print(f"Downloading yt-dlp -> {BIN}")
    try:
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(YTDLP_URL, tmp)
        tmp.replace(dst)
        dst.chmod(0o755)
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


def main() -> None:
    print("=" * 58)
    print("  Dubline setup")
    print("=" * 58)
    print(f"  Project : {ROOT}")
    print(f"  Binaries: {BIN}")
    print()

    missing = check_python_deps()
    if missing:
        print("  Missing Python packages: " + ", ".join(missing))
        print("  Install them with:")
        print("    pip install " + " ".join(missing))
        print()

    ff, fp = have("ffmpeg"), have("ffprobe")
    if ff and fp:
        print(f"  ffmpeg  found: {ff}")
        print(f"  ffprobe found: {fp}")
    else:
        download_ffmpeg()
        ff, fp = have("ffmpeg"), have("ffprobe")

    print()
    for name, path in (("ffmpeg", ff), ("ffprobe", fp)):
        try:
            out = subprocess.run([path, "-version"], capture_output=True, timeout=25)
            line = out.stdout.decode("utf-8", "ignore").splitlines()[0]
            print(f"  OK  {line[:70]}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            sys.exit(1)

    print()
    download_ytdlp()

    for d in ["data/library", "data/tmp"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    print()
    if missing:
        print("  Install the packages listed above, then start the app:")
    else:
        print("  Setup complete. Start the app with:")
    print("    python run.py")
    print()


if __name__ == "__main__":
    main()
