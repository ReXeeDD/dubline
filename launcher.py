"""Bootstrap and update Dubline for both Windows launchers."""
from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PureWindowsPath
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
TMP = ROOT / "data" / "tmp"
UPDATES = ROOT / "data" / "updates"
VENV = ROOT / ".venv"
PYTHON = VENV / "Scripts" / "python.exe"
REQUIREMENTS_STAMP = VENV / "dubline-requirements.sha256"
REPO = "ReXeeDD/dubline"
SHA = re.compile(r"[0-9a-f]{40}")
RUNTIME = {"run.py", "setup.py", "requirements.txt", "app", "static"}


def git_head() -> tuple[bool, str | None]:
    """Use the working tree on developer branches or when local work is present."""
    if not (ROOT / ".git").exists():
        return True, None

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "--no-optional-locks", *args], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL, timeout=5).strip()

    try:
        origin = git("remote", "get-url", "origin").lower().removesuffix(".git")
        if origin not in {f"https://github.com/{REPO.lower()}", f"git@github.com:{REPO.lower()}"}:
            return False, None
        if git("status", "--porcelain", "--branch").splitlines() != ["## main...origin/main"]:
            return False, None
        return True, git("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return False, None


def fetch(url: str, limit: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Dubline Windows launcher"})
    with urllib.request.urlopen(request, timeout=6) as response:
        if urlparse(response.geturl()).scheme != "https":
            raise ValueError("GitHub redirected the download away from HTTPS")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("GitHub response exceeded the download size limit")
    return data


def read_stamp(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return ""


def write_stamp(path: Path, value: str) -> None:
    pending = path.with_suffix(".new")
    pending.write_text(value + "\n", encoding="ascii")
    pending.replace(path)


def ready_snapshot(path: Path) -> bool:
    return all((path / name).is_file() for name in
               ("run.py", "setup.py", "requirements.txt", "app/config.py", "static/index.html"))


def stage_update(commit: str) -> Path:
    target = UPDATES / commit
    if ready_snapshot(target):
        return target

    print(f"  Downloading Dubline update {commit[:8]}...", flush=True)
    url = f"https://codeload.github.com/{REPO}/zip/{commit}"
    data = fetch(url, 16 * 1024 * 1024)
    with tempfile.TemporaryDirectory(prefix="staging-", dir=UPDATES) as temporary:
        stage = Path(temporary) / "app"
        stage.mkdir()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            prefix = f"dubline-{commit}/"
            if prefix + "launcher.py" not in archive.namelist():
                raise ValueError("This GitHub revision does not support the Windows updater")
            total_size = 0
            for entry in archive.infolist():
                if entry.is_dir() or not entry.filename.startswith(prefix):
                    continue
                relative = entry.filename[len(prefix):]
                parts = relative.split("/")
                if parts[0] not in RUNTIME:
                    continue
                if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part.endswith(".")
                       or PureWindowsPath(part).is_reserved() for part in parts):
                    raise ValueError(f"Invalid update archive path: {relative}")
                total_size += entry.file_size
                if total_size > 64 * 1024 * 1024:
                    raise ValueError("Unpacked update exceeded the size limit")
                output = stage.joinpath(*parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, output.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
        if not ready_snapshot(stage):
            raise ValueError("Update archive is missing required Dubline files")
        stage.replace(target)
    return target


def prepare(source: Path, env: dict[str, str]) -> None:
    requirements = (source / "requirements.txt").resolve()
    fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
    fresh = not (VENV / "dubline-setup-complete").exists() and not REQUIREMENTS_STAMP.exists()
    if read_stamp(REQUIREMENTS_STAMP) != fingerprint:
        REQUIREMENTS_STAMP.unlink(missing_ok=True)
        print("  Checking Python packages...", flush=True)
        if subprocess.call([PYTHON, "-m", "pip", "--version"], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
            subprocess.run([PYTHON, "-m", "ensurepip", "--default-pip"], env=env, check=True)
        subprocess.run([PYTHON, "-m", "pip", "install", "--disable-pip-version-check",
                        "--quiet", "-r", requirements], cwd=source, env=env, check=True)
    setup = [PYTHON, source / "setup.py"]
    if not fresh:
        setup.append("--ffmpeg-only")
    subprocess.run(setup, cwd=source, env=env, check=True)
    if read_stamp(REQUIREMENTS_STAMP) != fingerprint:
        write_stamp(REQUIREMENTS_STAMP, fingerprint)


def prepare_installation(no_update: bool, env: dict[str, str]) -> Path:
    active_file = UPDATES / "active"
    allowed, head = (False, None) if no_update else git_head()
    current = ROOT
    candidate = None
    if allowed:
        active = read_stamp(active_file)
        if SHA.fullmatch(active) and ready_snapshot(UPDATES / active):
            current = UPDATES / active
        try:
            url = f"https://api.github.com/repos/{REPO}/commits/main"
            commit = json.loads(fetch(url, 2 * 1024 * 1024))["sha"]
            if not isinstance(commit, str) or not SHA.fullmatch(commit):
                raise ValueError("Invalid commit from GitHub")
            if commit == head:
                current = ROOT
                active_file.unlink(missing_ok=True)
            elif commit != active or current == ROOT:
                candidate = stage_update(commit)
        except (OSError, KeyError, TypeError, ValueError, RuntimeError,
                http.client.HTTPException, zipfile.BadZipFile) as exc:
            print(f"  Update unavailable. Starting the installed version. ({exc})", flush=True)

    if candidate is not None:
        try:
            prepare(candidate, env)
            write_stamp(active_file, candidate.name)
            current = candidate
            print(f"  Dubline updated to {candidate.name[:8]}.", flush=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"  Update setup failed. Starting the installed version. ({exc})", flush=True)
    if current != candidate:
        prepare(current, env)
    return current


def main() -> int:
    import msvcrt

    if sys.version_info < (3, 10):
        print("  Dubline requires Python 3.10 or newer.", file=sys.stderr)
        return 1
    args = sys.argv[1:]
    # The executable passes arguments without interpreting them in cmd.exe.
    encoded = os.environ.pop("DUBLINE_EXE_ARGS", "")
    if encoded:
        args = [base64.b64decode(item).decode("utf-8") for item in encoded[1:].split(",")]

    TMP.mkdir(parents=True, exist_ok=True)
    UPDATES.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DUBLINE_INSTALL_DIR=str(ROOT), TMP=str(TMP),
               TEMP=str(TMP), TMPDIR=str(TMP), PIP_CACHE_DIR=str(ROOT / "data" / "cache" / "pip"))
    try:
        with (UPDATES / "launcher.lock").open("a+b") as lock:
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                print("  Another Dubline launcher is setting up. Try again in a moment.")
                return 1
            if not PYTHON.is_file():
                print("  Creating the Python environment...", flush=True)
                subprocess.run([sys.executable, "-m", "venv", VENV], env=env, check=True)
            source = prepare_installation("--no-update" in args, env)
        server_args = [arg for arg in args if arg != "--no-update"]
        return subprocess.call([PYTHON, source / "run.py", "--open", *server_args],
                               cwd=source, env=env)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"  Dubline could not start. Run it again to retry. ({exc})", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
