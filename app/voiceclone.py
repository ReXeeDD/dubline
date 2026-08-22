"""Cloned voices: re-timbre the finished dub track with an RVC model.

This is voice *conversion*, not synthesis. edge-tts still speaks every line, so
all the timing work upstream is untouched - the rate fitting, the hard-anchored
placement, the pitch and loudness settings. The dub track is then passed through
an RVC model that replaces the timbre and returns audio of the same length.

Conversion runs under a separate virtualenv (torch + CUDA + Applio's pinned
dependency set) so none of that lands in the app's own environment.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import DATA, ROOT

VOICES_DIR = DATA / "voices"

# Where the training toolchain was installed. Overridable in settings.json for
# a machine that keeps it somewhere else.
DEFAULT_APPLIO = Path("D:/zt/voice-clone/applio")
DEFAULT_PYTHON = Path("D:/zt/voice-clone/venv/Scripts/python.exe")

# Long files are converted in pieces: it bounds peak memory and lets progress
# be reported. Cuts land inside silence so no word is ever split.
CHUNK_SECONDS = 300.0
MIN_GAP_FOR_CUT = 0.25

# Loading the model costs far more than converting with it, so the worker is
# kept warm between jobs and shut down only after a spell of inactivity - it
# holds GPU memory the rest of the machine may want.
IDLE_SHUTDOWN = 600.0
TAG = "##RVC##"          # must match app/rvc_infer.py
_LOCK = threading.RLock()   # serialises the single GPU worker


class _Worker:
    """A long-lived conversion process with the model resident."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.last_used = 0.0
        self._timer: threading.Timer | None = None

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _read_reply(self) -> dict:
        """Next tagged line. Applio's own chatter on stdout is discarded."""
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    "the voice-conversion worker stopped unexpectedly")
            if line.startswith(TAG):
                return json.loads(line[len(TAG):])

    def start(self, applio: Path, python: Path) -> None:
        if self._alive():
            return
        self.proc = subprocess.Popen(
            [str(python), str(ROOT / "app" / "rvc_infer.py"), str(applio)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, cwd=str(applio))
        if not self._read_reply().get("ready"):
            raise RuntimeError("the voice-conversion worker did not start")

    def run(self, job: dict) -> None:
        if not self._alive():
            raise RuntimeError("the voice-conversion worker stopped unexpectedly")
        self.proc.stdin.write(json.dumps(job) + "\n")
        self.proc.stdin.flush()
        res = self._read_reply()
        if not res.get("ok"):
            raise RuntimeError(res.get("error", "voice conversion failed"))
        self.last_used = time.time()

    def touch_idle_timer(self) -> None:
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(IDLE_SHUTDOWN, self.stop)
        self._timer.daemon = True
        self._timer.start()

    def stop(self) -> None:
        with _LOCK:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            if self._alive():
                try:
                    self.proc.stdin.write("\n")
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=10)
                except Exception:
                    self.proc.kill()
            self.proc = None


_WORKER = _Worker()


def shutdown() -> None:
    """Release the GPU. Safe to call when nothing is running."""
    _WORKER.stop()


# ------------------------------------------------------------------ voices ---
def list_clones() -> list[dict]:
    """Every cloned voice installed under data/voices/."""
    out = []
    if not VOICES_DIR.exists():
        return out
    for d in sorted(VOICES_DIR.iterdir()):
        meta = d / "voice.json"
        pth = d / "model.pth"
        if not (meta.exists() and pth.exists()):
            continue
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        info.setdefault("id", d.name)
        info.setdefault("name", d.name.title())
        info.setdefault("gender", "Unknown")
        info.setdefault("base_voice", "en-US-AndrewMultilingualNeural")
        info["clone"] = True
        info["dir"] = str(d)
        out.append(info)
    return out


def get_clone(voice_id: str) -> dict | None:
    if not voice_id:
        return None
    for c in list_clones():
        if c["id"] == voice_id:
            return c
    return None


def is_clone(voice_id: str) -> bool:
    return get_clone(voice_id) is not None


def base_voice_for(voice_id: str) -> str:
    """The edge-tts voice that actually speaks, before conversion."""
    c = get_clone(voice_id)
    return c["base_voice"] if c else voice_id


def toolchain(cfg: dict | None = None) -> tuple[Path, Path]:
    cfg = cfg or {}
    applio = Path(cfg.get("clone_applio_dir") or DEFAULT_APPLIO)
    python = Path(cfg.get("clone_python") or DEFAULT_PYTHON)
    return applio, python


def available(cfg: dict | None = None) -> bool:
    applio, python = toolchain(cfg)
    return python.exists() and (applio / "core.py").exists()


# ------------------------------------------------------------------- audio ---
def _cut_points(x: np.ndarray, sr: int) -> list[int]:
    """Split offsets inside silent gaps, roughly CHUNK_SECONDS apart."""
    step = int(CHUNK_SECONDS * sr)
    if len(x) <= step:
        return [0, len(x)]

    win = int(0.020 * sr)
    env = np.convolve(np.abs(x), np.ones(win) / win, "same")
    quiet = env < 1e-5

    points = [0]
    target = step
    while target < len(x) - step // 4:
        lo = max(points[-1] + step // 4, target - step // 3)
        hi = min(len(x) - 1, target + step // 3)
        seg = quiet[lo:hi]
        best = None
        if seg.any():
            # longest silent run in the window, cut through its middle
            run = start = 0
            for i, q in enumerate(seg):
                if q:
                    start = i if run == 0 else start
                    run += 1
                    if run * (1 / sr) >= MIN_GAP_FOR_CUT and (
                            best is None or run > best[1]):
                        best = (lo + start + run // 2, run)
                else:
                    run = 0
        points.append(best[0] if best else min(target, len(x) - 1))
        target = points[-1] + step
    points.append(len(x))
    return points


def _gate(converted: np.ndarray, source: np.ndarray, sr: int) -> np.ndarray:
    """Lock length to the source and restore its silence.

    The dub track is speech placed into digital silence. Conversion fills those
    gaps with a low-level bed and can drift a frame or two in length; both are
    corrected here rather than left in the output.
    """
    if len(converted) < len(source):
        converted = np.pad(converted, (0, len(source) - len(converted)))
    converted = converted[:len(source)]

    win = int(0.020 * sr)
    env = np.convolve(np.abs(source), np.ones(win) / win, "same")
    keep = env > 1e-5
    # keep a little either side so onsets and tails are never clipped
    pad = int(0.020 * sr)
    keep = np.convolve(keep.astype(np.float32), np.ones(2 * pad + 1), "same") > 0
    ramp = max(1, int(0.006 * sr))
    mask = np.convolve(keep.astype(np.float32), np.ones(ramp) / ramp, "same")
    return converted * np.clip(mask, 0.0, 1.0)


# -------------------------------------------------------------- conversion ---
def convert(src: Path, dst: Path, voice_id: str, cfg: dict | None = None,
            progress=None, wait: float = 900.0) -> Path:
    """Re-timbre `src` with the cloned voice and write `dst`.

    Output length always equals input length, so subtitle alignment is
    unaffected. Raises RuntimeError if the toolchain or model is missing.

    `wait` is how long to queue behind another conversion before giving up. A
    dub can afford to wait; an interactive preview should fail with a message
    rather than hang, so it passes something short.
    """
    cfg = cfg or {}
    clone = get_clone(voice_id)
    if clone is None:
        raise RuntimeError(f"No cloned voice named '{voice_id}' is installed.")
    applio, python = toolchain(cfg)
    if not available(cfg):
        raise RuntimeError(
            f"The voice-conversion toolchain was not found at {applio}. "
            "Reinstall it, or pick a standard voice in Settings.")

    vdir = Path(clone["dir"])
    pth = vdir / "model.pth"
    index = vdir / "model.index"
    index_rate = float(cfg.get("clone_index_rate", clone.get("index_rate", 0.5)))
    protect = float(cfg.get("clone_protect", clone.get("protect", 0.33)))
    soften = float(cfg.get("soften", clone.get("soften", 0)))

    x, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)

    # One GPU and one worker process, so one job at a time. Without this two
    # requests - a slider drag fires several - write to the same stdin pipe and
    # read each other's replies, and the line protocol comes apart.
    if not _LOCK.acquire(timeout=float(wait)):
        raise RuntimeError(
            "The voice converter is busy with another job. Try again in a moment.")
    try:
        # Scratch names must be unique per job as well: a shared folder let one
        # request delete the chunk another was still reading.
        work = src.parent / f"clone_{uuid.uuid4().hex[:10]}"
        work.mkdir(parents=True, exist_ok=True)
        points = _cut_points(x, sr)
        total = len(points) - 1
        out = np.zeros(len(x), dtype=np.float32)
        started = time.time()

        # One worker for every chunk: the model loads once instead of per chunk.
        _WORKER.start(applio, python)

        for i in range(total):
            a, b = points[i], points[i + 1]
            piece = x[a:b]
            if not np.any(np.abs(piece) > 1e-5):
                continue                               # silent stretch

            pin = work / f"in_{i:03d}.wav"
            pout = work / f"out_{i:03d}.wav"
            sf.write(str(pin), piece, sr)

            _WORKER.run({
                "in": str(pin), "out": str(pout), "pth": str(pth),
                "index": str(index) if index.exists() else "",
                "index_rate": index_rate, "protect": protect, "soften": soften,
            })
            if not pout.exists():
                raise RuntimeError(
                    "Voice conversion produced no audio for part "
                    f"{i + 1} of {total}.")

            y, ysr = sf.read(str(pout), dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if ysr != sr:                              # model rate differs
                n = int(round(len(y) * sr / ysr))
                y = np.interp(np.linspace(0, len(y) - 1, n),
                              np.arange(len(y)), y).astype(np.float32)
            out[a:b] = _gate(y, piece, sr)

            pin.unlink(missing_ok=True)
            pout.unlink(missing_ok=True)
            if progress:
                done = i + 1
                progress(f"Applying the {clone['name']} voice {done}/{total}",
                         96 + int(2 * done / max(1, total)))
    finally:
        _LOCK.release()

    sf.write(str(dst), out, sr)
    shutil.rmtree(work, ignore_errors=True)
    _WORKER.touch_idle_timer()      # release the GPU if nothing follows
    if progress:
        progress(f"Voice conversion finished in {time.time() - started:.0f}s", 98)
    return dst
