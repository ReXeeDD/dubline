"""ffmpeg helpers: probing, audio extraction, PCM decode, precise assembly, mux."""
from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import numpy as np

from .config import ffmpeg, ffprobe

SR = 24000  # dub track sample rate (edge-tts native is 24kHz mono)

_NOWIN = {}
try:  # hide console popups on Windows
    _NOWIN["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
except AttributeError:
    pass


def run(cmd: list[str], desc: str = "") -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, **_NOWIN)
    if p.returncode != 0:
        tail = p.stderr.decode("utf-8", "ignore")[-1500:]
        raise RuntimeError(f"ffmpeg failed{' during ' + desc if desc else ''}:\n{tail}")
    return p


def probe(path: Path) -> dict:
    p = subprocess.run(
        [ffprobe(), "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, **_NOWIN,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe could not read {path.name}")
    return json.loads(p.stdout.decode("utf-8", "ignore"))


def media_info(path: Path) -> dict:
    info = probe(path)
    fmt = info.get("format", {})
    v = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    a = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    return {
        "duration": float(fmt.get("duration") or 0.0),
        "size": int(fmt.get("size") or 0),
        "has_video": v is not None,
        "has_audio": a is not None,
        "width": int(v.get("width", 0)) if v else 0,
        "height": int(v.get("height", 0)) if v else 0,
        "vcodec": v.get("codec_name") if v else None,
        "acodec": a.get("codec_name") if a else None,
    }


# Whisper resamples everything to 16 kHz mono and works on a log-mel
# spectrogram, so lossless audio buys no accuracy - only upload size. Opus at
# 32 kbps in speech mode is transparent for that purpose and roughly 10x
# smaller than FLAC, which keeps every chunk far below Groq's 25 MB limit.
ASR_EXT = ".ogg"
_ASR_ENCODE = ["-vn", "-ac", "1", "-ar", "16000",
               "-c:a", "libopus", "-b:a", "32k", "-application", "voip"]


def extract_audio(src: Path, dst: Path) -> Path:
    """16 kHz mono Opus - the smallest thing Whisper reads without quality loss."""
    run([ffmpeg(), "-y", "-i", str(src), *_ASR_ENCODE, str(dst)], "audio extraction")
    return dst


def slice_audio(src: Path, dst: Path, start: float, duration: float) -> Path:
    run([ffmpeg(), "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
         "-i", str(src), *_ASR_ENCODE, str(dst)], "audio slicing")
    return dst


def thumbnail(src: Path, dst: Path, at: float = 3.0) -> Path | None:
    try:
        run([ffmpeg(), "-y", "-ss", f"{max(0.0, at):.2f}", "-i", str(src),
             "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", str(dst)])
        return dst if dst.exists() else None
    except Exception:
        return None


def decode_pcm(path: Path, atempo: float = 1.0) -> np.ndarray:
    """Decode any audio file to float32 mono @ SR, optionally time-compressed.

    atempo is chained because a single ffmpeg atempo stage only accepts
    0.5–2.0; chaining keeps pitch intact at any ratio.
    """
    filters = []
    t = float(atempo)
    if abs(t - 1.0) > 1e-3:
        while t > 2.0:
            filters.append("atempo=2.0")
            t /= 2.0
        while t < 0.5:
            filters.append("atempo=0.5")
            t /= 0.5
        filters.append(f"atempo={t:.6f}")

    cmd = [ffmpeg(), "-v", "error", "-i", str(path)]
    if filters:
        cmd += ["-filter:a", ",".join(filters)]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]

    p = subprocess.run(cmd, capture_output=True, **_NOWIN)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore")[-800:])
    return np.frombuffer(p.stdout, dtype="<f4").astype(np.float32)


def duration_of(path: Path) -> float:
    try:
        return float(probe(path)["format"]["duration"])
    except Exception:
        return len(decode_pcm(path)) / SR


def write_wav(samples: np.ndarray, dst: Path) -> Path:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0.995:                      # only touch it if we actually clip
        samples = samples * (0.995 / peak)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return dst


def place(track: np.ndarray, clip: np.ndarray, start_s: float) -> None:
    """Mix `clip` into `track` at an exact sample offset, in place."""
    i = max(0, int(round(start_s * SR)))
    n = min(len(clip), len(track) - i)
    if n > 0:
        track[i:i + n] += clip[:n]


def fade(clip: np.ndarray, ms: float = 8.0) -> np.ndarray:
    """Short in/out ramp so butt-joined lines don't click."""
    n = min(int(SR * ms / 1000.0), len(clip) // 2)
    if n <= 0:
        return clip
    out = clip.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def mux(video: Path, dub: Path, dst: Path, original: Path | None = None,
        bed_gain: float = 0.0, subs: Path | None = None) -> Path:
    """Replace the video's audio with the dub track (optionally over a quiet
    bed of the original), keeping the video stream untouched."""
    cmd = [ffmpeg(), "-y", "-i", str(video), "-i", str(dub)]
    use_bed = original is not None and bed_gain > 0.001

    if use_bed:
        cmd += ["-i", str(original)]
        cmd += ["-filter_complex",
                f"[2:a]volume={bed_gain:.3f},aresample={SR}[bed];"
                f"[1:a][bed]amix=inputs=2:duration=first:normalize=0[aout]"]
        amap = ["-map", "[aout]"]
    else:
        amap = ["-map", "1:a"]

    vfilter = []
    if subs is not None:
        # ffmpeg's subtitles filter needs an escaped, forward-slashed path
        esc = str(subs).replace("\\", "/").replace(":", "\:")
        vfilter = ["-vf", f"subtitles='{esc}':force_style='FontSize=18'"]

    cmd += ["-map", "0:v:0"] + amap
    cmd += vfilter
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] if vfilter \
        else ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-ac", "2",
            "-movflags", "+faststart", "-shortest", str(dst)]
    run(cmd, "muxing")
    return dst


def to_wav(src: Path, dst: Path, sr: int = SR) -> Path:
    """Decode any audio file to mono PCM wav - the format RVC expects."""
    run([ffmpeg(), "-y", "-v", "error", "-i", str(src),
         "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)])
    return dst


def to_mp3(src: Path, dst: Path, bitrate: str = "192k") -> Path:
    run([ffmpeg(), "-y", "-v", "error", "-i", str(src), "-b:a", bitrate, str(dst)])
    return dst
