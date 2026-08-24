"""Live HLS output: the dub becomes watchable while it is still being made.

The video and the dub are published as two separate HLS renditions, which is
what makes this cheap:

* **Video** never changes, so it is segmented once, straight from the source
  with `-c:v copy`. Measured on a 54 minute 720p episode: 0.94 seconds for the
  whole file. The picture is therefore complete and seekable before any dubbing
  has happened at all.
* **Audio** is produced a window at a time and fed to ONE long-running encoder
  over a pipe, which appends segments to a growing playlist as they fill.

Feeding one continuous encoder rather than encoding each window separately is
the part that matters. AAC carries encoder priming at the start of every
independent encode, so per-window encodes accumulate: measured at +0.427s over
twenty minutes, which is visible lip-sync error by the end of an hour. A single
stream measured +0.064s over ten minutes, all of it in the final flush, and
does not accumulate.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np

from .config import ffmpeg
from .media import SR, _NOWIN

SEGMENT_SECONDS = 6


class LiveStream:
    """Publishes a video as watchable HLS while its dub is still being built."""

    def __init__(self, folder: Path, source: Path, duration: float):
        self.dir = folder / "hls"
        self.source = source
        self.duration = duration
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._fed = 0.0            # seconds of dub audio handed to the encoder
        self._closed = False

    # ------------------------------------------------------------- setup ---
    def start(self) -> None:
        """Segment the picture and open the audio encoder. Cheap - call early."""
        if self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
        (self.dir / "v").mkdir(parents=True, exist_ok=True)
        (self.dir / "a").mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [ffmpeg(), "-y", "-v", "error", "-i", str(self.source),
             "-map", "0:v:0", "-c:v", "copy", "-an",
             "-f", "hls", "-hls_time", str(SEGMENT_SECONDS),
             # The picture is finished the moment this returns, but it is
             # published as EVENT with no end marker to match the audio beside
             # it. A player given one finished rendition and one still growing
             # cannot tell whether the stream has ended, and hls.js responds by
             # refetching the audio playlist in a tight loop without ever
             # loading a segment.
             "-hls_playlist_type", "event", "-hls_list_size", "0",
             "-hls_flags", "omit_endlist",
             "-hls_segment_type", "mpegts",
             "-hls_segment_filename", str(self.dir / "v" / "v%05d.ts"),
             str(self.dir / "v" / "video.m3u8")],
            check=True, capture_output=True, **_NOWIN)

        self._proc = subprocess.Popen(
            [ffmpeg(), "-y", "-v", "error",
             "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
             "-c:a", "aac", "-b:a", "128k",
             "-f", "hls", "-hls_time", str(SEGMENT_SECONDS),
             "-hls_playlist_type", "event", "-hls_list_size", "0",
             "-hls_segment_type", "mpegts",
             # omit_endlist tells the player more is coming rather than that
             # the video ended here. Not append_list: start() wipes the folder,
             # so there is nothing to append to, and it makes ffmpeg open the
             # playlist with EXT-X-DISCONTINUITY - which stops hls.js lining
             # the audio up against the picture at all.
             "-hls_flags", "omit_endlist",
             "-hls_segment_filename", str(self.dir / "a" / "a%05d.ts"),
             str(self.dir / "a" / "audio.m3u8")],
            # stderr goes to the void rather than a pipe nobody reads: this
            # process outlives the whole dub, and a pipe left unread fills its
            # buffer and blocks ffmpeg forever, which would in turn block every
            # write of finished audio into it.
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, **_NOWIN)

        self._write_master()
        self._publish(done=False)

    def _write_master(self) -> None:
        (self.dir / "master.m3u8").write_text(
            "#EXTM3U\n"
            "#EXT-X-VERSION:3\n"
            '#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="dub",NAME="English",'
            'DEFAULT=YES,AUTOSELECT=YES,URI="a/audio.m3u8"\n'
            '#EXT-X-STREAM-INF:BANDWIDTH=2000000,AUDIO="dub"\n'
            "v/video.m3u8\n", encoding="utf-8")

    # -------------------------------------------------------------- feed ---
    def feed(self, pcm: np.ndarray) -> None:
        """Hand the next stretch of finished dub audio to the encoder.

        Windows must arrive in order and butt up against each other exactly -
        the encoder has no notion of timestamps, only of how much audio it has
        been given, so a gap here is a permanent sync error from that point on.
        """
        if self._closed or self._proc is None or self._proc.stdin is None:
            return
        with self._lock:
            try:
                self._proc.stdin.write(np.asarray(pcm, dtype=np.float32).tobytes())
                self._proc.stdin.flush()
                self._fed += len(pcm) / SR
            except (BrokenPipeError, OSError):
                self._closed = True
        self._publish(done=False)

    def _publish(self, done: bool) -> None:
        """Record how much is watchable, for the library list and the player."""
        try:
            (self.dir / "ready.json").write_text(
                json.dumps({"seconds": round(self._fed, 2),
                            "duration": round(self.duration, 2),
                            "done": done}), encoding="utf-8")
        except OSError:
            pass

    @property
    def ready_seconds(self) -> float:
        """How much of the video currently has a dub on it."""
        return self._fed

    # ------------------------------------------------------------- close ---
    def finish(self) -> None:
        """Close the stream and mark the playlist complete."""
        with self._lock:
            if self._closed or self._proc is None:
                return
            self._closed = True
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=120)
            except Exception:
                self._proc.kill()

        # omit_endlist left both playlists open so players kept waiting for
        # more; now that there is no more, say so on both or they stall on the
        # last segment. They must agree - see the note in start().
        for pl in (self.dir / "a" / "audio.m3u8", self.dir / "v" / "video.m3u8"):
            try:
                text = pl.read_text(encoding="utf-8")
                if "#EXT-X-ENDLIST" not in text:
                    pl.write_text(text.rstrip() + "\n#EXT-X-ENDLIST\n", encoding="utf-8")
            except OSError:
                pass
        self._publish(done=True)

    def abandon(self) -> None:
        """Stop the encoder without pretending the stream completed."""
        with self._lock:
            self._closed = True
            if self._proc is None:
                return
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=20)
            except Exception:
                self._proc.kill()


def playable(folder: Path) -> bool:
    """Is there a stream here a player could open?"""
    return (folder / "hls" / "master.m3u8").is_file()


def status(folder: Path) -> dict:
    """How much of this video is dubbed and watchable right now."""
    try:
        return json.loads((folder / "hls" / "ready.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
