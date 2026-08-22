"""SRT / WebVTT writers."""
from __future__ import annotations

import textwrap
from pathlib import Path

MAX_LINE = 42   # characters per subtitle line, broadcast convention


def _clock(t: float, comma: bool = True) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    whole = int(s)
    ms = int(round((s - whole) * 1000))
    if ms == 1000:
        whole, ms = whole + 1, 0
    sep = "," if comma else "."
    return f"{int(h):02d}:{int(m):02d}:{whole:02d}{sep}{ms:03d}"


def _wrap(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_LINE:
        return text
    return "\n".join(textwrap.wrap(text, width=MAX_LINE, max_lines=2,
                                   placeholder="…"))


def _rows(segments: list[dict], field: str) -> list[tuple[float, float, str]]:
    rows = []
    for s in segments:
        text = (s.get(field) or "").strip()
        if not text:
            continue
        rows.append((float(s["start"]), float(s["end"]), _wrap(text)))
    return rows


def write_srt(segments: list[dict], dst: Path, field: str = "en") -> Path:
    lines = []
    for i, (start, end, text) in enumerate(_rows(segments, field), 1):
        lines.append(f"{i}\n{_clock(start)} --> {_clock(end)}\n{text}\n")
    dst.write_text("\n".join(lines), encoding="utf-8")
    return dst


def write_vtt(segments: list[dict], dst: Path, field: str = "en") -> Path:
    lines = ["WEBVTT", ""]
    for start, end, text in _rows(segments, field):
        lines.append(f"{_clock(start, False)} --> {_clock(end, False)}")
        lines.append(text)
        lines.append("")
    dst.write_text("\n".join(lines), encoding="utf-8")
    return dst
