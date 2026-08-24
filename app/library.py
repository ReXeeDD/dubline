"""SQLite-backed library of processed videos plus live job state."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .config import DATA, LIBRARY

DB = DATA / "library.db"
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    status        TEXT NOT NULL,
    stage         TEXT,
    progress      INTEGER DEFAULT 0,
    error         TEXT,
    created_at    REAL,
    updated_at    REAL,
    duration      REAL DEFAULT 0,
    source_lang   TEXT,
    voice         TEXT,
    asr_model     TEXT,
    llm_model     TEXT,
    line_count    INTEGER DEFAULT 0,
    stats         TEXT,
    original_name TEXT
);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init() -> None:
    with _lock, _conn() as c:
        c.executescript(SCHEMA)


def vdir(vid: str) -> Path:
    d = LIBRARY / vid
    d.mkdir(parents=True, exist_ok=True)
    return d


def create(title: str, original_name: str, **fields) -> str:
    vid = uuid.uuid4().hex[:12]
    now = time.time()
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO videos (id, title, status, stage, progress, created_at,"
            " updated_at, source_lang, voice, asr_model, llm_model, original_name)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (vid, title, "queued", "Waiting to start", 0, now, now,
             fields.get("source_lang"), fields.get("voice"),
             fields.get("asr_model"), fields.get("llm_model"), original_name),
        )
    vdir(vid)
    return vid


def update(vid: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    if "stats" in fields and not isinstance(fields["stats"], str):
        fields["stats"] = json.dumps(fields["stats"])
    cols = ", ".join(f"{k}=?" for k in fields)
    with _lock, _conn() as c:
        c.execute(f"UPDATE videos SET {cols} WHERE id=?", (*fields.values(), vid))


def get(vid: str) -> dict | None:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
    return _row(row) if row else None


def all_videos() -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
    return [_row(r) for r in rows]


def delete(vid: str) -> bool:
    import shutil

    with _lock, _conn() as c:
        cur = c.execute("DELETE FROM videos WHERE id=?", (vid,))
        gone = cur.rowcount > 0
    shutil.rmtree(LIBRARY / vid, ignore_errors=True)
    return gone


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    try:
        d["stats"] = json.loads(d["stats"]) if d.get("stats") else {}
    except Exception:
        d["stats"] = {}
    folder = LIBRARY / d["id"]
    # How much of the dub is watchable right now. Present while a video is
    # still processing, which is the whole point of it.
    try:
        d["stream"] = json.loads(
            (folder / "hls" / "ready.json").read_text(encoding="utf-8"))
    except Exception:
        d["stream"] = {}
    d["has_output"] = (folder / "dubbed.mp4").exists()
    d["has_thumb"] = (folder / "thumb.jpg").exists()
    d["has_source"] = any((folder / f"source{e}").exists()
                          for e in (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"))
    return d


# ------------------------------------------------------------- segment i/o ---
def save_segments(vid: str, segments: list[dict]) -> None:
    (vdir(vid) / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8")


def load_segments(vid: str) -> list[dict]:
    f = LIBRARY / vid / "segments.json"
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []


def source_file(vid: str) -> Path | None:
    folder = LIBRARY / vid
    for e in (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"):
        p = folder / f"source{e}"
        if p.exists():
            return p
    return None
