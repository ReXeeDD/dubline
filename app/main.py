"""FastAPI app: upload, process, browse and play dubbed videos."""
from __future__ import annotations

import mimetypes
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from . import library, llm, media, pipeline, subtitles, translate, tts, voiceclone
from .config import (LIBRARY, ROOT, key_locations, load_settings,
                     save_settings)

app = FastAPI(title="Dubline", docs_url=None, redoc_url=None)
STATIC = ROOT / "static"

# One worker: ffmpeg and Whisper uploads are heavy, and a queue keeps the
# machine responsive while several videos are waiting.
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dub")

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".mpg", ".mpeg",
             ".ts", ".flv", ".wmv"}


@app.on_event("startup")
def _startup() -> None:
    library.init()
    # anything left mid-flight by a previous run is not actually running
    for v in library.all_videos():
        if v["status"] == "processing":
            library.update(v["id"], status="failed",
                           error="Interrupted - the server restarted.",
                           stage="Interrupted")


# ------------------------------------------------------------------- pages ---
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# -------------------------------------------------------------- ranged file ---
def ranged(path: Path, request: Request, media_type: str | None = None) -> Response:
    """Serve a file honouring Range requests so the player can seek."""
    if not path.exists():
        raise HTTPException(404, "File not found")

    size = path.stat().st_size
    media_type = media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(str(path), media_type=media_type,
                            headers={"Accept-Ranges": "bytes"})

    m = re.match(r"bytes=(\d*)-(\d*)", range_header)
    if not m:
        raise HTTPException(416, "Malformed range header")

    start = int(m.group(1)) if m.group(1) else 0
    end = int(m.group(2)) if m.group(2) else size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def stream():
        remaining = end - start + 1
        with path.open("rb") as f:
            f.seek(start)
            while remaining > 0:
                data = f.read(min(1024 * 1024, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        stream(), status_code=206, media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
        },
    )


# ------------------------------------------------------------------- media ---
@app.get("/media/{vid}/video")
def media_video(vid: str, request: Request, src: int = 0):
    folder = LIBRARY / vid
    if src:
        p = library.source_file(vid)
        if p is None:
            raise HTTPException(404, "Original video not found")
    else:
        p = folder / "dubbed.mp4"
    return ranged(p, request, "video/mp4")


@app.get("/media/{vid}/thumb.jpg")
def media_thumb(vid: str):
    p = LIBRARY / vid / "thumb.jpg"
    if not p.exists():
        raise HTTPException(404, "No thumbnail")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/media/{vid}/{name}")
def media_file(vid: str, name: str):
    if not re.fullmatch(r"(english|original)\.(srt|vtt)", name):
        raise HTTPException(404, "Not found")
    p = LIBRARY / vid / name
    if not p.exists():
        raise HTTPException(404, "Not found")
    mt = "text/vtt" if name.endswith(".vtt") else "application/x-subrip"
    return FileResponse(str(p), media_type=mt, filename=name,
                        headers={"Cache-Control": "no-cache"})


# --------------------------------------------------------------- library api ---
@app.get("/api/videos")
def api_videos():
    return library.all_videos()


@app.get("/api/videos/{vid}")
def api_video(vid: str):
    v = library.get(vid)
    if not v:
        raise HTTPException(404, "Video not found")
    v["segments"] = library.load_segments(vid)
    return v


@app.get("/api/videos/{vid}/status")
def api_status(vid: str):
    v = library.get(vid)
    if not v:
        raise HTTPException(404, "Video not found")
    return {k: v[k] for k in ("id", "status", "stage", "progress", "error",
                              "line_count", "duration")}


@app.delete("/api/videos/{vid}")
def api_delete(vid: str):
    v = library.get(vid)
    if v and v["status"] == "processing":
        raise HTTPException(409, "This video is still processing.")
    if not library.delete(vid):
        raise HTTPException(404, "Video not found")
    return {"ok": True}


@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    title: str = Form(""),
    voice: str = Form(""),
    source_language: str = Form(""),
):
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"{ext or 'That file type'} is not a supported video format.")

    cfg = load_settings()
    opts = dict(cfg)
    if voice:
        opts["voice"] = voice
    if source_language:
        opts["source_language"] = source_language

    name = Path(file.filename or "video").stem
    vid = library.create(
        title=(title.strip() or name)[:200],
        original_name=file.filename or "video",
        source_lang=opts["source_language"], voice=opts["voice"],
        asr_model=opts["asr_model"], llm_model=llm.translation_model(opts),
    )

    dest = library.vdir(vid) / f"source{ext}"
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(4 * 1024 * 1024):
                out.write(chunk)
    except Exception as e:
        library.delete(vid)
        raise HTTPException(500, f"Upload failed: {e}")
    finally:
        await file.close()

    if dest.stat().st_size == 0:
        library.delete(vid)
        raise HTTPException(400, "The uploaded file was empty.")

    library.update(vid, stage="Queued", status="queued")
    POOL.submit(pipeline.process, vid, opts)
    return {"id": vid}


@app.post("/api/videos/{vid}/retry")
def api_retry(vid: str):
    v = library.get(vid)
    if not v:
        raise HTTPException(404, "Video not found")
    if v["status"] == "processing":
        raise HTTPException(409, "Already processing.")
    opts = dict(load_settings())
    opts["source_language"] = v.get("source_lang") or opts["source_language"]
    opts["voice"] = v.get("voice") or opts["voice"]
    library.update(vid, status="queued", stage="Queued", progress=0, error=None)
    POOL.submit(pipeline.process, vid, opts)
    return {"ok": True}


@app.post("/api/videos/{vid}/revoice")
def api_revoice(vid: str, body: dict):
    v = library.get(vid)
    if not v:
        raise HTTPException(404, "Video not found")
    if v["status"] == "processing":
        raise HTTPException(409, "Already processing.")
    if not library.load_segments(vid):
        raise HTTPException(400, "No transcript yet - run the full pipeline first.")

    opts = dict(load_settings())
    if body.get("voice"):
        opts["voice"] = body["voice"]
    for k in ("keep_original_audio", "original_audio_gain", "max_speedup",
              "burn_subtitles", "pitch", "speed", "volume", "soften"):
        if k in body:
            opts[k] = body[k]

    library.update(vid, status="queued", stage="Queued", progress=0, error=None,
                   voice=opts["voice"])
    POOL.submit(pipeline.revoice, vid, opts)
    return {"ok": True}


@app.post("/api/videos/{vid}/retranslate")
def api_retranslate(vid: str):
    """Re-run translation on the existing transcript - no new Whisper call."""
    v = library.get(vid)
    if not v:
        raise HTTPException(404, "Video not found")
    if v["status"] == "processing":
        raise HTTPException(409, "Already processing.")
    if not library.load_segments(vid):
        raise HTTPException(400, "No transcript yet - run the full pipeline first.")

    opts = dict(load_settings())
    opts["source_language"] = v.get("source_lang") or opts["source_language"]
    opts["voice"] = v.get("voice") or opts["voice"]
    stats = v.get("stats") or {}
    for k in ("pitch", "speed", "volume", "soften"):
        if k in stats:
            opts[k] = stats[k]

    library.update(vid, status="queued", stage="Queued", progress=0, error=None)
    POOL.submit(pipeline.retranslate, vid, opts)
    return {"ok": True}


@app.post("/api/videos/{vid}/remix")
def api_remix(vid: str, body: dict):
    """Change the audio mix without re-synthesising speech."""
    v = library.get(vid)
    if not v:
        raise HTTPException(404, "Video not found")
    if v["status"] == "processing":
        raise HTTPException(409, "Already processing.")

    opts = dict(load_settings())
    for k in ("keep_original_audio", "original_audio_gain", "burn_subtitles"):
        if k in body:
            opts[k] = body[k]

    library.update(vid, status="queued", stage="Queued", progress=0, error=None)
    POOL.submit(pipeline.remix, vid, opts)
    return {"ok": True}


@app.put("/api/videos/{vid}/segments")
def api_edit_segments(vid: str, body: dict):
    """Save subtitle edits. Re-dubbing is a separate, explicit step."""
    if not library.get(vid):
        raise HTTPException(404, "Video not found")
    segments = library.load_segments(vid)
    edits = {int(k): str(v) for k, v in (body.get("edits") or {}).items()}
    changed = 0
    for s in segments:
        if s["id"] in edits and s.get("en") != edits[s["id"]]:
            s["en"] = edits[s["id"]].strip()
            changed += 1
    library.save_segments(vid, segments)
    folder = library.vdir(vid)
    subtitles.write_srt(segments, folder / "english.srt", "en")
    subtitles.write_vtt(segments, folder / "english.vtt", "en")
    return {"ok": True, "changed": changed}


@app.post("/api/videos/{vid}/retranslate-line")
def api_retranslate_line(vid: str, body: dict):
    segments = library.load_segments(vid)
    seg = next((s for s in segments if s["id"] == int(body["id"])), None)
    if seg is None:
        raise HTTPException(404, "Line not found")
    cfg = load_settings()
    text = translate.retranslate_one(
        llm.make_translation_client(cfg), seg, llm.translation_model(cfg),
        cfg["source_language"])
    return {"id": seg["id"], "en": text}


# -------------------------------------------------------------- voices/cfg ---
@app.get("/api/voices")
def api_voices():
    try:
        return tts.list_voices()
    except Exception as e:
        raise HTTPException(503, f"Could not load the voice list: {e}")


@app.post("/api/voices/preview")
def api_voice_preview(body: dict):
    from .config import DATA

    cfg = load_settings()
    voice = body.get("voice") or cfg["voice"]
    pitch = max(-50, min(30, int(body.get("pitch", cfg.get("pitch", 0)))))
    speed = max(-40, min(50, int(body.get("speed", cfg.get("speed", 0)))))
    volume = max(-50, min(50, int(body.get("volume", cfg.get("volume", 0)))))
    cfg["soften"] = max(0, min(100, int(body.get("soften", cfg.get("soften", 0)))))
    text = (body.get("text") or
            "This is how the English dub of your video will sound.")[:300]
    tmp = DATA / "previews"
    tmp.mkdir(exist_ok=True)
    dst = tmp / f"{uuid.uuid4().hex}.mp3"
    wav = dst.with_suffix(".wav")
    conv = dst.with_name(dst.stem + "_c.wav")
    try:
        # tts.preview resolves a cloned voice to its base voice on its own, so
        # this renders what will actually be spoken before conversion.
        tts.preview(text, voice, dst, pitch=pitch, rate_pct=speed, volume=volume)

        if voiceclone.is_clone(voice):
            media.to_wav(dst, wav)
            # Short wait: a preview should say the converter is busy rather
            # than hang behind a whole episode.
            voiceclone.convert(wav, conv, voice, cfg, wait=90.0)
            media.to_mp3(conv, dst)

        data = dst.read_bytes()
    except Exception as e:
        raise HTTPException(502, f"Voice preview failed: {e}")
    finally:
        for f in (dst, wav, conv):
            f.unlink(missing_ok=True)
    return Response(data, media_type="audio/mpeg")


@app.get("/api/settings")
def api_get_settings():
    cfg = load_settings()
    key = cfg.get("groq_api_key") or ""
    cfg["groq_api_key"] = (key[:4] + "..." + key[-4:]) if len(key) > 12 else ""
    cfg["has_key"] = bool(key)
    # Extra keys are masked the same way. Sending them back in full would put
    # someone else's credential into the page for no reason.
    cfg["groq_api_keys"] = [(k[:4] + "..." + k[-4:]) if len(k) > 12 else ""
                            for k in (cfg.get("groq_api_keys") or [])]
    cfg["key_storage"] = key_locations()
    return cfg


@app.post("/api/settings")
def api_set_settings(body: dict):
    if "groq_api_key" in body and "..." in str(body["groq_api_key"]):
        body.pop("groq_api_key")     # masked value echoed back - leave it alone

    # Same for the extras, but position matters: a masked entry means "keep the
    # key that was in this slot", so the stored list is matched up by index.
    if "groq_api_keys" in body:
        stored = load_settings().get("groq_api_keys") or []
        kept = []
        for i, k in enumerate(body["groq_api_keys"] or []):
            k = str(k or "").strip()
            if "..." in k:
                k = stored[i] if i < len(stored) else ""
            if k:
                kept.append(k)
        body["groq_api_keys"] = kept

    save_settings(body)
    return api_get_settings()


@app.post("/api/settings/test-key")
def api_test_key(body: dict):
    """Validate a key against Groq. Saves it only if it works."""
    key = str(body.get("key") or body.get("groq_api_key") or "").strip()
    if not key or "..." in key:
        key = load_settings().get("groq_api_key", "")
    result = llm.verify_key(key)
    if result["ok"] and body.get("save"):
        save_settings({"groq_api_key": key})
        result["storage"] = key_locations()
    return result


@app.get("/api/models")
def api_models(local: int = 0):
    """`local=1` also probes LM Studio. Off by default so the Settings panel
    never waits on a server that usually is not running."""
    cfg = load_settings()
    return {
        "groq": llm.list_groq_models(cfg.get("groq_api_key", "")),
        "local": llm.list_local_models(cfg.get("local_base_url")
                                       or llm.DEFAULT_LOCAL_URL) if local else [],
        "local_probed": bool(local),
        "current": {
            "asr_model": cfg["asr_model"],
            "llm_model": cfg["llm_model"],
            "llm_provider": cfg["llm_provider"],
            "local_model": cfg.get("local_model", ""),
        },
    }


@app.get("/api/health")
def api_health():
    from .config import ffmpeg

    out = {"ffmpeg": False, "groq_key": bool(load_settings().get("groq_api_key"))}
    try:
        ffmpeg()
        out["ffmpeg"] = True
    except Exception as e:
        out["ffmpeg_error"] = str(e)
    return out


@app.exception_handler(404)
def _404(request: Request, exc):
    if request.url.path.startswith(("/api", "/media")):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))
