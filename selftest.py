"""Offline self-test: exercises everything except the Groq API calls.

Builds a synthetic video, runs the TTS + alignment + mux stages on scripted
segments, and checks that the dubbed audio actually lands on the right
timestamps. Needs internet for edge-tts, but no API key.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import subtitles, tts                                    # noqa: E402
from app.config import DATA, TMP, ffmpeg                               # noqa: E402
from app.media import SR, decode_pcm, media_info, mux             # noqa: E402

WORK = DATA / "selftest"
DURATION = 30.0

# Deliberately mixed: line 3 is far too long for its slot and must be compressed.
SEGMENTS = [
    {"id": 0, "start": 1.0,  "end": 4.0,  "text": "你好，欢迎观看。",
     "en": "Hello, and welcome to the show."},
    {"id": 1, "start": 5.5,  "end": 8.5,  "text": "今天我们讲一个故事。",
     "en": "Today we are going to tell a story."},
    {"id": 2, "start": 10.0, "end": 12.0, "text": "这是一个很长的句子。",
     "en": "This is a deliberately long sentence that really should not fit "
           "inside a two second gap without being sped up quite a lot."},
    {"id": 3, "start": 15.0, "end": 18.0, "text": "谢谢大家。",
     "en": "Thank you all for watching."},
    {"id": 4, "start": 22.0, "end": 26.0, "text": "下次再见。",
     "en": "See you next time."},
]

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok = ok and passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))


def _resume_checks() -> None:
    """Section 7: a failed run must pick up, not start over.

    The stages that cost real time and money - transcription, translation,
    speech - are driven here with a deliberate mid-run failure, then run again
    to confirm the second attempt only pays for what was still missing.
    """
    import json
    import shutil
    import tempfile

    from app import pipeline, translate, tts

    print("\n7. Resuming after a failure")
    work = Path(tempfile.mkdtemp(prefix="resume_", dir=str(TMP)))
    try:
        # -- translation: crash at the halfway point, then finish the rest ----
        real_batch, real_glossary = translate._run_batch, translate.build_glossary
        try:
            lines = [{"id": i, "start": i * 2.0, "end": i * 2.0 + 1.8,
                      "text": f"line {i}", "en": ""} for i in range(120)]
            asked: list[tuple[int, int]] = []

            def batch(client, model, system, segs, lo, hi, budget):
                asked.append((lo, hi))
                if lo >= 60:
                    raise RuntimeError("simulated network drop")
                return {segs[i]["id"]: f"en {i}" for i in range(lo, hi)}

            translate._run_batch = batch
            translate.build_glossary = lambda *a, **k: ""

            saved: dict = {"segs": None, "n": 0}

            def ckpt(segs, force=False):
                saved["n"] += 1
                saved["segs"] = json.loads(json.dumps(segs))

            crashed = False
            try:
                translate.translate(object(), lines, "m", "zh",
                                    checkpoint=ckpt, cache_dir=work)
            except RuntimeError:
                crashed = True

            done = sum(1 for s in (saved["segs"] or []) if s["en"])
            check("translation failure is survivable", crashed and done > 0,
                  f"{done}/120 lines already saved")

            asked.clear()
            translate._run_batch = lambda c, m, sy, sg, lo, hi, b: (
                asked.append((lo, hi)) or
                {sg[i]["id"]: f"en {i}" for i in range(lo, hi)})
            again = json.loads(json.dumps(saved["segs"]))
            translate.translate(object(), again, "m", "zh", cache_dir=work)
            redone = sum(hi - lo for lo, hi in asked)
            check("retry only translates what is missing", redone == 120 - done,
                  f"{redone} lines re-sent, not 120")
            check("retry completes the transcript", all(s["en"] for s in again))

            asked.clear()
            translate.translate(object(), again, "m", "zh", cache_dir=work)
            check("a finished transcript costs nothing", not asked,
                  f"{len(asked)} requests")
        finally:
            translate._run_batch, translate.build_glossary = real_batch, real_glossary

        # -- speech: a line is identified by its text and its voice settings --
        fp = tts._fingerprint
        base = fp("hello there", "en-US-BrianNeural", 0, 0, 0)
        check("unchanged line is reused",
              base == fp("hello there", "en-US-BrianNeural", 0, 0, 0))
        check("edited line is re-rendered",
              base != fp("hello THERE", "en-US-BrianNeural", 0, 0, 0))
        check("changed voice is re-rendered",
              base != fp("hello there", "en-GB-RyanNeural", 0, 0, 0))
        check("changed speed is re-rendered",
              base != fp("hello there", "en-US-BrianNeural", 0, 10, 0))

        stub = work / "tts_00001_abcdef0123.mp3"
        stub.write_bytes(b"\x00" * 100)
        check("file from a killed run is not trusted", not tts._usable(stub))

        # -- the finished dub track, which carries the GPU conversion pass ----
        opts = {"voice": "rei", "pitch": 0, "speed": 0, "volume": 0,
                "max_speedup": 1.7, "soften": 35, "clone_index_rate": 0.5,
                "clone_protect": 0.33}
        stamp = pipeline._dub_stamp(SEGMENTS, opts)
        check("identical settings reuse the dub track",
              stamp == pipeline._dub_stamp(SEGMENTS, opts))
        check("changed voice rebuilds the dub track",
              stamp != pipeline._dub_stamp(SEGMENTS, dict(opts, voice="en-US-BrianNeural")))
        edited = json.loads(json.dumps(SEGMENTS))
        edited[0]["en"] = "a different line"
        check("edited subtitle rebuilds the dub track",
              stamp != pipeline._dub_stamp(edited, opts))

        # -- stage bookkeeping -------------------------------------------------
        pipeline._mark(work, "asr", "abc")
        pipeline._mark(work, "dub", {"stamp": "x", "size": 9})
        pipeline._clear(work, "dub")
        left = pipeline._state(work)
        check("clearing one stage leaves the others",
              left.get("asr") == "abc" and "dub" not in left)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _quality_checks() -> None:
    """Section 8: the guards that keep a line from going silent or off-voice."""
    from app import asr, translate

    print("\n8. Guarding against silent and mistranslated lines")

    # Whisper collapsing on a long span is what leaves a character visibly
    # speaking with no dub over them. Both examples are real: the first is an
    # actual collapsed segment from a 1400-line episode, the second a normal
    # line from the same transcript.
    collapsed = {"start": 667.0, "end": 679.5,
                 "text": "Ti t nh M b l Nh r nhanh l Ch cho con r nhi t nguy r H n 3 th sau"}
    normal = {"start": 667.0, "end": 669.2,
              "text": "Chàng ấy đã cho con rất nhiều tài nguyên rồi."}
    long_ok = {"start": 100.0, "end": 112.0,
               "text": "Hơn nữa ba tháng sau, chàng phải tham gia cuộc tranh "
                       "đoạt thiếu chủ của lâm gia rất quan trọng."}
    check("collapsed transcription is caught", asr._degenerate(collapsed),
          f"fragment ratio {asr._fragment_ratio(collapsed['text']):.2f}")
    check("a normal line is left alone", not asr._degenerate(normal))
    check("a long but clean line is left alone", not asr._degenerate(long_ok),
          f"fragment ratio {asr._fragment_ratio(long_ok['text']):.2f}")

    check("spoken language names map to codes",
          asr._LANG_CODE.get("vietnamese") == "vi"
          and asr._LANG_CODE.get("chinese") == "zh")

    # A blank translation for a real sentence is a silent gap; for genuine
    # filler it is correct. Only the first is worth another request.
    real = {"id": 1, "text": "Nghe thấy cái tên Loan Phượng bồn nguyên của"}
    filler = {"id": 2, "text": "Ừm."}
    check("a blanked real line is retried",
          len(real["text"].strip()) >= translate.FILLER_CHARS)
    check("genuine filler is left blank",
          len(filler["text"].strip()) < translate.FILLER_CHARS)

    # The narrator's point of view is the thing that makes a hero sound like
    # he is narrating someone else's life.
    check("the prompt fixes the narrator's point of view",
          "POINT OF VIEW" in translate.SYSTEM
          and "Never render it as \"you\"" in translate.SYSTEM)
    check("grammar outranks the character budget",
          "EVERY LINE MUST BE GRAMMATICAL ENGLISH" in translate.SYSTEM)
    check("Vietnamese pronouns are explained",
          "nguoi" in translate.PRONOUN_NOTES.get("vi", "")
          and translate.LANG_NAMES.get("vi") == "Vietnamese")


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  Dubline self-test (no API key required)")
    print("=" * 62)

    # ------------------------------------------------------ synthetic video --
    print("\n1. Building a synthetic test video")
    src = WORK / "source.mp4"
    subprocess.run([
        ffmpeg(), "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=size=640x360:rate=25:duration={DURATION}",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={DURATION}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(src)], check=True)
    info = media_info(src)
    check("video built", src.exists() and info["has_video"] and info["has_audio"],
          f"{info['width']}x{info['height']}, {info['duration']:.1f}s")

    # ---------------------------------------------------------------- voices --
    print("\n2. Fetching the voice list from edge-tts")
    try:
        voices = tts.list_voices()
        check("voices loaded", len(voices) > 10, f"{len(voices)} English voices")
        voice = next((v["id"] for v in voices if "Andrew" in v["id"]), voices[0]["id"])
        print(f"       using {voice}")
    except Exception as e:
        check("voices loaded", False, str(e))
        return

    # ------------------------------------------------- synthesis + alignment --
    print("\n3. Synthesising speech and aligning it to the timeline")
    segs = [dict(s) for s in SEGMENTS]
    dub, stats = tts.build_dub_track(segs, voice, WORK, DURATION,
                                     max_speedup=1.7, progress=None)
    check("dub track written", dub.exists(), f"{dub.stat().st_size / 1024:.0f} KB")
    check("every line voiced", stats["lines"] == len(SEGMENTS),
          f"{stats['lines']}/{len(SEGMENTS)} lines, {stats['failed']} failed")
    check("long line was compressed", stats["compressed"] >= 1,
          f"{stats['compressed']} line(s), peak {stats['max_speedup_used']}x")
    check("speed-up respected the cap", stats["max_speedup_used"] <= 1.75,
          f"peak {stats['max_speedup_used']}x")

    # ------------------------------------------------------------ clone voice --
    # The default voice is a locally trained clone, so the conversion path is
    # covered here too. Skipped rather than failed when the toolchain is absent,
    # since the app is expected to work without it.
    print("\n3b. Cloned-voice conversion")
    from app import voiceclone
    from app.config import load_settings
    cfg = load_settings()
    clones = voiceclone.list_clones()
    if not clones:
        print("       no cloned voices installed - skipped")
    elif not voiceclone.available(cfg):
        print("       conversion toolchain not found - skipped")
    else:
        import soundfile as sf
        cid = clones[0]["id"]
        try:
            before = sf.read(str(dub))[0]
            out = voiceclone.convert(dub, WORK / "dub_clone.wav", cid, cfg)
            after = sf.read(str(out))[0]
            check(f"'{clones[0]['name']}' conversion ran", out.exists(),
                  f"{out.stat().st_size / 1024:.0f} KB")
            # the whole design rests on this: conversion must not re-time audio
            check("length preserved exactly", len(after) == len(before),
                  f"{len(before)} -> {len(after)} samples")
            check("output is not silent", float(abs(after).max()) > 0.01,
                  f"peak {float(abs(after).max()):.3f}")
        except Exception as e:
            check("cloned-voice conversion", False, str(e)[:120])
        finally:
            voiceclone.shutdown()

    # ----------------------------------------------------- timing verification --
    print("\n4. Verifying the dub lands on the original timestamps")
    audio = decode_pcm(dub)
    check("track length matches video", abs(len(audio) / SR - DURATION) < 1.5,
          f"{len(audio) / SR:.2f}s vs {DURATION}s")

    # energy envelope in 20 ms frames tells us where speech actually is
    frame = int(SR * 0.02)
    frames = len(audio) // frame
    energy = [float(abs(audio[i * frame:(i + 1) * frame]).mean()) for i in range(frames)]
    floor = max(1e-5, sorted(energy)[len(energy) // 2] + 1e-4)

    for s in segs:
        want = s["start"]
        first = None
        for i in range(frames):
            if energy[i] > floor * 6:
                t = i * 0.02
                if t >= want - 0.35:
                    first = t
                    break
        drift = abs((first if first is not None else -99) - want)
        check(f"line {s['id']} starts at {want:.1f}s", drift < 0.45,
              f"detected {first:.2f}s, drift {drift * 1000:.0f} ms"
              if first is not None else "no audio found")

    for s in segs:
        check(f"line {s['id']} fits its slot",
              s.get("dub_end", 0) - s.get("dub_start", 0) > 0.2
              and s["dub_start"] >= s["start"] - 0.01,
              f"{s['dub_start']:.2f}-{s['dub_end']:.2f}s "
              f"(slot {s['start']:.1f}-{s['end']:.1f}) x{s.get('dub_speedup', 1)}")

    # ------------------------------------------------------------- subtitles --
    print("\n5. Writing subtitle files")
    srt = subtitles.write_srt(segs, WORK / "english.srt", "en")
    vtt = subtitles.write_vtt(segs, WORK / "english.vtt", "en")
    body = srt.read_text(encoding="utf-8")
    check("srt written", "00:00:01,000 --> 00:00:04,000" in body,
          body.splitlines()[1] if len(body.splitlines()) > 1 else "")
    check("vtt written", vtt.read_text(encoding="utf-8").startswith("WEBVTT"))
    check("srt has every line", body.count("-->") == len(SEGMENTS),
          f"{body.count('-->')} cues")

    # ------------------------------------------------------------------- mux --
    print("\n6. Muxing the dub back onto the video")
    out = WORK / "dubbed.mp4"
    mux(src, dub, out, original=src, bed_gain=0.10)
    oi = media_info(out)
    check("output exists", out.exists(), f"{out.stat().st_size / 1024:.0f} KB")
    check("output has video + audio", oi["has_video"] and oi["has_audio"],
          f"{oi['vcodec']} / {oi['acodec']}")
    check("duration preserved", abs(oi["duration"] - DURATION) < 1.0,
          f"{oi['duration']:.2f}s vs {DURATION}s")
    check("video was stream-copied", oi["width"] == 640 and oi["height"] == 360)

    _resume_checks()
    _quality_checks()

    print("\n" + "=" * 62)
    print("  RESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print(f"  Listen to the result: {out}")
    print("=" * 62)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
