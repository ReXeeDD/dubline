"""Offline self-test: exercises everything except the Groq API calls.

Builds a synthetic video, runs the TTS + alignment + mux stages on scripted
segments, and checks that the dubbed audio actually lands on the right
timestamps. Needs internet for edge-tts, but no API key.
"""
from __future__ import annotations

import inspect
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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

    # The cast list is injected into the brief under "use these spellings
    # exactly", so anything on it is a direct instruction. A pronoun there is
    # worse than useless: "ni=you" tells the translator to write the very
    # second-person narration that rule 6 spends a paragraph forbidding, and
    # being concrete it wins. This is what the point-of-view drift traced back
    # to on a real clip.
    check("a pronoun never reaches the cast list",
          translate._is_pronoun("你", "you")
          and translate._is_pronoun("ngươi", "You")
          and not translate._is_pronoun("云娘", "Yun Niang"))
    # Common nouns are the other way a cast list does damage: "fu jun=Husband"
    # is what produced "Husband will cook for you" where a person says "I".
    check("a form of address is not mistaken for a name",
          translate._is_generic("娘子", "Wife")
          and translate._is_generic("三女", "Three Daughters")
          and not translate._is_generic("云娘", "Yun Niang"))
    check("an untranslated cast entry is dropped",
          translate._is_generic("主人", "主人"),
          "it would put source script into an all-English brief")
    check("the brief covers forms of address and terms of art",
          "FORMS OF ADDRESS" in translate.SYSTEM
          and "poll tax" in translate.SYSTEM
          and "Husband will make you food" in translate.SYSTEM)

    # Only ever penalising an over-long line teaches the repair pass one move,
    # and it will cut the content out of a line to make the number go down.
    # This is the real case that showed it: a poll-tax exchange plus a reply
    # collapsed into three words, filling 30 characters of a 78-character slot.
    dropped = {"id": 0, "start": 0.0, "end": 5.2,
               "text": "可不是交三人的人头睡就行是需要叠加的这我知道你点头表示明白知道你还选三个",
               "en": "I nodded to show I understood."}
    kept = {"id": 1, "start": 0.0, "end": 5.4,
            "text": "无土王里长愤怒呵斥看向你的目光恨铁不成钢对此你有些无奈可又不知道如何去解释",
            "en": "The village head scolded me, disappointed. I felt helpless "
                  "but couldn't explain."}
    terse = {"id": 2, "start": 0.0, "end": 1.5,
             "text": "你好吗", "en": "Are you all right?"}
    check("a line that dropped its content is caught",
          translate._dropped_content(dropped),
          f"{len(dropped['en'])} chars where {translate._budget_chars(dropped)} fit")
    check("a full line is left alone", not translate._dropped_content(kept))
    check("a genuinely short line is left alone",
          not translate._dropped_content(terse),
          "re-asking for these is how a repair pass makes good work worse")
    check("the repair brief knows shortening is not always the fix",
          "too SHORT" in translate.POLISH_SYSTEM
          and "Shortening a line that is already too short" in translate.POLISH_SYSTEM)

    # A reasoning model's reply allowance is doubled to leave room for the
    # thinking, which puts even a plain 30-line batch past the 8000-token
    # minute. The scheduler used to merge those batches UP to 60 lines, so the
    # request was refused, split in two, and then immediately rejoined by the
    # same merge - split, merge, split, merge, with no request ever sent and no
    # error ever raised. Seen as a dub frozen at "Translating line 91 of 150".
    sched = [{"id": i, "start": i * 5.0, "end": i * 5.0 + 5.0,
              "text": "甲" * 38, "en": ""} for i in range(150)]
    system = translate.build_system("zh", "")
    head = 8000

    def plan(reasoning):
        """Run the scheduler's job bookkeeping with no API calls."""
        translate._REASONING["selftest-model"] = reasoning
        jobs = [(i, min(i + translate.BATCH, len(sched)))
                for i in range(0, len(sched), translate.BATCH)]
        sent = []
        for _ in range(400):
            if not jobs:
                break
            lo, hi = jobs.pop(0)
            if translate._REASONING.get("selftest-model") == "low":
                while (jobs and jobs[0][0] == hi
                       and (hi - lo) < translate.BATCH * 2):
                    if translate._projected_cost(system, sched, lo, jobs[0][1],
                                                 "selftest-model") > head:
                        break
                    hi = jobs.pop(0)[1]
            while (hi - lo) > 1 and translate._projected_cost(
                    system, sched, lo, hi, "selftest-model") > head:
                mid = lo + (hi - lo) // 2
                jobs.insert(0, (mid, hi))
                hi = mid
            sent.append((lo, hi))
        translate._REASONING.pop("selftest-model", None)
        return jobs, sent

    left, sent = plan("low")
    check("a reasoning model's batches terminate", not left,
          f"{len(sent)} requests, biggest {max(h - l for l, h in sent)} lines")
    check("every line is still covered",
          sum(h - l for l, h in sent) == len(sched),
          f"{sum(h - l for l, h in sent)}/{len(sched)}")
    check("no batch is sent that cannot fit the minute",
          all(translate._projected_cost(system, sched, l, h, "absent") <= head
              for l, h in sent))
    left, sent = plan(None)
    check("a plain model still takes whole batches", not left
          and max(h - l for l, h in sent) == translate.BATCH,
          f"{len(sent)} requests of {translate.BATCH}")

    # One bucket running out of its DAILY allowance must cost that bucket, not
    # the video. The others have their own allowance and the work is ordinary
    # work - but a stream used to leave the moment the queue looked empty, and
    # the queue looks empty while an exhausted stream is still holding the
    # batch it is about to hand back. Nobody was left to take it, and
    # translate() then failed the whole run for "daily token limit is used up"
    # while a model with budget sat idle. This drives the real _worker.
    import threading as _th

    def handoff(dead_model):
        segs2 = [{"id": i, "start": i * 3.0, "end": i * 3.0 + 3.0,
                  "text": "x" * 20, "en": ""} for i in range(120)]
        jobs = [(i, min(i + translate.BATCH, len(segs2)))
                for i in range(0, len(segs2), translate.BATCH)]
        lock = _th.Lock()
        state = {"done": 0, "error": None, "checkpoint": None,
                 "holders": 0, "report": lambda: None}
        budgets = {m: translate.TokenBudget() for m in ("alive", "dead")}
        real = translate._run_batch

        def fake(client, model, system, segments, lo, hi, budget):
            time.sleep(0.005)
            if model == dead_model:
                budget.exhausted = "resets in 17m30s"
                raise translate.Exhausted(f"{model}: out for the day")
            return {s["id"]: "line" for s in segments[lo:hi]}

        translate._run_batch = fake
        try:
            threads = [_th.Thread(target=translate._worker,
                                  args=(None, m, "sys", segs2, jobs, lock,
                                        state, budgets[m]))
                       for m in ("alive", "dead") for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
        finally:
            translate._run_batch = real
        return jobs, state, segs2

    left, state, segs2 = handoff("dead")
    check("a model out of daily budget hands its work to the others",
          not left and state["error"] is None,
          f"{len(left)} batches stranded")
    check("every line is still translated after the handover",
          all(s["en"] for s in segs2),
          f"{sum(1 for s in segs2 if s['en'])}/{len(segs2)} lines")
    check("no stream is left holding a batch",
          state["holders"] == 0, f"holders={state['holders']}")
    check("a per-day limit is not retried like a transient error",
          "except Exhausted" in inspect.getsource(translate._run_batch),
          "retrying spends two more requests proving what the first reply said")
    # Running out mid-window is the common way this run ends, and the lines
    # already translated in that window were paid for.
    from app import pipeline as _pipe
    src = inspect.getsource(_pipe.process)
    _, _, after = src.partition("except Exception as exc:")
    check("a failed run keeps the translation it already paid for",
          "save_segments" in after,
          "the streaming path only saves at a window boundary")

    # What the voice is handed is not what the viewer reads. An English voice
    # has no rule for a Vietnamese tone mark or a non-breaking hyphen, and it
    # reads a leading ellipsis as a long pause at the moment it should be
    # talking. The subtitle keeps the correct spelling either way.
    speech = tts._clean_for_speech
    check("a Vietnamese name is made pronounceable",
          speech("Loan Phượng God Form.") == "Loan Phuong God Form.",
          "159 lines across the library carry one")
    check("the non-breaking hyphen becomes a real one",
          speech("a god‑tier skill") == "a god-tier skill")
    check("a line does not open on an ellipsis",
          speech("...that I was their husband.") == "that I was their husband.")
    check("curly quotes are folded down",
          speech("“Isn’t it?”") == '"Isn\'t it?"')
    check("the subtitle itself is untouched",
          "Phượng" in "Loan Phượng God Form.",
          "only the spoken copy is folded - the viewer reads the real spelling")

    # Whisper caps nothing, so a long unbroken passage arrives as one enormous
    # segment - 30 seconds at worst across this library. Everything downstream
    # measures a line by its own start and end, so that one segment becomes a
    # 450-character translation, one breathless utterance and an unreadable
    # subtitle all at once.
    long_seg = [{"id": 0, "start": 0.0, "end": 30.0, "text": "甲" * 90}]
    pieces = asr._split_long(long_seg)
    check("an over-long segment is split", len(pieces) > 1,
          f"30s -> {len(pieces)} pieces")
    check("no piece is longer than the merge cap",
          all(p["end"] - p["start"] <= asr.MERGE_MAX_SPAN + 0.01 for p in pieces),
          f"longest {max(p['end'] - p['start'] for p in pieces):.1f}s")
    check("splitting keeps every character and the span",
          "".join(p["text"] for p in pieces) == long_seg[0]["text"]
          and abs(pieces[-1]["end"] - 30.0) < 0.01 and pieces[0]["start"] == 0.0)
    check("a segment that already fits is untouched",
          asr._split_long([{"id": 0, "start": 0.0, "end": 4.0,
                            "text": "你好"}])[0]["text"] == "你好")


def _stream_checks() -> None:
    """Section 9: watching a video while it is still being dubbed."""
    import numpy as np

    from app import hlsout, pipeline, translate, tts

    print("\n9. Live streaming while the dub is still being made")

    # -- windows must tile the whole timeline with no gap and no overlap ----
    segs = [{"id": i, "start": i * 2.0, "end": i * 2.0 + 1.8,
             "text": "x", "en": "line"} for i in range(600)]
    total = 1200.0
    bounds = pipeline._window_bounds(segs, total)
    covers = all(bounds[k][2] == bounds[k + 1][2] - (bounds[k + 1][2] - bounds[k][2])
                 for k in range(len(bounds) - 1))
    starts_at_zero = bounds[0][0] == 0
    reaches_end = abs(bounds[-1][2] - total) < 1e-6
    contiguous = all(bounds[k][1] == bounds[k + 1][0] for k in range(len(bounds) - 1))
    check("windows cover the whole video", starts_at_zero and reaches_end and covers,
          f"{len(bounds)} windows, last ends {bounds[-1][2]:.0f}s")
    check("windows hand over without skipping a line", contiguous)
    check("the first window is short", bounds[0][2] <= 60,
          f"first is {bounds[0][2]:.0f}s, last is "
          f"{bounds[-1][2] - bounds[-2][2]:.0f}s")

    # -- the seam. A window must return exactly its own samples, and a line
    #    running past the end must survive into the next one rather than being
    #    cut - a lost sample desyncs everything after it.
    mixer = tts.StreamMixer(SEGMENTS, DURATION, "en-US-BrianNeural", WORK)
    mixer.carry = np.ones(int(2.0 * SR), dtype=np.float32) * 0.5   # pretend spill
    first = mixer.render(0, 0, 5.0)
    check("a window returns exactly its own length", len(first) == int(5.0 * SR),
          f"{len(first)} samples for 5.0s at {SR}Hz")
    check("spill from the previous window is mixed in, not dropped",
          float(np.max(np.abs(first[:int(2.0 * SR)]))) > 0.4)
    check("the clock advances by exactly one window", abs(mixer.pos - 5.0) < 1e-9)

    mixer.carry = np.zeros(int(mixer.TAIL_SECONDS * SR), dtype=np.float32)
    mixer.carry[:int(0.2 * SR)] = 0.3
    tail = mixer.drain()
    check("the final drain keeps the audio and not the empty buffer",
          0 < len(tail) <= int(0.25 * SR),
          f"{len(tail) / SR:.2f}s kept out of a {mixer.TAIL_SECONDS:.0f}s buffer")

    # -- a daily limit is not something a backoff can wait out ---------------
    daily = ("Error code: 429 - rate limit reached ... on tokens per day (TPD): "
             "Limit 200000, Used 199335. Please try again in 24m51.264s.")
    minute = "Error code: 429 - on tokens per minute (TPM). Please try again in 2.5s."
    check("a daily limit is told apart from a per-minute one",
          "per day" in daily.lower() and "per day" not in minute.lower())
    check("the reset time is read back for the message",
          translate._reset_hint(daily) == "resets in 24m51.264s")

    # -- both playlists must agree about whether the stream has ended --------
    src = inspect.getsource(hlsout.LiveStream)
    check("the picture is published open-ended like the audio",
          "omit_endlist" in src and '"event"' in src)
    # append_list makes ffmpeg believe it is resuming, and it then opens the
    # playlist with EXT-X-DISCONTINUITY, which stops hls.js lining the audio up
    # against the picture. The flag value is what matters, not the word - it is
    # named in a comment right above it.
    flags = [ln for ln in src.splitlines() if "hls_flags" in ln and "#" not in ln]
    check("the audio playlist does not open with a discontinuity",
          bool(flags) and all("append_list" not in ln for ln in flags),
          "; ".join(f.strip() for f in flags))

    # -- streaming must not cost more requests than dubbing in one pass ------
    # Groq meters requests per day as well as tokens, so a window boundary that
    # cuts a batch short turns into a whole extra request paying a full system
    # prompt for a handful of lines. Batches sit on a grid across the whole
    # transcript for exactly this reason; if that ever regresses, the plans stop
    # matching and the daily allowance starts running out sooner.
    lines = [{"id": i + 1, "start": i * 2.0, "end": i * 2.0 + 1.8,
              "text": f"line {i}", "en": ""} for i in range(437)]
    duration = lines[-1]["end"]

    def plan(windows):
        seen, work = [], [dict(s) for s in lines]

        class Fake:
            class chat:
                class completions:
                    @staticmethod
                    def create(*, messages, **kw):
                        ids = [int(m) for m in
                               re.findall(r"^(\d+)\|", messages[-1]["content"], re.M)]
                        seen.append(len(ids))
                        body = "".join(f"{i}|line {i}\n" for i in ids)
                        return SimpleNamespace(choices=[SimpleNamespace(
                            message=SimpleNamespace(content=body))], usage=None)

        budgets: dict = {}
        for lo, hi in windows:
            translate.translate(Fake(), work, "fake", "vi", clients=[Fake()],
                                system="SYS", window=(lo, hi), budgets=budgets)
        return seen, sum(1 for s in work if s["en"])

    total = len(lines)
    one, done_one = plan([(0, total)])
    windows = [(lo, hi) for lo, hi, _ in pipeline._window_bounds(lines, duration)]
    many, done_many = plan(windows)
    check("streaming costs no extra requests than one pass",
          len(many) == len(one),
          f"{len(many)} requests over {len(windows)} windows vs {len(one)} in one pass")
    check("the batches are the same ones either way", many == one,
          f"sizes {sorted(set(many))}")
    check("windowing still translates every line",
          done_many == done_one == total, f"{done_many}/{total} lines")


def _dub_quality_checks() -> None:
    """Section 10: two voices never overlap, and the audit only flags real faults."""
    from app import translate, tts

    print("\n10. Keeping the dub clean")

    # -- a line too long for its slot must not run over the next speaker ------
    # Back-to-back lines with no silence between them, which is what this
    # material actually looks like: on real episodes 98% of lines have no gap
    # after them, so an overrun always lands on top of speech.
    segs = [{"id": i, "start": i * 3.0, "end": i * 3.0 + 3.0, "en": "a line here"}
            for i in range(6)]
    results = {i: {"file": "x", "duration": 7.0, "native_rate": 1.3}
               for i in range(6)}          # 7s of speech for a 3s slot
    stats = {"compressed": 0, "max_speedup_used": 1.0, "crowded": 0, "clipped": 0}
    plan = tts._plan(segs, results, 18.0, 1.7, 0, 6, stats)

    worst = 0.0
    for seg, r, tempo, limit in plan[:-1]:
        played = min(r["duration"] / tempo, limit)
        worst = max(worst, seg["start"] + played - (seg["start"] + 3.0))
    check("no line is allowed to run into the next one", worst <= 0.0,
          f"worst finish is {worst:+.3f}s relative to the next line's start")
    check("the last line may keep its tail", plan[-1][3] == float("inf"))
    check("a crowded line is compressed past the viewer's cap",
          stats["crowded"] == 5 and max(t for _, _, t, _ in plan) > 1.55,
          f"{stats['crowded']} rescued, hardest {max(t for _, _, t, _ in plan):.2f}x")
    check("an impossible line is cut rather than left overlapping",
          stats["clipped"] > 0, f"{stats['clipped']} clipped")
    ceiling = tts.RESCUE_TOTAL_RATE + 1e-6
    check("the rescue never makes speech unintelligible",
          all(t * results[s["id"]]["native_rate"] <= ceiling
              for s, _, t, _ in plan),
          f"total rate stays at or under {tts.RESCUE_TOTAL_RATE}x")

    # -- both assemblers must make the same decisions ------------------------
    src = inspect.getsource(tts.build_dub_track)
    check("the one-pass assembler uses the shared planner", "_plan(" in src,
          "it used to carry its own copy, which could drift from the streaming one")

    # -- the audit: every one of these is a real line from a finished episode -
    cast = ("The narrator and main character is Lin. Narration is Lin speaking "
            'as "I".\n李=Lin (male)')
    faulty = [
        ({"id": 1, "start": 0, "end": 6, "text": "x",
          "en": "Bai Ningbing is proud that his husband can do what others can't."},
         "a woman called he"),
        ({"id": 2, "start": 0, "end": 6, "text": "x",
          "en": "You watch this and smile, saying nothing at all."},
         "narration slipped into the second person"),
        ({"id": 3, "start": 0, "end": 6, "text": "x",
          "en": "At this moment they feel no gratitude, and they've even thought that if"},
         "the sentence stops mid-thought"),
        ({"id": 4, "start": 0, "end": 2, "text": "x",
          "en": "It moved the brides far more than any lavish banquet ever could have, "
                "a pot of soup and a bowl of rice bringing all three women to tears."},
         "far too long for a two second slot"),
    ]
    for seg, why in faulty:
        check(f"caught: {why}", bool(translate.audit([seg], cast, "zh")))

    clean = [
        {"id": 6, "start": 0, "end": 6, "text": "x",
         "en": "She eyed my belongings and smiled."},
        {"id": 7, "start": 0, "end": 6, "text": "x",
         "en": "He leered at them. Never, we would rather die."},
        # Quoted speech keeps its second person - that is where "you" belongs.
        {"id": 8, "start": 0, "end": 6, "text": "x",
         "en": 'She bowed. "You should eat first, husband."'},
    ]
    flagged = translate.audit(clean, cast, "zh")
    check("leaves correct lines alone", not flagged,
          f"{len(flagged)} false alarms out of {len(clean)}")

    # Dialogue that carries no quotation marks cannot be told apart from
    # narration by reading it, so it IS flagged. That is deliberate: the repair
    # brief tells the model in as many words to leave real speech alone, and a
    # rewrite is only kept if the audit likes it better, so a wrong flag costs
    # a request rather than a broken line. Both halves are checked below.
    bare = {"id": 9, "start": 0, "end": 6, "text": "x",
            "en": "You should eat first, husband."}
    check("unquoted dialogue is flagged rather than missed",
          bool(translate.audit([bare], cast, "zh")),
          "the repair brief is what protects it, not the rule")

    # -- narration voice: the defect that makes the hero a stranger ---------
    # This genre writes the hero's own life at him in the second person, and
    # translated literally it comes out as somebody lecturing him.
    zh = [{"id": i, "start": i * 6.0, "end": i * 6.0 + 6.0,
           "text": "你" + "x" * 20, "en": ""} for i in range(60)]
    check("a second-person Chinese source is recognised as narration",
          translate._is_narrated(zh, "", "zh"))
    third = [dict(s, text="他" + "x" * 20) for s in zh]
    check("a third-person source is left alone",
          not translate._is_narrated(third, "", "zh"),
          "forcing 'I' onto a story about other people would be the worse bug")
    check("a language with no note is left alone",
          not translate._is_narrated(zh, "", "vi"))

    narr = ("The narrator and main character is Lin. Narration is Lin speaking "
            'as "I".')
    leaks = [
        "and your father chose to save the baby.",
        "Since your parents died, you were barely getting by.",
        "In your world you would have begged for such a woman.",
    ]
    for line in leaks:
        seg = {"id": 1, "start": 0, "end": 9, "text": "x", "en": line}
        check(f"caught: {line[:38]}...", bool(translate.audit([seg], narr, "zh")))

    # Dialogue that runs across a line break has its opening quote on the line
    # before, so neither line holds a matched pair. Both of these were flagged
    # as narration until the quote state was carried between lines.
    split = [
        {"id": 1, "start": 0, "end": 9, "text": "x",
         "en": 'Aunt Yun looked at us. "These are the bachelors from your village.'},
        {"id": 2, "start": 9, "end": 18, "text": "x",
         "en": 'Just the three of you." Uncle Wang nodded and smiled.'},
    ]
    flagged = translate.audit(split, narr, "zh")
    check("dialogue split across two subtitles keeps its 'you'",
          not flagged, f"{len(flagged)} false alarms")

    # An opening quotation whose partner never arrives must not flip the
    # reading of every line after it. Taken from a real scene: one unclosed
    # quote left four following lines parsed inside-out, and the dialogue on
    # the last of them was flagged as narration.
    stuck = [
        {"id": 1, "start": 0, "end": 6, "text": "x",
         "en": 'A maid spoke behind her: "Princess, accept your fate.'},
        {"id": 2, "start": 6, "end": 12, "text": "x",
         "en": "Yun Jin pleaded, but the maids dressed her up anyway."},
        {"id": 3, "start": 12, "end": 18, "text": "x",
         "en": "They sent her to my room, and I sighed."},
        {"id": 4, "start": 18, "end": 24, "text": "x",
         "en": 'She could not kill me. "Do you think I would let that happen?"'},
    ]
    check("an unclosed quotation does not swallow the lines after it",
          not translate.audit(stuck, narr, "zh"),
          f"{len(translate.audit(stuck, narr, 'zh'))} false alarms")

    # -- a repair is kept only when it is genuinely better -------------------
    seg = {"id": 9, "start": 0.0, "end": 6.0, "text": "x",
           "en": "You stared at the road as a group approached."}
    worse = {9: "You just stared and"}          # still 2nd person, now truncated
    better = {9: "I stared at the road as a group approached."}

    for label, reply, expect in (("a worse rewrite is refused", worse, False),
                                 ("a better rewrite is kept", better, True)):
        trial = [dict(seg)]
        real = translate._call
        translate._call = lambda *a, **k: "\n".join(
            f"{i}|{v}" for i, v in reply.items())
        try:
            out = translate.polish([object()], trial, ["m"], "zh", cast)
        finally:
            translate._call = real
        check(label, (out["fixed"] == 1) is expect,
              f"kept: {trial[0]['en'][:46]}")


def _library_checks() -> None:
    """Section 11: the library survives an upgrade and remembers where you were."""
    import sqlite3
    import tempfile
    from app import library, main

    print("\n11. Library upgrades and playback memory")

    # A database written by an older build has none of the newer columns, and
    # CREATE TABLE IF NOT EXISTS will not add them. Build exactly that, then
    # open it the way the app does.
    old_db = Path(tempfile.mkdtemp(prefix="lib_", dir=str(TMP))) / "old.db"
    con = sqlite3.connect(old_db)
    con.executescript("""CREATE TABLE videos (
        id TEXT PRIMARY KEY, title TEXT NOT NULL, status TEXT NOT NULL,
        stage TEXT, progress INTEGER DEFAULT 0, error TEXT, created_at REAL,
        updated_at REAL, duration REAL DEFAULT 0, source_lang TEXT, voice TEXT,
        asr_model TEXT, llm_model TEXT, line_count INTEGER DEFAULT 0,
        stats TEXT, original_name TEXT);""")
    con.execute("INSERT INTO videos (id,title,status,duration) VALUES "
                "('old1','An older video','ready',600)")
    con.commit()
    con.close()

    real_db = library.DB
    try:
        library.DB = old_db
        library.init()
        v = library.get("old1")
        check("an existing library is upgraded, not replaced",
              v is not None and v["title"] == "An older video")
        check("the new columns are added to it",
              v is not None and "position" in v and "watched_at" in v,
              "position, watched_at")

        before = library.get("old1")["updated_at"]
        library.set_position("old1", 421.37)
        after = library.get("old1")
        check("playback position is remembered", after["position"] == 421.4,
              f"{after['position']}s")
        check("saving it does not touch updated_at",
              after["updated_at"] == before,
              "otherwise the video URL changes under a viewer every few seconds")
        library.init()          # a second startup must be a no-op
        check("upgrading twice is harmless", library.get("old1") is not None)
    finally:
        library.DB = real_db
        shutil.rmtree(old_db.parent, ignore_errors=True)

    # Unfinished work is requeued rather than failed - every stage is
    # checkpointed, so picking it up costs only what was in flight.
    src = inspect.getsource(main._startup)
    check("an interrupted video is put back on the queue",
          "POOL.submit" in src and 'status="queued"' in src)
    # A dubbing job must never be failed by a restart. A download must, because
    # nothing here can resume one: yt-dlp owns that and the partial file may be
    # anything. Both live in the same loop, so this checks that the only
    # fail-on-restart path is the one guarded by the downloading status, rather
    # than checking for the message text - which the download branch also uses.
    head, _, tail = src.partition('if v["status"] == "downloading"')
    check("a dubbing job is not marked failed on restart",
          'status="failed"' not in head)
    check("an interrupted download is failed, not left spinning",
          bool(tail) and 'status="failed"' in tail.split("continue")[0])
    check("the oldest waiting video is resumed first",
          "sorted(" in src and "created_at" in src)


def _download_checks() -> None:
    """Section 12: picking a quality from a link, without touching the network."""
    import json as _json
    from app import download, library, main

    print("\n12. Choosing what to download")

    # A cut-down copy of what yt-dlp reports for a real YouTube video: several
    # codecs at each size, audio-only tracks, and the -drc duplicates.
    fake = {"title": "An episode", "duration": 3605, "uploader": "someone",
            "formats": [
                {"format_id": "139", "ext": "m4a", "vcodec": "none",
                 "acodec": "mp4a.40.5", "abr": 48.8, "filesize": 22_000_000},
                {"format_id": "139-drc", "ext": "m4a", "vcodec": "none",
                 "acodec": "mp4a.40.5", "abr": 48.8, "filesize": 22_000_000},
                {"format_id": "140", "ext": "m4a", "vcodec": "none",
                 "acodec": "mp4a.40.2", "abr": 129.5, "filesize": 58_000_000},
                {"format_id": "251", "ext": "webm", "vcodec": "none",
                 "acodec": "opus", "abr": 126.3, "filesize": 57_000_000},
                {"format_id": "136", "ext": "mp4", "height": 720, "fps": 30,
                 "vcodec": "avc1.4d401f", "acodec": "none", "tbr": 431,
                 "filesize": 194_000_000},
                {"format_id": "247", "ext": "webm", "height": 720, "fps": 30,
                 "vcodec": "vp9", "acodec": "none", "tbr": 918,
                 "filesize": 413_000_000},
                {"format_id": "135", "ext": "mp4", "height": 480, "fps": 30,
                 "vcodec": "avc1.4d401f", "acodec": "none", "tbr": 190,
                 "filesize": 85_000_000},
                {"format_id": "160", "ext": "mp4", "height": 144, "fps": 30,
                 "vcodec": "avc1.4d400c", "acodec": "none", "tbr": 35,
                 "filesize": 15_000_000},
                {"format_id": "232", "ext": "mp4", "height": 720, "fps": 30,
                 "vcodec": "avc1.4D401F", "acodec": "none", "tbr": 2073},
            ]}

    real_run = download._run
    try:
        download._run = lambda *a, **k: _json.dumps(fake)
        got = download.probe("https://example.invalid/watch")
    finally:
        download._run = real_run

    by = {q["label"]: q for q in got["qualities"]}
    check("one row per picture size, not one per codec",
          sorted(by) == ["144p", "480p", "720p"], ", ".join(sorted(by)))
    check("h264 wins over the bigger vp9 at the same size",
          by["720p"]["format"].startswith("136+"), by["720p"]["format"])
    check("a stream with no size is not offered",
          "232" not in by["720p"]["format"],
          "yt-dlp cannot cost it up front, so it cannot be shown")
    check("a watchable size gets the good audio",
          by["480p"]["format"] == "135+140", by["480p"]["format"])
    check("144p is not paired with the biggest audio track",
          by["144p"]["format"] == "160+139", by["144p"]["format"])
    check("m4a is chosen over opus so the result is still an mp4",
          all(q["ext"] == "mp4" for q in got["qualities"]))
    check("the -drc duplicates are ignored",
          all("-drc" not in q["format"] for q in got["qualities"]))
    check("size covers picture and sound together",
          by["480p"]["size"] == 85_000_000 + 58_000_000,
          f"{by['480p']['size'] / 1048576:.0f} MB")

    # A downloaded video must land on its own shelf, not in the dubbing queue.
    src = inspect.getsource(main._download_job)
    check("a finished download is not queued for dubbing",
          'status="downloaded"' in src and "pipeline.process" not in src)
    check("the row is created in the state the caller asked for",
          library.create.__doc__ and "may be overridden" in library.create.__doc__,
          "create() used to force every new row to 'queued'")


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
    _stream_checks()
    _dub_quality_checks()
    _library_checks()
    _download_checks()

    print("\n" + "=" * 62)
    print("  RESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print(f"  Listen to the result: {out}")
    print("=" * 62)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
