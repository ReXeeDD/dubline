"""Subtitle translation through a Groq-hosted LLM, batched and rate-aware.

Groq's on-demand tier allows 8000 tokens per minute per model, and `max_tokens`
is reserved against that budget at request time. So the work is split into small
batches, paced against the live rate-limit headers, and any batch that is still
refused is halved and retried.
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BATCH = 30          # lines per request
CONTEXT = 3         # neighbouring lines sent for continuity (not re-translated)
# Source lines at least this long are real speech, never "um" or "ah", so a
# blank translation for one is a mistake worth a second request.
FILLER_CHARS = 8
MAX_TPM_WAIT = 75.0  # longest pause while waiting for the budget to reset

# Requests per model, in flight at once. A reply takes a second or more, and the
# token bucket refills the whole time it is being waited on, so sending one at a
# time leaves throughput unused. Measured on 30-line batches: 1 stream 291
# tokens/sec, 3 streams 317, 6 streams 386 - but retries went 6 -> 18 -> 17, and
# past 3 most of the extra speed is spent re-sending refused requests. Three is
# where the gain is real and the bucket is not being fought.
STREAMS_PER_MODEL = 3

# Subtitle lines are short - often two or three characters - so wrapping each one
# in a JSON object spent more tokens on field names and punctuation than on the
# actual text. A pipe-delimited line carries the same information for less than
# half the tokens, which matters directly because throughput is capped per minute.
SYSTEM = """You are a professional subtitle translator and dubbing script writer.
You translate {src_name} into natural, fluent English.

INPUT: one subtitle per line, formatted  id|max_chars|text
OUTPUT: one translation per line, formatted  id|translation

Rules:
1. Translate EVERY id you are given, exactly once, in the same order.
2. Output natural spoken English - the way a person actually talks, not a literal
   word-for-word gloss. Idioms become equivalent English idioms.
3. EVERY LINE MUST BE GRAMMATICAL ENGLISH. This matters more than anything
   below it. A line is read aloud by a voice actor, so it has to be something a
   person could actually say: a real sentence, with its subject, its articles
   and its tense in place.
   - Never translate an idiom or a fixed phrase word by word. Find what an
     English speaker says in that situation and write that instead.
     "As his words fell" is not English - write "The moment he finished
     speaking". "Our bond is lacking" is not English - write "We barely know
     each other". "Isn't this the standard plot" - write "Straight out of a
     storybook".
   - Do not stack short fragments where one clause is meant. Two lines in a row
     reading "She clenched her fists." "She felt guilty." should carry the
     connection the original had.
4. THIS IS DUBBING, NOT SUBTITLING. Each line is spoken aloud over the original
   actor, so aim to fit max_chars characters. Get there by choosing shorter,
   plainer words and by cutting what the picture already shows - never by
   dropping the grammar. "I smiled and shook my head." (27) becomes "I shook my
   head." (16), which is still a whole sentence. If a line cannot be said
   correctly in the space, go slightly over rather than write something broken.
5. Keep names, numbers, units and technical terms accurate and consistent.
6. POINT OF VIEW. These stories are told by their main character, looking out
   at their own world, and the narration is that character speaking about
   himself. Getting this wrong is the single most damaging mistake you can
   make, because it makes the hero sound like he is describing someone else.
   - Narration - a line describing what is happening, rather than words spoken
     aloud to another character - is the main character's own voice. It is
     "I", "me", "my". Never render it as "you".
   - First and second person are unreliable in this transcript. It has been
     through speech recognition, and often through an earlier machine
     translation before that, and both drop first-person pronouns into second
     person. A narration line that arrives saying "you" almost always means
     "I". Trust the sense of the passage, not the pronoun you were given.
   - Dialogue keeps its own point of view. When one character speaks aloud TO
     another, "you" is correct and must stay "you".
   - Some scenes have no main character in them, and narrate other people
     instead. There, use those characters' names, or "he" and "she" - never
     "I" or "my". Only the main character narrates in first person.
7. PRONOUNS. The source came from a speech recogniser and its pronouns are
   guesses, frequently wrong. Do NOT copy them. Work out who is meant from the
   surrounding lines and the cast list, then use the correct English pronoun.
   - A noun that states a gender is reliable and outranks any pronoun near it:
     mother, wife, madam, daughter, sister, lady, Mrs are female; father,
     husband, son, brother, sir, Mr are male. Never write "he" about someone
     the line calls a mother.
   - Once you know a character's gender, use it consistently everywhere.
   - If a referent is genuinely unclear, use the person's name rather than
     guessing a pronoun.
8. Output NOTHING but the id|translation lines - no commentary, no headers, no
   original text, no romanisation, no markdown, no blank lines.
9. Leave a line blank ONLY for pure filler ("um", "ah") or unintelligible
   noise. If the source is a real sentence you must return a real translation,
   even a rough one - a blank line becomes silence over an actor who is
   visibly speaking. If the recogniser garbled it, translate what it most
   likely meant from the lines around it.
10. Write out numerals as a speaker would say them where it reads more
   naturally.

Example output:
41|I never lost.
42|Don't contact me again."""

LANG_NAMES = {
    "zh": "Mandarin Chinese", "ja": "Japanese", "ko": "Korean", "es": "Spanish",
    "fr": "French", "de": "German", "ru": "Russian", "ar": "Arabic",
    "hi": "Hindi", "pt": "Portuguese", "it": "Italian", "vi": "Vietnamese",
    "th": "Thai", "id": "Indonesian", "en": "English",
    "auto": "the source language",
}

# What each language's pronouns actually mean, for languages whose system does
# not map onto English one for one. Web-serial narration leans on archaic and
# status-marked pronouns that a general translator renders inconsistently, and
# getting them wrong is what makes the narrator drift between "I" and "you".
PRONOUN_NOTES = {
    "vi": (
        "Vietnamese pronouns in this genre: 'ta' = I/me, the narrator speaking "
        "of himself; 'nguoi' = you, said to someone's face; 'han', 'y', 'ga' = "
        "he; 'nang' = she, a young woman; 'chang' = he, a young man; 'lao' = an "
        "old man; 'phu than' = father, 'mau than' = mother, 'ca ca' = elder "
        "brother, 'muoi' = younger sister.\n"
        "CRITICAL: this transcript writes 'nguoi' (you) in narration lines "
        "where the speaker is plainly describing HIS OWN actions. That is an "
        "error upstream of you, and it is the single most common defect in this "
        "material. Do not reproduce it. Whenever a narration line reads "
        "'nguoi' but the surrounding lines show the narrator acting, "
        "translate it as I/me/my.\n"
        "Worked example - three consecutive narration lines:\n"
        "  'Nghi den day, ta tang toc tien ve phia Huyen Thien Tong'\n"
        "     -> 'Thinking this, I sped toward Xuantian Sect'\n"
        "  'Khi nguoi vua dat chan den Huyen Thien Tong'\n"
        "     -> 'The moment I set foot in Xuantian Sect'\n"
        "  'Theo tieng noi cua nguoi vua dut'\n"
        "     -> 'The moment I finished speaking'\n"
        "The second and third lines say 'nguoi', but the passage is the "
        "narrator continuing to describe his own arrival and his own speech, "
        "so both are 'I'. Writing 'you' there would be wrong.\n"
        "'nguoi' stays 'you' ONLY inside dialogue - words one character says "
        "out loud to another."),
    "zh": (
        "In spoken Mandarin the words for he, she and it are identical, so the "
        "character written in the transcript is only the recogniser's guess. "
        "Resolve each one from context and the cast list instead."),
}


class TooLarge(Exception):
    """The request exceeded the per-minute token allowance even when idle."""


class Exhausted(Exception):
    """This model has spent its allowance for the day, not just the minute."""


def _reset_hint(msg: str) -> str:
    """The 'try again in ...' Groq puts in a rate-limit message, if it is there."""
    m = re.search(r"try again in ([\dhms.]+)", msg)
    return f"resets in {m.group(1).rstrip('.')}" if m else "resets later today"


# ------------------------------------------------------------ token budget ---
def _parse_reset(value: str) -> float:
    """Groq formats these as '30.097s', '1m26.4s' or '2m'."""
    if not value:
        return 0.0
    total, m = 0.0, re.match(r"(?:(\d+)m)?([\d.]+)?s?", value.strip())
    if m:
        if m.group(1):
            total += int(m.group(1)) * 60
        if m.group(2):
            total += float(m.group(2))
    return total


def _estimate(text: str) -> int:
    """Rough token count. CJK runs about one token per character, Latin ~4 chars."""
    cjk = len(re.findall(r"[㐀-鿿぀-ヿ가-힯]", text))
    return cjk + max(0, len(text) - cjk) // 3 + 8


class TokenBudget:
    """Paces requests using the rate-limit headers the API sends back.

    One of these per model, shared by every stream running against it, because
    the limit itself is per model. It is a token bucket that refills
    continuously - measured at limit/60 per second, with `reset-tokens` reported
    in milliseconds - not an allowance that drains and resets on the minute.

    Shared across threads, so the bookkeeping is locked. A reservation is taken
    before a request goes out and reconciled from the response headers: without
    that, several streams all read "plenty left" at once and fire together,
    which is how a concurrency win turns into a storm of 429s.
    """

    def __init__(self) -> None:
        # Set to a human-readable reason once this model's daily allowance is
        # gone, which no amount of waiting inside one job will recover.
        self.exhausted: str = ""
        self.remaining: int | None = None
        self.reset_at = 0.0
        self.limit: int | None = None
        self._reserved = 0
        self._lock = threading.Lock()

    def _free(self) -> int | None:
        if self.remaining is None:
            return None
        return self.remaining - self._reserved

    def would_wait(self, cost: int) -> float:
        """Seconds this request would have to wait, 0 if it can go now."""
        with self._lock:
            free = self._free()
            if free is None or free >= cost:
                return 0.0
            return max(self.reset_at - time.time(), 0.0)

    def reserve(self, cost: int) -> None:
        """Claim budget for a request that is about to go out."""
        with self._lock:
            self._reserved += cost

    def release(self, cost: int) -> None:
        with self._lock:
            self._reserved = max(0, self._reserved - cost)

    def wait_for(self, cost: int) -> None:
        deadline = time.time() + MAX_TPM_WAIT
        while True:
            with self._lock:
                free = self._free()
                if free is None or free >= cost:
                    return
                delay = max(self.reset_at - time.time(), 0.0) + 0.6
            if time.time() >= deadline:
                with self._lock:
                    self.remaining = None      # stop blocking; let the call try
                return
            time.sleep(min(delay, 2.0))        # re-check often, the bucket refills

    def update(self, headers) -> None:
        try:
            get = headers.get
        except AttributeError:
            return
        rem, lim = get("x-ratelimit-remaining-tokens"), get("x-ratelimit-limit-tokens")
        with self._lock:
            if rem is not None and str(rem).isdigit():
                self.remaining = int(rem)
            if lim is not None and str(lim).isdigit():
                self.limit = int(lim)
            reset = _parse_reset(get("x-ratelimit-reset-tokens") or "")
            if reset:
                self.reset_at = time.time() + reset

    def headroom(self) -> int:
        return self.limit or 8000


# ------------------------------------------------------------------- calls ---
# Reasoning behaviour differs per model and is discovered on first use.
#   None  -> not probed yet
#   ""    -> reasoning can be switched off entirely (qwen3): no extra allowance
#   "low" -> reasoning is mandatory (gpt-oss): the chain of thought is billed as
#            output, so max_tokens must cover it or the reply comes back empty
_REASONING: dict[str, str] = {}
REASONING_ALLOWANCE = 2.0    # multiplier on max_tokens when reasoning is forced


# --------------------------------------------------------------- auditing ---
# Faults worth a second request, found by reading the finished English rather
# than by trusting the model to have followed the brief. Each one was taken from
# a real dubbed episode:
#
#   "he's proud that his husband can do what others can't"   (a wife, called he)
#   "You watch this and smile, saying..."                    (narration in 2nd person)
#   "...and they've even thought that if they truly..."      (line stops mid-thought)
#
# The rules are deliberately narrow. A false flag costs a re-translation of a
# line that was already right, and the model usually returns something worse the
# second time, so precision matters more here than recall.

# Relationships that give away a mismatch on their own, whatever the cast list
# says: in this material a husband is a man and a wife is a woman.
_CLASH = re.compile(
    r"\bhis (?:husband|wives)\b|\bher wife\b|\bshe is his (?:husband)\b",
    re.I)
_SHE = re.compile(r"\b(?:she|her|hers|herself)\b", re.I)
_HE = re.compile(r"\b(?:he|him|his|himself)\b", re.I)
# A line that opens in the second person is narration that slipped out of "I".
# Restricted to the opening word: a "you" further in is usually a character
# genuinely addressing someone, and re-asking those makes the dub worse.
#
# Someone really being spoken to is told to do something or asked about
# themselves - "You should eat first, husband", "You're amazing!", "Do you have
# a sword?". Narration that has slipped reads as plain report: "You were a
# modern youth", "Your mother died in childbirth", "You stared at the road".
# Excluding the modal and the question keeps genuine dialogue out of the net.
_SECOND_PERSON = re.compile(
    r"^\s*you(?:r\b|\b(?!\s*(?:'|’)?\s*(?:re|ll|ve|d|should|must|can|"
    r"could|will|would|may|might|need|want|know|see|have to|had better|"
    r"do|don't)\b))", re.I)
# The same slip inside the line rather than at the start of it: "Good." You
# nodded again and walked to the forge.' A "you" opening a SENTENCE and followed
# by a narrative past-tense verb is a report of what the narrator did, not words
# said to anybody. Measured over five finished episodes this matches 104 lines,
# of which roughly six in seven are genuine slips; the rest are a character
# saying something like "You saved me last time", so the repair brief is told
# in as many words to leave real dialogue alone.
_SECOND_PERSON_MID = re.compile(
    r"(?:^|(?<=[.!?])\s+)You\s+(?:\w+ed|were|had|went|took|felt|saw|knew|"
    r"got|made|came|gave|found|thought|stood|sat|ran|held|began|kept|"
    r"seemed|watched|walked|turned|grabbed|stared)")
# Quoted speech, so it can be taken out of the line before looking for a POV
# slip. Lines routinely carry both - 'Good." You nodded again and walked to the
# forge.' - and skipping any line with a quote in it hid every one of those,
# while testing the whole line would flag the dialogue for being in the second
# person, which is exactly where it belongs.
_QUOTED = re.compile(r'["“”‘’][^"“”]*'
                     r'["“”]')
_UNFINISHED = re.compile(
    r"(?:\.\.\.|…)\s*$|\b(?:and|but|so|then|the|an?|to|of|with|that|because|"
    r"as|for|if|when|while|is|was|were)\s*[.,]?\s*$", re.I)
_SOURCE_SCRIPT = re.compile(r"[぀-ヿ一-鿿가-힯]")
_OVER_BUDGET = 1.55      # times max_chars before a line is worth shortening


def _cast_genders(glossary: str) -> dict[str, str]:
    """English name -> 'f'/'m', read back out of the cast list."""
    out: dict[str, str] = {}
    for entry in re.split(r",\s*|\n", glossary):
        m = re.match(r"\s*[^=]+=(.+?)\s*\((female|male)\)\s*$", entry)
        if m and len(m.group(1)) > 2:
            out[m.group(1)] = m.group(2)[0]
    return out


def audit(segments: list[dict], glossary: str = "", src: str = "",
          window: tuple[int, int] | None = None) -> dict[int, list[str]]:
    """Which finished lines look wrong, and why. No API calls."""
    narrated = "narrator and main character" in glossary
    lo, hi = window if window else (0, len(segments))
    faults: dict[int, list[str]] = {}

    for seg in segments[lo:hi]:
        en = (seg.get("en") or "").strip()
        if not en:
            continue
        bad: list[str] = []

        if _CLASH.search(en):
            bad.append("a man is called she, or a woman he - check who is "
                       "speaking and who is being spoken about")
        # A question is someone talking to someone, never narration. The two
        # rules are scoped differently on purpose: opening in the second person
        # only means anything on a line that is not dialogue at all, while a
        # narrative past-tense verb is evidence in its own right and can be
        # trusted in the narration left over once the quotes are removed.
        narration = _QUOTED.sub(" ", en)
        slipped = (_SECOND_PERSON_MID.search(narration)
                   or ('"' not in en and '“' not in en
                       and _SECOND_PERSON.search(en)))
        if narrated and slipped and not en.rstrip().endswith("?"):
            bad.append("narration must be first person - this is the main "
                       "character talking about himself, so 'I', not 'you'. "
                       "But if this line is really one character SPEAKING to "
                       "another, 'you' is correct: leave it exactly as it is")
        if _UNFINISHED.search(en):
            bad.append("the sentence stops in the middle - finish it")
        if _SOURCE_SCRIPT.search(en):
            bad.append("untranslated source text is left in the line")
        limit = _budget_chars(seg)
        if len(en) > limit * _OVER_BUDGET:
            bad.append(f"far too long to speak in the time - {len(en)} "
                       f"characters where {limit} fit, say it shorter")

        if bad:
            faults[seg["id"]] = bad
    return faults


POLISH_BATCH = 20        # flagged lines per repair request

POLISH_SYSTEM = """You are correcting single lines of an English dubbing script.
Another translator produced them from {src_name} and each line below has a
specific fault.

INPUT: one line per faulty subtitle, formatted  id|max_chars|fault|source|current
OUTPUT: one line per id, formatted  id|corrected English

Rules:
1. Return EVERY id you are given, exactly once, in the same order. Output only
   the id and the corrected line - never repeat max_chars, the fault or the
   source text back.
2. Fix the stated fault. Keep everything else about the line - its meaning, its
   tone, and the names it uses.
3. The result must be a complete, grammatical English sentence that a voice
   actor can read aloud in one breath. Never return a fragment, and never end
   mid-thought or with an ellipsis standing in for words you left out.
4. It is spoken over the original actor, so it must fit max_chars characters.
   Say less rather than saying it badly: cut detail the picture already shows,
   choose shorter words, and drop what the previous line already established.
5. If you genuinely cannot improve the line, return it unchanged.
6. Output NOTHING but the id|line rows."""


def _parse_repair(content: str) -> dict[int, str]:
    """Read the repaired lines, whatever shape the model sent them back in.

    Asked for `id|line`, a model given `id|max_chars|fault|source|line` quite
    often echoes the whole row instead - and then the plain parser keeps the
    max_chars field as the translation, so a repair pass silently rewrites every
    line into the word "105". The corrected text is last either way, so that is
    what gets taken.
    """
    out: dict[int, str] = {}
    for line in _strip_think(content).splitlines():
        m = re.match(r"^\s*[-*• ]*(\d{1,7})\s*\|(.*)$", line)
        if not m:
            continue
        text = m.group(2).split("|")[-1].strip().strip('"')
        if text:
            out[int(m.group(1))] = text
    return out


def _polish_payload(segments: list[dict], faults: dict[int, list[str]],
                    ids: list[int]) -> str:
    by_id = {s["id"]: s for s in segments}
    rows = []
    for i in ids:
        s = by_id[i]
        why = "; ".join(faults[i])
        rows.append(f"{i}|{_budget_chars(s)}|{why}|{s.get('text', '')}|{s['en']}")
    return "\n".join(rows)


def polish(clients: list, segments: list[dict], models: list[str], src: str,
           glossary: str = "", progress=None, budgets: dict | None = None,
           window: tuple[int, int] | None = None) -> dict:
    """Re-ask for the lines that came back faulty, and keep only real gains.

    Translation happens 30 lines at a time with no sight of the finished script,
    so a fault the brief warned about still slips through: the narration drops
    into "you", a wife is called "he", a sentence stops halfway, a line lands at
    twice the length that can be spoken in its slot. Measured over seven
    finished episodes, 3% of Vietnamese lines and up to a third of the Chinese
    ones carried at least one of those.

    A replacement is accepted only when the audit likes it better than what was
    there. That matters: a model asked to repair a line it cannot improve will
    happily return something shorter and worse, and this stage must never make a
    good line bad.
    """
    faults = audit(segments, glossary, src, window)
    if not faults:
        return {"checked": 0, "fixed": 0}

    ids = sorted(faults)
    groups = [ids[i:i + POLISH_BATCH] for i in range(0, len(ids), POLISH_BATCH)]
    by_id = {s["id"]: s for s in segments}
    system = POLISH_SYSTEM.format(
        src_name=LANG_NAMES.get(src, "the source language"))
    if glossary:
        system += ("\n\nCAST LIST. These spellings and genders are settled - "
                   "use them:\n" + glossary)

    pool_models = [m for m in models if m]
    pairs = [(ci, m) for ci in range(len(clients)) for m in pool_models] or [(0, pool_models[0])]
    if budgets is None:
        budgets = {}
    for ci, m in pairs:
        budgets.setdefault((ci, m), TokenBudget())

    done = {"n": 0}
    lock = threading.Lock()

    def run(job):
        gi, group = job
        ci, model = pairs[gi % len(pairs)]
        payload = _polish_payload(segments, faults, group)
        limit = int(min(2600, max(400, sum(_budget_chars(by_id[i]) for i in group)
                                  / 3.2 + 6 * len(group) + 80) * 2.0))
        # A refused request here used to return quietly, so a rate limit at the
        # wrong moment meant the whole repair pass did nothing and said so only
        # by reporting zero fixes. Retry the way a translation batch does.
        reply: dict[int, str] = {}
        for attempt in range(3):
            try:
                reply = _parse_repair(_call(clients[ci], model, system, payload,
                                            limit, budgets[(ci, model)]))
                break
            except Exhausted:
                return
            except Exception:
                if attempt == 2:
                    return
                time.sleep(2.0 * (attempt + 1))
        with lock:
            for i, new in reply.items():
                seg = by_id.get(i)
                new = (new or "").strip()
                if seg is None or not new or new == seg.get("en"):
                    continue
                # Accept only a strict improvement, judged the same way the
                # line was flagged in the first place.
                before = len(faults.get(i, []))
                trial = dict(seg, en=new)
                after = len(audit([trial], glossary, src).get(i, []))
                if after < before:
                    seg["en"] = new
                    done["n"] += 1
            if progress:
                progress(f"Polishing {min(len(ids), (gi + 1) * POLISH_BATCH)}"
                         f" of {len(ids)} lines", 70)

    jobs = list(enumerate(groups))
    if len(jobs) == 1:
        run(jobs[0])
    else:
        with ThreadPoolExecutor(max_workers=min(len(pairs) * 2, 6)) as ex:
            list(ex.map(run, jobs))
    return {"checked": len(ids), "fixed": done["n"]}


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse(content: str) -> dict[int, str]:
    """Read `id|translation` lines, tolerating stray prose around them."""
    content = _strip_think(content)
    out: dict[int, str] = {}

    for line in content.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        m = re.match(r"^(\d{1,7})\s*\|\s*(.*)$", line)
        if not m:
            continue
        text = m.group(2).strip().strip('"')
        # a model that ignored the format and echoed id|max_chars|text
        again = re.match(r"^\d{1,5}\s*\|\s*(.*)$", text)
        if again and not re.search(r"[A-Za-z]", text.split("|")[0]):
            text = again.group(1).strip()
        out[int(m.group(1))] = text

    if not out and content.lstrip().startswith("{"):
        try:                      # a model that fell back to JSON anyway
            data = json.loads(content[content.index("{"):content.rindex("}") + 1])
            for item in data.get("lines", []):
                out[int(item["id"])] = str(item.get("en", "")).strip()
        except Exception:
            pass

    if not out:
        raise ValueError("model reply contained no id|translation lines")
    return out


def _is(exc: Exception, *codes: int) -> bool:
    text = str(exc)
    return any(f"Error code: {c}" in text or f"'{c}'" in text for c in codes)


def _call(client, model: str, system: str, user: str, max_tokens: int,
          budget: TokenBudget) -> str:
    """One chat completion, paced against the token budget."""
    mode = _REASONING.get(model)
    if mode == "low":
        # thinking tokens are billed as output, so leave room for them
        max_tokens = int(max_tokens * REASONING_ALLOWANCE)

    cost = _estimate(system) + _estimate(user) + max_tokens
    if cost > budget.headroom():
        raise TooLarge(f"a {cost} token request cannot fit in "
                       f"{budget.headroom()} tokens per minute")
    budget.wait_for(cost)
    budget.reserve(cost)

    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    # Left unchecked a reasoning model spends the whole allowance thinking and
    # returns an empty message. Switch it off where the model allows that.
    kwargs["reasoning_effort"] = mode if mode else "none"

    try:
        return _call_inner(client, kwargs, model, max_tokens, budget)
    finally:
        budget.release(cost)


def _call_inner(client, kwargs: dict, model: str, max_tokens: int,
                budget: TokenBudget) -> str:
    for attempt in range(4):
        try:
            raw = getattr(client.chat.completions, "with_raw_response", None)
            if raw is not None:
                resp = raw.create(**kwargs)
                budget.update(resp.headers)
                content = resp.parse().choices[0].message.content
            else:
                content = client.chat.completions.create(
                    **kwargs).choices[0].message.content

            if (content or "").strip():
                return content
            # Empty means thinking consumed the whole allowance before any
            # answer was produced. Give it more room and try once more.
            if attempt < 3:
                kwargs["max_tokens"] = min(8000, int(kwargs["max_tokens"] * 1.8))
                _REASONING.setdefault(model, "low")
                continue
            raise RuntimeError(f"{model} returned an empty reply")

        except Exception as e:
            msg = str(e)
            if "reasoning_effort" in msg and "reasoning_effort" in kwargs:
                # "must be one of low, medium, high" -> reasoning is mandatory
                if "low" in msg:
                    _REASONING[model] = "low"
                    kwargs["reasoning_effort"] = "low"
                    kwargs["max_tokens"] = int(max_tokens * REASONING_ALLOWANCE)
                else:
                    _REASONING[model] = ""
                    kwargs.pop("reasoning_effort")
                continue
            if _is(e, 413):
                raise TooLarge(msg) from e
            # A per-DAY limit is not something a backoff can outwait: the
            # window is hours, not seconds. Retrying it burns three quarters of
            # a minute each time and leaves the stage frozen with no
            # explanation, so this model is taken out of the pool instead and
            # the others carry the work.
            if _is(e, 429) and "per day" in msg.lower():
                budget.exhausted = _reset_hint(msg)
                raise Exhausted(f"{model}: {budget.exhausted}") from e
            if _is(e, 429, 503, 502) and attempt < 3:
                wait = _parse_reset(
                    re.search(r"try again in ([\dhms.]+)", msg).group(1)
                    if re.search(r"try again in ([\dhms.]+)", msg) else "") or 12.0
                time.sleep(min(wait + 0.5, MAX_TPM_WAIT))
                budget.remaining = None
                continue
            raise
    raise RuntimeError("translation request kept failing after several retries")


def _budget_chars(seg: dict) -> int:
    """Roughly how many English characters fit in this line's time slot.

    Natural English narration runs ~15 characters per second.
    """
    span = max(0.4, seg["end"] - seg["start"])
    return max(12, int(span * 15))


def _max_tokens(batch: list[dict]) -> int:
    """Size the reply allowance from the actual character budgets in the batch.

    `max_tokens` is reserved against the per-minute allowance whether or not it
    is used, so over-provisioning it directly costs throughput. The reply is
    `id|translation` per line: roughly one token per 3.2 English characters plus
    about five for the id and separator. Doubled for safety.
    """
    chars = sum(_budget_chars(s) for s in batch)
    need = chars / 3.2 + 5 * len(batch) + 60
    return int(max(300, min(2600, need * 2.0)))


def _payload(segments: list[dict], lo: int, hi: int) -> str:
    body = "\n".join(f"{s['id']}|{_budget_chars(s)}|{s['text']}"
                     for s in segments[lo:hi])
    before = [s["text"] for s in segments[max(0, lo - CONTEXT):lo]]
    after = [s["text"] for s in segments[hi:hi + CONTEXT]]
    head = f"# preceding context (do not translate): {' / '.join(before)}\n" if before else ""
    tail = f"\n# following context (do not translate): {' / '.join(after)}" if after else ""
    return head + body + tail


def _run_batch(client, model: str, system: str, segments: list[dict],
               lo: int, hi: int, budget: TokenBudget) -> dict[int, str]:
    batch = segments[lo:hi]
    user = _payload(segments, lo, hi)
    limit = _max_tokens(batch)

    result: dict[int, str] = {}
    for attempt in range(3):
        try:
            result.update(_parse(_call(client, model, system, user, limit, budget)))
        except TooLarge:
            raise
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
            continue

        missing = [s for s in batch if s["id"] not in result]
        if not missing or attempt == 2:
            break
        # re-ask for only the lines that came back missing
        user = "\n".join(f"{s['id']}|{_budget_chars(s)}|{s['text']}" for s in missing)
        limit = _max_tokens(missing)
    return result


def _sample(segments: list[dict], budget: int = 3200, lines: int = 110) -> str:
    """Text spread across the whole transcript, not just the opening.

    Characters who matter often do not appear until well in, and their gender
    is usually established once, in one line, somewhere in the middle.

    Stopping once a character budget had been spent meant the sample only ever
    covered the FRONT of a video. Measured on a real 617-line episode: it ran
    out at line 450, and two of the three wives are introduced after that - so
    they never reached the cast list, and the dub called both of them "he" for
    the rest of the hour. Every sampled line is trimmed to a share of the budget
    instead, so the spread always reaches the last line.
    """
    text = [s.get("text", "").strip() for s in segments]
    text = [t for t in text if t]
    if not text:
        return ""
    step = max(1, len(text) / min(lines, len(text)))
    picked = [text[min(len(text) - 1, int(i * step))]
              for i in range(min(lines, len(text)))]
    per = max(24, budget // len(picked))
    return "\n".join(t[:per] for t in picked)


def build_glossary(client, segments: list[dict], model: str, src: str) -> str:
    """Agree names AND genders before translating.

    Two jobs in one cheap call:

    * Rate limits are per model, so long videos are split across several models
      running at once. Without a shared glossary each one invents its own
      transliteration and the same character ends up spelled three ways.
    * More importantly, in spoken Chinese "he", "she" and "it" are the same
      word. Whisper only hears the sound and has to guess the character, so the
      transcript's 他/她 is unreliable and the dub ends up calling a woman "he".
      Naming each character's gender up front gives every batch something solid
      to resolve pronouns against.
    """
    lang = LANG_NAMES.get(src, "the source language")
    system = (
        f"You are preparing a cast list from {lang} subtitles for a dubbing team.\n"
        "FIRST LINE - the narrator. Video like this is usually narrated by its "
        "main character in the first person, mixed with ordinary third-person "
        "scene description. Decide whether ANY of the narration uses a "
        "first-person pronoun for a character who is part of the story, and if "
        "so which character that is. Output  NARRATOR|name  as the very first "
        "line, or NARRATOR|- only if the narration is entirely third person.\n"
        "Then list every recurring person and place. Include relationship titles and "
        "forms of address as well as given names - someone's mother, father, "
        "wife, a teacher's widow, a grandfather - because those are the entries "
        "that settle a character's gender.\n"
        "Output one per line as  original|English|gender\n"
        "gender is f for female, m for male, or - for anything that is not a "
        "person. Work the gender out from how the character is described and "
        "addressed - words like mother, wife, daughter, madam, father, husband, "
        "son, sir. Do not rely on the pronouns in the text: they are speech "
        "recognition guesses and are often wrong.\n"
        "Use the most natural English rendering of each name. "
        "Output nothing else. If there are none, output nothing.")
    # The opening is where a first-person narrator establishes himself, and a
    # sample spread over the rest of the video is where the supporting cast
    # turns up. Sampling alone misses the narrator: on a real episode it
    # answered "third person" because every line it drew came from the middle.
    opening = " / ".join(s["text"] for s in segments[:40])
    user = (f"# opening of the video\n{opening}\n\n"
            f"# spread over the rest\n{_sample(segments, 1800)}")
    try:
        raw = _strip_think(_call(client, model, system, user, 600, TokenBudget()))
    except Exception:
        return ""

    entries = []
    narrator = ""
    for line in raw.splitlines():
        line = line.strip().lstrip("-*• ")
        if "|" not in line or len(line) > 110:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        if parts[0].upper() == "NARRATOR":
            if parts[1] not in ("-", "?"):
                narrator = parts[1]
            continue
        sex = (parts[2][:1].lower() if len(parts) > 2 else "")
        tag = {"f": " (female)", "m": " (male)"}.get(sex, "")
        entries.append(f"{parts[0]}={parts[1]}{tag}")

    cast = ", ".join(entries[:40])
    if narrator:
        # Naming the narrator is what lets a batch tell "I" from "he": every
        # line of narration belongs to this character, and no other character
        # may be given his voice.
        cast = (f"The narrator and main character is {narrator}. Narration is "
                f"{narrator} speaking as \"I\". When another character is the "
                f"subject, name them or use he/she.\n" + cast)
    return cast


def build_system(src: str, glossary: str = "") -> str:
    """The full translator brief: rules, language notes and the cast list."""
    system = SYSTEM.format(src_name=LANG_NAMES.get(src, "the source language"))
    if PRONOUN_NOTES.get(src):
        system += "\n\nABOUT THIS LANGUAGE. " + PRONOUN_NOTES[src]
    if glossary:
        system += ("\n\nCAST LIST. Use these spellings exactly, and these "
                   "genders when choosing English pronouns - they outrank "
                   "the pronouns in the transcript:\n" + glossary)
    return system


def cast_list(client, segments: list[dict], model: str, src: str,
              cache_dir: Path | None = None) -> str:
    """The cast list for a transcript, read from cache when one was made."""
    cache = (cache_dir / "glossary.txt") if cache_dir else None
    if cache is not None and cache.is_file():
        try:
            return cache.read_text(encoding="utf-8")
        except OSError:
            pass
    glossary = build_glossary(client, segments, model, src)
    if cache is not None and glossary:
        try:
            cache.write_text(glossary, encoding="utf-8")
        except OSError:
            pass
    return glossary


def _worker(client, model: str, system: str, segments: list[dict],
            jobs: list, lock, state: dict, budget: TokenBudget) -> None:
    """Drain the shared job list using one model, paced by a shared budget.

    Several of these run per model. They share one TokenBudget because the rate
    limit is per model, not per connection - so the streams pace against each
    other rather than each believing it has the whole allowance.

    A model whose reasoning cannot be switched off spends a fixed overhead on
    every request, so it takes bigger bites: the system prompt and the thinking
    are then amortised over more lines instead of being paid per small batch.
    """
    handed_back = 0
    while True:
        with lock:
            if not jobs or state.get("error"):
                return
            lo, hi = jobs.pop(0)
            # merge in following batches when this model wants larger bites
            if _REASONING.get(model) == "low":
                while jobs and jobs[0][0] == hi and (hi - lo) < BATCH * 2:
                    hi = jobs.pop(0)[1]

            # If this model is out of budget but others still have work to hand
            # it, pass the job on rather than blocking on it for a minute.
            if jobs and handed_back < 3:
                cost = _estimate(system) + 40 * (hi - lo) + _max_tokens(segments[lo:hi])
                if budget.would_wait(cost) > 25.0:
                    jobs.append((lo, hi))
                    handed_back += 1
                    wait_a_moment = True
                else:
                    wait_a_moment = False
            else:
                wait_a_moment = False

        if wait_a_moment:
            time.sleep(2.0)
            continue
        handed_back = 0

        try:
            result = _run_batch(client, model, system, segments, lo, hi, budget)
        except Exhausted:
            # This model has no allowance left today. Give the batch back so a
            # model that still has budget picks it up, and stop this stream.
            with lock:
                jobs.insert(0, (lo, hi))
            return
        except TooLarge:
            if hi - lo > 1:
                mid = lo + (hi - lo) // 2
                with lock:
                    jobs.insert(0, (mid, hi))
                    jobs.insert(0, (lo, mid))
                continue
            with lock:
                state["error"] = RuntimeError(
                    f"A single subtitle line exceeds the {budget.headroom()} "
                    "tokens-per-minute limit.")
            return
        except Exception as e:
            with lock:
                state["error"] = e
            return

        # _run_batch re-asks for ids that never came back, but a model is also
        # allowed to answer "id|" with nothing after it to mark a line as
        # filler - and it sometimes does that to real dialogue. That counts as
        # answered, so it slips through as a silent gap in the finished dub
        # over a character who is visibly speaking. Anything long enough to be
        # real speech gets asked a second time.
        missing = [s for s in segments[lo:hi]
                   if not result.get(s["id"], "").strip()
                   and len(s.get("text", "").strip()) >= FILLER_CHARS]
        if missing and len(missing) < (hi - lo):
            try:
                again = _run_batch(client, model, system, missing,
                                   0, len(missing), budget)
                result.update({k: v for k, v in again.items() if v.strip()})
            except Exception:
                pass    # the gap is already handled below; do not fail the run

        with lock:
            for s in segments[lo:hi]:
                s["en"] = result.get(s["id"], "").strip()
            state["done"] += hi - lo
            state["report"]()
            # Persist as we go. Translation is the longest paid stage, so a
            # crash near the end must not throw away everything before it.
            if state.get("checkpoint"):
                try:
                    state["checkpoint"](segments)
                except Exception:
                    pass


def translate(client, segments: list[dict], model: str, src: str,
              progress=None, models: list[str] | None = None,
              clients: list | None = None, checkpoint=None,
              cache_dir: Path | None = None,
              window: tuple[int, int] | None = None,
              system: str | None = None,
              budgets: dict | None = None) -> list[dict]:
    """Translate every line, spread over every model and key available.

    Three things multiply here, and they are independent:

    * **models** - the limit is metered per model, so each extra model is a
      separate token bucket on the same account;
    * **keys** - the limit is metered per organization, so a key from a
      different account is another set of buckets again. A second key from the
      *same* account is not, and adds nothing;
    * **streams** - a reply takes a second or more and the bucket refills while
      it is awaited, so several requests are kept in flight per bucket.

    Lines that already carry an `en` value are left alone, so a re-run after a
    failure only pays for what is still missing.
    """
    pool_clients = clients or [client]
    pool_models = [model] + [m for m in (models or []) if m and m != model]
    total = len(segments)

    # `window` restricts which lines get translated without hiding the rest:
    # the full transcript stays available so each batch still sees the lines
    # either side of it for continuity. Used by the streaming pipeline, which
    # translates a few minutes at a time so playback can start early.
    first, last = window if window else (0, total)
    # Batches sit on one grid across the whole transcript, not one per window.
    # A window boundary almost never lands on a multiple of BATCH, so cutting
    # batches at it would leave a stub - a two-line request paying the same
    # ~1900 token system prompt as a thirty-line one. Snapping outwards instead
    # means a window translates a few lines past its own end, which the next
    # window then finds already done: measured on a 54 minute episode, 48
    # requests either way instead of 55, and the daily request allowance is
    # what runs out first.
    first = (first // BATCH) * BATCH
    last = min(total, -(-last // BATCH) * BATCH)
    # Batch-level granularity: a batch is translated all at once, so a batch
    # with every line already filled in is one that finished last time.
    jobs = [(lo, min(lo + BATCH, last)) for lo in range(first, last, BATCH)]
    jobs = [(lo, hi) for lo, hi in jobs
            if not all(segments[i].get("en", "").strip() for i in range(lo, hi))]
    if not jobs:
        return segments
    resumed = (last - first) - sum(hi - lo for lo, hi in jobs)

    glossary = ""
    if system is None and total > BATCH:
        # The cast list costs a request and is the same for every run over the
        # same transcript, so a retry reads it back instead of rebuilding it.
        cache = (cache_dir / "glossary.txt") if cache_dir else None
        if cache is not None and cache.is_file():
            glossary = cache.read_text(encoding="utf-8")
        if not glossary:
            if progress:
                progress("Working out the cast", 44)
            # On a pool, run it on a helper so the primary model starts
            # translating with a full budget rather than one already spent here.
            glossary = build_glossary(client, segments, pool_models[-1], src)
            if cache is not None and glossary:
                try:
                    cache.write_text(glossary, encoding="utf-8")
                except Exception:
                    pass

    # The streaming pipeline builds this once and hands it to every window, so
    # the cast list is settled before the first line is translated and stays
    # identical for the whole video.
    if system is None:
        system = build_system(src, glossary)

    lock = threading.Lock()
    state = {"done": resumed, "error": None, "checkpoint": checkpoint}

    def report() -> None:
        if progress:
            d = state["done"]
            bits = []
            if len(pool_models) > 1:
                bits.append(f"{len(pool_models)} models")
            if len(pool_clients) > 1:
                bits.append(f"{len(pool_clients)} keys")
            note = f" on {' and '.join(bits)}" if bits else ""
            span = last - first
            progress(f"Translating line {min(d + 1, span)} of {span}{note}",
                     45 + int(25 * d / max(1, span)))

    state["report"] = report
    report()

    # A bucket belongs to one (key, model) pair, so that is what gets its own
    # budget. Several streams then share each bucket to keep it busy.
    #
    # The caller can hand these in to keep them alive across calls, and the
    # streaming pipeline does. A budget only learns what is left by reading it
    # off a reply, so a fresh one believes the whole allowance is available:
    # start every window with new budgets and each one opens by firing every
    # stream at once, blowing the per-minute limit and buying a rate-limit
    # backoff that stalls playback for minutes.
    if budgets is None:
        budgets = {}
    for ci in range(len(pool_clients)):
        for m in pool_models:
            budgets.setdefault((ci, m), TokenBudget())
    streams = [(ci, m) for ci in range(len(pool_clients)) for m in pool_models
               for _ in range(STREAMS_PER_MODEL)]

    def drain(job) -> None:
        ci, m = job
        _worker(pool_clients[ci], m, system, segments, jobs, lock, state,
                budgets[(ci, m)])

    if len(streams) == 1:
        drain(streams[0])
    else:
        with ThreadPoolExecutor(max_workers=len(streams)) as pool:
            list(pool.map(drain, streams))

    if state["error"]:
        raise state["error"]

    # Every stream returned but work is left over: the models handed their
    # batches back because they are out of allowance for the day. Say so
    # plainly - the alternative is a video that quietly ends up half dubbed.
    if jobs:
        out = sorted({f"{m} ({b.exhausted})" for (_, m), b in budgets.items()
                      if b.exhausted})
        if out:
            raise RuntimeError(
                "Groq's daily token limit is used up on " + ", ".join(out)
                + ". Translation stopped with "
                + f"{sum(hi - lo for lo, hi in jobs)} lines still to do; "
                "everything already translated is saved, so pick this up again "
                "with Try again once the limit resets.")
    return segments


def retranslate_one(client, seg: dict, model: str, src: str) -> str:
    """Re-do a single line - used by the subtitle editor in the UI."""
    system = SYSTEM.format(src_name=LANG_NAMES.get(src, "the source language"))
    user = json.dumps({"translate": [
        {"id": seg["id"], "text": seg["text"], "max_chars": _budget_chars(seg)}]},
        ensure_ascii=False)
    content = _call(client, model, system, user, _max_tokens([seg]), TokenBudget())
    return _parse(content).get(seg["id"], seg.get("en", ""))
