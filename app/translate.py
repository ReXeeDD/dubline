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
   max_chars is a slot to FILL, not just a ceiling to stay under. The actor is
   speaking for that whole time, so a line that uses a third of its room has
   dropped something the source said, and what is left is silence over a moving
   mouth. Translate everything in the line - both the narration and any speech
   in it - and let the length land near the budget.
5. Keep names, numbers, units and technical terms accurate and consistent.
   - ONE character gets ONE English spelling, everywhere. The recogniser is
     guessing at homophones, so it writes the same name several ways across a
     transcript; those are the same person, not two people. Pick the spelling
     the cast list gives and never drift from it.
   - Stray digits and loose syllables are recogniser noise, not words. Drop
     them. Never turn one into somebody's name, and never read a number back
     as the syllables it is said with. "You found the target Longjue Shenfei
     69 froze" is "I found the target, Longjue Shenfei, and she froze" - the
     69 is nothing at all, and it is certainly not a person called Liu Jiu.
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
   - Some scenes have no main character in them at all. The camera follows
     other people and the story narrates THEM. Those lines stay in the third
     person, and the narration must never borrow the first person from
     whoever the scene happens to be following. Watch for this when a scene
     stays with one other character for a while - it is exactly where the
     narration starts speaking as her.
     Two lines from a scene the hero is not in:
       "Then she returned to Lianshan County, and Shen Xiaoxue heard the news"
          - NOT "Then I returned to Lianshan County, and I heard the news"
       "Li Zhiyue saw Li Bai coming toward her, and she was instantly on
        her guard"
          - NOT "I saw Li Bai coming toward me, and I was on my guard"
     The first person belongs to ONE person for the whole video: the narrator
     named in the cast list. No other character is ever "I" in narration, no
     matter how long the camera stays on them. When they speak out loud, they
     say "I" inside their own speech - that is different, and that is fine.
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
11. NARRATION VOICE. The narration is the main character telling his own story
   after the fact. Write it the way a person actually tells a story out loud.
   - PAST TENSE throughout the narration. "I looked up at the sky", never "I
     look up at the sky". Dialogue keeps whatever tense the speaker uses, and
     a general truth stays present ("gold buys anything here").
   - Join what belongs together instead of stacking stubs. Not "I nodded. Yes.
     I am forging a longsword." but "I nodded - I was forging a plain
     longsword."
   - Let a line run into its own second half the way speech does:
     "I was watching the sky, and that was when I saw Li Zhiyue coming straight
     at me."
   - Plain and confident. No narrator flourishes, no "little did I know", no
     explaining a joke the picture already told.
   - Keep the pronoun for the same person steady inside a line. If the line
     starts with "I", it does not switch to "you" halfway through.
12. FORMS OF ADDRESS. This genre is full of relationship words used where
   English uses a pronoun or a name, and copying them across is the single
   clearest sign of a translation nobody rewrote.
   - A speaker referring to HIMSELF by his title says "I". A husband telling
     his wives he will cook says "I'll make us something to eat" - never
     "Husband will make you food". Same for a servant, a father, a teacher.
   - Addressing someone by a bare relationship noun is not English. "Wife,
     come here" and "Wives, let's go" are not things people say. Use a natural
     English address - "ladies", "you two", the person's name - or simply drop
     it, which is what English usually does.
   - Speaking TO someone by their title becomes "you": "Does husband want tea?"
     is "Do you want some tea?"
   - Honorifics that carry real weight in the scene (Your Majesty, Master,
     Elder) stay, because dropping them loses the relationship.
13. TERMS OF ART. Work out what a phrase MEANS before rendering it. A literal
   reading of a compound noun is how this material goes wrong.
   - A tax counted per person is a "poll tax" or a "head tax" - it is money.
     It is never "handing over heads". Read the scene: if people are paying
     it to an official, it is a tax.
   - The cultivation-story furniture has settled English: sect, elder, realm,
     breakthrough, spirit stone, pill, technique, inner disciple.
   - A game-style prompt is a notification, and English keeps those clipped.
     "Congratulations, Host, on your successful marriage. Reward obtained:
      Vitality Pill" is said as "Marriage complete. Reward: Vitality Pill."
     Never repeat the whole announcement for each of three rewards - say it
     once and list what followed.

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
        "Resolve each one from context and the cast list instead.\n"
        "CRITICAL - the narration is written in the SECOND person. This genre "
        "narrates the hero's own life at him as 'ni' (you): 'ni chuan yue guo "
        "lai shi ge qiong xiao zi' is not somebody addressing him, it is how "
        "the story tells us what happened to him. Rendered literally it turns "
        "the hero into a stranger being lectured, and it is the single most "
        "common defect in this material. Every narration 'ni' and 'ni de' is "
        "I / me / my.\n"
        "Worked example - four consecutive narration lines:\n"
        "  'ni chuan yue guo lai shi ge shen wu fen wen de qiong xiao zi'\n"
        "     -> 'I came into this world with nothing to my name'\n"
        "  'ni de mu qin nan chan er si fu qin xuan ze bao zhu hai zi'\n"
        "     -> 'My mother died giving birth to me, and my father chose to "
        "save the baby'\n"
        "  'zi cong fu mu guo shi ni du yao kuai huo bu xia qu'\n"
        "     -> 'Since my parents died I had barely been getting by'\n"
        "  'ni ding zhe na tiao tu lu'  -> 'I stared down the dirt road'\n"
        "Only when a character is speaking OUT LOUD to another person does "
        "'ni' stay 'you' - and those lines are the ones inside quotation "
        "marks, or plainly addressed to someone standing there."),
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
# Quoted speech, so it can be taken out of the line before looking for a POV
# slip. Lines routinely carry both - 'Good." You nodded again and walked to the
# forge.' - and skipping any line with a quote in it hid every one of those,
# while testing the whole line would flag the dialogue for being in the second
# person, which is exactly where it belongs.
_QUOTE_CHAR = re.compile(r"[\"“”]")


def _narration_of(text: str, inside: bool) -> tuple[str, bool]:
    """The part of a line that is NOT someone speaking, plus the state after it.

    Quotes are counted rather than matched with a regex, and the open/closed
    state is carried from one subtitle to the next. Dialogue routinely runs
    across a line break:

        ...Aunt Yun looked at us and said, "These are your village's bachelors.
        Fewer than usual. Just the three of you." Uncle Wang nodded.

    Neither line contains a matched pair, so pair-matching left the dialogue in
    place and both lines were flagged for saying "your" - the two false alarms
    this rule produced on a real episode were both exactly this.
    """
    out, pos = [], 0
    for m in _QUOTE_CHAR.finditer(text):
        if not inside:
            out.append(text[pos:m.start()])
        inside = not inside
        pos = m.end()
    if not inside:
        out.append(text[pos:])
    return " ".join(out), inside
_SECOND_PERSON_ANY = re.compile(
    r"\b(?:you|your|yours|yourself|yourselves)\b|\byou'(?:re|d|ve|ll)\b",
    re.I)
_UNFINISHED = re.compile(
    r"(?:\.\.\.|…)\s*$|\b(?:and|but|so|then|the|an?|to|of|with|that|because|"
    r"as|for|if|when|while|is|was|were)\s*[.,]?\s*$", re.I)
_SOURCE_SCRIPT = re.compile(r"[぀-ヿ一-鿿가-힯]")
_OVER_BUDGET = 1.55      # times max_chars before a line is worth shortening

# The other half of that rule. Only ever penalising a long line teaches the
# repair pass one move - cut - and it will happily cut the content out of a
# line to make the number go down. What comes back fits the slot perfectly and
# has lost the scene: "It's not just three poll taxes, they stack." "I know."
# I nodded.' collapsed to 'I nodded to show I understood.', which is thirty
# characters of a seventy-seven character slot and three seconds of silence
# over an actor who is plainly still talking.
#
# How much English a source line is worth is measurable, and it depends on the
# script. Across 2,310 finished Chinese lines the median is 2.79 English
# characters per Chinese character and the 5th percentile is 1.53; across 2,635
# Vietnamese lines, where the source is already Latin and much longer, the
# median is 0.81 per source character and the 5th percentile 0.52. The floors
# below sit under both 5th percentiles, so a line has to be shorter than
# essentially every real translation in the library before it is flagged.
_EXPANSION_FLOOR = {"cjk": 1.20, "latin": 0.42}
# ...and it must also be leaving real silence behind. A short line that still
# fills its slot is just a concise line, and re-asking for those is how a
# repair pass makes good work worse.
_UNDERFILL = 0.55
_MIN_SOURCE = 10         # below this the source is too short to judge by


def _dropped_content(seg: dict) -> bool:
    """Is this line far shorter than its source could possibly translate to?"""
    en = (seg.get("en") or "").strip()
    src = (seg.get("text") or "").strip()
    if not en or not src:
        return False
    cjk = len(_SOURCE_SCRIPT.findall(src))
    # A source that is mostly CJK is counted in CJK characters; anything else
    # is counted whole, which is what the Vietnamese figure was measured on.
    size, floor = ((cjk, _EXPANSION_FLOOR["cjk"]) if cjk >= len(src) * 0.5
                   else (len(src), _EXPANSION_FLOOR["latin"]))
    if size < _MIN_SOURCE:
        return False
    return (len(en) < size * floor
            and len(en) < _budget_chars(seg) * _UNDERFILL)


# The pronoun this genre narrates the hero's own life at him with, and the
# third-person pronouns it uses for everybody else.
_SECOND_PERSON_SRC = {"zh": (re.compile("你"), re.compile("他|她"))}
# Share of SOURCE lines carrying that pronoun before a video counts as being
# narrated at its main character. Measured on the episodes here the two groups
# are far apart: the three second-person Chinese ones sit at 58, 63 and 65 per
# cent, while the third-person Chinese one is at 24 per cent and carries the
# third-person pronouns in 61 per cent of its lines.
SECOND_PERSON_SHARE = 0.40
# Fewer lines than this and the share above is noise rather than a measurement.
# It used to be 40, which is more caution than the numbers ask for and it cost
# real accuracy: the share is the discriminating signal - the second-person
# episodes here sit at 58-69 per cent and the third-person one at 24 - so a
# dozen lines already separate them cleanly. At 40 a short clip got no
# point-of-view anchor at all, which is the one case where the model's own
# answer ("NARRATOR|-") is least likely to be right.
NARRATION_SAMPLE = 12


def _is_narrated(segments: list[dict], glossary: str, src: str = "") -> bool:
    """Is this video the main character's own story, told at him as "you"?

    The cast list answers this when it can - but it is one cheap call, and on
    two of the three Chinese episodes here it left the narrator out, which
    switched the whole point-of-view check off for exactly the videos that
    needed it most.

    The finished English cannot be the fallback, tempting as it looks: a video
    mistranslated wholly into the second person contains no first-person
    narration to find, so asking the output whether it reads as first person
    only agrees with the mistake. The SOURCE is the honest witness - this genre
    writes the hero's own life at him in the second person, line after line,
    and no third-person story does that.
    """
    # Either wording of the anchor built by _format_cast.
    if ("narrator and main character" in glossary
            or "THE MAIN CHARACTER is" in glossary):
        return True
    marker = _SECOND_PERSON_SRC.get(src)
    if not marker:
        return False
    second, third = marker
    lines = [t for t in (s.get("text") or "" for s in segments) if t.strip()]
    if len(lines) < NARRATION_SAMPLE:
        return False
    you = sum(1 for t in lines if second.search(t))
    them = sum(1 for t in lines if third.search(t))
    return you / len(lines) >= SECOND_PERSON_SHARE and you >= them


def _continues(segments: list[dict], i: int) -> bool:
    """Does the next subtitle carry this one's sentence on?

    Written the way a reader would judge it: a line picked up by something
    starting in lower case, or by an ellipsis, was never broken - it was split.
    """
    if i + 1 >= len(segments):
        return False
    en = (segments[i].get("en") or "").strip()
    # Speech running on into the next subtitle leaves its quotation open, and
    # the line after it starts mid-sentence however it is capitalised.
    if _narration_of(en, False)[1]:
        return True
    nxt = (segments[i + 1].get("en") or "").strip()
    return bool(nxt) and (nxt[0].islower() or nxt[0] in ".,…\"”")


def audit(segments: list[dict], glossary: str = "", src: str = "",
          window: tuple[int, int] | None = None) -> dict[int, list[str]]:
    """Which finished lines look wrong, and why. No API calls."""
    narrated = _is_narrated(segments, glossary, src)
    lo, hi = window if window else (0, len(segments))
    faults: dict[int, list[str]] = {}
    # Is a line of dialogue still open from the subtitle before? Speech does
    # run across a line break, but only just: an opening quotation whose
    # partner never arrives would otherwise flip the reading of every line
    # after it. Observed on a real scene - one unclosed quote left four
    # following lines parsed inside-out, and real dialogue was flagged as
    # narration. So an open quotation is honoured for one line and then
    # given up on.
    spoken, carried = False, 0

    for i in range(lo, hi):
        seg = segments[i]
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
        # Once the quoted speech is taken out, what is left is narration - and
        # in a video whose narrator IS the main character, narration has no
        # second person in it at all. Anything still saying "you" or "your"
        # there is the source's own second-person narration coming through
        # untranslated, which is the defect that makes the hero sound like a
        # stranger being lectured at.
        narration, spoken = _narration_of(en, spoken)
        carried = carried + 1 if spoken else 0
        if carried > 1:
            spoken, carried = False, 0
        if (narrated and _SECOND_PERSON_ANY.search(narration)
                and not en.rstrip().endswith("?")):
            bad.append("narration must be first person - this is the main "
                       "character talking about himself, so 'I', not 'you'. "
                       "But if this line is really one character SPEAKING to "
                       "another, 'you' is correct: leave it exactly as it is")
        # A line ending mid-clause is only wrong if nothing picks it up. The
        # source carries no punctuation at all and its sentences routinely run
        # across two or three subtitles, so the translation does too - and
        # flagging those meant asking for a rewrite of 38 lines that were
        # already right, out of the 173 this caught across the library.
        if _UNFINISHED.search(en) and not _continues(segments, i):
            bad.append("the sentence stops in the middle - finish it")
        if _SOURCE_SCRIPT.search(en):
            bad.append("untranslated source text is left in the line")
        limit = _budget_chars(seg)
        if len(en) > limit * _OVER_BUDGET:
            bad.append(f"far too long to speak in the time - {len(en)} "
                       f"characters where {limit} fit, say it shorter")
        elif _dropped_content(seg):
            bad.append(f"far too short for what the source says - {len(en)} "
                       f"characters where {limit} fit. Something in the line "
                       f"was left out: translate ALL of it, including any "
                       f"speech, and use the room")

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
   The opposite fault is just as real. A line flagged as too SHORT is not
   asking to be trimmed further - it has dropped part of what the source says,
   and the slot is now silence over an actor who is still speaking. Read the
   source again, put back what went missing, and fill the room you are given.
   Shortening a line that is already too short is the worst thing you can do
   to it.
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
                    seg["fixed"] = True
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
        except Exhausted:
            # A per-day limit is not a transient failure. Retrying it three
            # times with a backoff spends six seconds and two more requests
            # proving what the first reply already said, once per stream and
            # once per batch - and it delays handing this batch to a model
            # that could still be doing it.
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


# Pronouns, in the source scripts and in English. A cast list is a list of
# PEOPLE, and a pronoun is not one - but this genre narrates the hero's life at
# him in the second person, so "ni" is far and away the most frequent token in
# the transcript and the model duly reports it as the leading character.
#
# The entry it produces is "ni=you (male)", which lands in the system prompt
# under "use these spellings exactly". That is a direct, concrete instruction to
# render every "ni" as "you" - and it beats the general prose rule three
# paragraphs up telling the model that narration is "I". Measured on a real
# clip, this one line was the difference between narration that says "I" and
# narration that says "you": the cast list was arguing against the brief, and
# winning. Nothing that is only a pronoun may reach it.
_PRONOUN_SRC = {
    "你", "妳", "您", "我", "他", "她", "它", "牠", "咱",
    "你们", "你們", "我们", "我們", "他们", "他們", "她们", "她們",
    "自己", "大家", "别人", "別人",
    "ta", "nguoi", "người", "ngươi", "han", "hắn", "nang", "nàng",
    "chang", "chàng", "toi", "tôi", "minh", "mình", "y", "gã", "ga",
}
_PRONOUN_EN = {
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "we", "us", "our", "ours", "they", "them", "their",
    "oneself", "self", "everyone", "others",
}


def _is_pronoun(source: str, english: str) -> bool:
    """Is this cast entry a pronoun rather than a character?

    Either side is enough to disqualify it. The source side catches the common
    case; the English side catches a name the model transliterated into a
    pronoun, and any source pronoun the lists here have not met yet.
    """
    return (source.strip().strip("的") in _PRONOUN_SRC
            or english.strip().lower() in _PRONOUN_EN)


# Words that are what somebody is called, not who they are. The prompt asks for
# names only, but these arrive anyway because they are the most frequent tokens
# in the transcript - and each one does real damage, because the cast list is
# presented to the translator as a spelling to use exactly. "niang zi=Wife"
# turns a man calling his wives over to dinner into "Wife, come here", and
# "fu jun=Husband" is what produced "Husband will cook for you" where the
# narrator simply means "I".
_GENERIC_CAST = {
    "wife", "wives", "husband", "spouse", "concubine", "bride", "brides",
    "host", "master", "mistress", "lord", "lady", "ladies", "madam", "sir",
    "mother", "father", "mom", "dad", "parents", "son", "sons", "daughter",
    "daughters", "brother", "brothers", "sister", "sisters", "aunt", "uncle",
    "grandfather", "grandmother", "elder", "elders", "senior", "junior",
    "child", "children", "girl", "girls", "boy", "boys", "woman", "women",
    "man", "men", "people", "everyone", "villager", "villagers", "crowd",
    "chief", "system", "narrator", "protagonist", "friend", "friends",
    "the three women", "three daughters", "little star", "little stars",
}


def _is_generic(source: str, english: str) -> bool:
    """Is this entry a common noun, a crowd label, or simply untranslated?

    An entry whose English side came back unchanged is one the model did not
    translate at all - it puts source script into a brief that is otherwise
    entirely English, and the translator copies it straight through into the
    subtitle.
    """
    en = english.strip().lower().strip(".,")
    return (not en
            or en in _GENERIC_CAST
            or en.lstrip("the ") in _GENERIC_CAST
            or english.strip() == source.strip()
            or bool(_SOURCE_SCRIPT.search(english)))


# A relationship word states a gender outright, and it is far more reliable
# than either the model's guess or the transcript's pronouns - spoken Mandarin
# makes he, she and it the same sound, so Whisper picks a character at random
# and the dub inherits the coin toss. Measured on one finished episode, a male
# character was called "she" in 25 lines and "he" in 19; two women were called
# "he" in 30 and 18 lines. Where the relation names a role, it decides.
_FEMALE_WORDS = (
    "wife", "wives", "mother", "mom", "mum", "daughter", "sister", "aunt",
    "grandmother", "granddaughter", "niece", "madam", "lady", "girl", "queen",
    "empress", "princess", "saintess", "priestess", "widow", "bride",
    "girlfriend", "fiancee", "fiancée", "concubine", "maid", "actress",
    "stepsister", "stepmother", "mistress", "nun",
)
_MALE_WORDS = (
    "husband", "father", "dad", "son", "brother", "uncle", "grandfather",
    "grandson", "nephew", "sir", "lord", "boy", "king", "emperor", "prince",
    "monk", "groom", "boyfriend", "fiance", "fiancé", "master", "man",
    "stepbrother", "stepfather", "gentleman",
)


# Chinese given names carry gender in the characters themselves, and far more
# strongly than the surrounding pronouns do. Most of the female set is built on
# the woman radical or on the classic "graceful/gentle" vocabulary; the male set
# on strength, height and rank. Kept to characters that are overwhelmingly one
# or the other in a given name - the merely common ones (清, 明, 华, 欣) are
# left out on purpose, because a wrong confident answer here is worse than none.
#
# This is what settles the case the relation cannot: the model returned
# "苏婉柔 ... male" while calling her a love interest, and 婉柔 - "gentle and
# soft" - is not a name a man in this genre is given.
_NAME_F = set("婉柔娟婷芳丽娜雅秀淑静洁玲燕莉妍姗媛琳蕾萍慧敏兰梅菊凤妃姬嫣娥娇妤婕瑶珊薇茜蓉颖怡妮娅莹璐岚彤霞黛娴")
_NAME_M = set("强伟军刚勇峰磊涛龙虎昊杰豪雄栋铭钢钧霆霸擎骁锋岳猛冲彪坤")


def _gender_from_name(names: list[str]) -> str:
    """'f'/'m'/'' from the characters of a Chinese given name.

    Surnames are left in: they are drawn from a small fixed set that shares
    almost nothing with either group above, so scanning the whole name costs
    nothing and saves having to know where the surname ends.
    """
    joined = "".join(names)
    f = sum(1 for c in joined if c in _NAME_F)
    m = sum(1 for c in joined if c in _NAME_M)
    if f > m:
        return "f"
    if m > f:
        return "m"
    return ""


def _settle_gender(stated: str, relation: str, names: list[str] | None = None) -> str:
    """'f' or 'm' or '' - the relation outranks the model's own answer.

    The model is asked for a gender and usually gets it right, but when it does
    not, the relation it wrote in the next field almost always gives it away:
    nothing called "his stepsister" is male. Checking the relation costs a
    string search and catches exactly the case the pronouns cannot.
    """
    said = (stated or "").strip()[:1].lower()
    said = said if said in ("f", "m") else ""
    words = re.findall(r"[a-z]+", (relation or "").lower())
    female = any(w in _FEMALE_WORDS for w in words)
    male = any(w in _MALE_WORDS for w in words)
    if female and not male:
        return "f"
    if male and not female:
        return "m"
    # No role word to go on. The name itself is the next best witness, and a
    # better one than the model's bare guess: it answered "male" for a
    # character called 婉柔.
    from_name = _gender_from_name(names or [])
    return from_name or said


def _format_cast(mc: dict | None, npcs: list[dict], segments: list[dict],
                 src: str) -> str:
    """The character chart, as the translator is given it.

    Written as a table rather than a comma-separated list because it carries
    four things per character now - every spelling the recogniser produced, the
    one English name, the gender and who they are - and the spellings are the
    half that stops one person walking through the dub under three names.
    """
    lines: list[str] = []
    if mc:
        alias = ", ".join(mc["names"])
        him = {"f": "she/her", "m": "he/him"}.get(mc["sex"], "he/him")
        lines.append(
            f"THE MAIN CHARACTER is {mc['en']} - written in the source as "
            f"{alias}. {mc['relation'] or 'the narrator'}. "
            f"The narration is {mc['en']} telling their own story, so every "
            f"narration line is \"I\", \"me\", \"my\" - never their name and "
            f"never \"you\". Inside somebody else's speech they are \"you\"; "
            f"spoken about by others they are {him}.")
    elif _is_narrated(segments, "", src):
        lines.append(
            "The narration is the MAIN CHARACTER telling his own story. He is "
            "never named in it - the source writes his life at him in the "
            "second person, and every one of those lines is him speaking as "
            "\"I\". This is the narrator and main character of the whole "
            "video, and no other character narrates.")
    if npcs:
        lines.append(
            "\nEVERYONE ELSE. Use the English name exactly as written here for "
            "every spelling on its row, and the gender here for every pronoun "
            "about them - it outranks the he/she in the transcript, which is a "
            "recogniser guess:")
        for n in npcs[:30]:
            g = {"f": "female - she/her", "m": "male - he/him"}.get(n["sex"], "?")
            alias = ", ".join(n["names"])
            rel = f" - {n['relation']}" if n["relation"] else ""
            lines.append(f"  {alias} = {n['en']} ({g}){rel}")
    return "\n".join(lines)


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
        "\nOne line per character, in this format:\n"
        "  ROLE|spellings|English name|gender|relation\n"
        "\nROLE is MC for the main character, NPC for everybody else.\n"
        "\nTHE MAIN CHARACTER. Exactly one, and the opening of the video is "
        "where to find them: this genre states the hero's situation in its "
        "first few lines and the story is told from inside their head "
        "afterwards. The main character is the one the narration is ABOUT - "
        "whose thoughts you are given, who other characters address directly. "
        "They are usually never named in the narration itself, only when "
        "somebody speaks to them, so read the dialogue for it. Output their "
        "line first. If the story really is about no one in particular, output "
        "no MC line at all rather than guessing.\n"
        "\nSPELLINGS. The recogniser guesses at homophones, so ONE person "
        "arrives under two or three spellings, and a familiar form of a name "
        "is another. List every spelling for that character separated by "
        "commas, most frequent first. This matters most for the main "
        "character, who is addressed by more people in more ways than anyone "
        "else. Do not put two spellings of one person on two lines - that is "
        "what makes one character walk through the dub under three names.\n"
        "\nGENDER is f or m. Work it out from what the story CALLS them, never "
        "from the pronouns: this transcript's he/she are recogniser guesses "
        "and are wrong often. A wife, mother, daughter, sister, aunt, "
        "grandmother, madam, empress or saintess is female. A husband, father, "
        "son, brother, uncle, grandfather, sir, lord or emperor is male. Those "
        "words are reliable; the pronoun beside them is not.\n"
        "\nRELATION is a few words on who they are to the main character - "
        "'his stepsister', 'his best friend', 'the girl he loves', 'his "
        "father'. Write it in English. If they have no relation to the main "
        "character, say what they are instead - 'a reporter', 'the sect "
        "elder'.\n"
        "\nDo NOT list, however often they occur:\n"
        " - ordinary words used as a form of address - wife, husband, madam, "
        "master, host, senior, young lady. They are not names, and listing one "
        "makes the translator write \"Husband will cook the rice\" where a "
        "person would say \"I'll cook the rice\";\n"
        " - labels for a group - the three women, the elders, the villagers;\n"
        " - figures of speech and scenery - stars in her eyes, the mountain "
        "gate - which are not characters at all;\n"
        " - anyone the story never actually names. Leave them out rather than "
        "inventing a name.\n"
        "A relationship title EARNS a place only when it is that character's "
        "actual name in the story - someone always called Aunt Yun and never "
        "anything else. Then its English must be a name in English too.\n"
        "\nExample:\n"
        "  MC|江晨,江城,小晨|Jiang Chen|m|the narrator, a student sitting his exams\n"
        "  NPC|顾清涵|Gu Qinghan|f|his stepsister, the girl he confessed to\n"
        "  NPC|宋景年,宋锦年|Song Jingnian|m|his best friend, who betrays him\n"
        "\nOutput nothing but those lines.")
    # The opening is where the main character is established and a sample
    # spread over the rest is where the supporting cast turns up. Sampling
    # alone misses the lead: on a real episode every line it drew came from the
    # middle, and it answered "third person". The opening is given more of it
    # now, because identifying the MC correctly is what the rest hangs on.
    opening = " / ".join(s["text"] for s in segments[:55])
    user = (f"# opening of the video - the main character is established here\n"
            f"{opening}\n\n"
            f"# spread over the rest of the video\n{_sample(segments, 1800)}")
    try:
        raw = _strip_think(_call(client, model, system, user, 600, TokenBudget()))
    except Exception:
        return ""

    mc, npcs = None, []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*• ")
        if "|" not in line or len(line) > 200:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        role = parts[0].upper()
        if role not in ("MC", "NPC"):
            continue
        spellings = [a.strip() for a in parts[1].split(",") if a.strip()]
        english = parts[2]
        if not spellings or not english:
            continue
        if _is_pronoun(spellings[0], english) or _is_generic(spellings[0], english):
            continue
        relation = parts[4].strip() if len(parts) > 4 else ""
        sex = _settle_gender(parts[3] if len(parts) > 3 else "",
                             relation, spellings)
        entry = {"names": spellings[:6], "en": english,
                 "sex": sex, "relation": relation[:80]}
        if role == "MC" and mc is None:
            mc = entry
        else:
            npcs.append(entry)
    return _format_cast(mc, npcs, segments, src)


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
              cache_dir: Path | None = None,
              fallbacks: tuple[str, ...] = ()) -> str:
    """The cast list for a transcript, read from cache when one was made.

    Tries the other models before giving up. The chart is one cheap call, but
    everything downstream leans on it - who the lead is, which name each
    spelling belongs to, which pronoun each character takes - so coming back
    empty because one model happened to be out of allowance for the day costs
    the whole video its point of view and its genders.
    """
    cache = (cache_dir / "glossary.txt") if cache_dir else None
    if cache is not None and cache.is_file():
        try:
            return cache.read_text(encoding="utf-8")
        except OSError:
            pass
    glossary = ""
    for m in (model, *fallbacks):
        if not m:
            continue
        glossary = build_glossary(client, segments, m, src)
        if glossary:
            break
    if cache is not None and glossary:
        try:
            cache.write_text(glossary, encoding="utf-8")
        except OSError:
            pass
    return glossary


def _projected_cost(system: str, segments: list[dict], lo: int, hi: int,
                    model: str) -> int:
    """What _call will charge this batch against the per-minute budget.

    Kept in step with _call deliberately: the merge below has to predict the
    same number _call will compute, or it builds a batch that cannot be sent.
    """
    reply = _max_tokens(segments[lo:hi])
    if _REASONING.get(model) == "low":
        reply = int(reply * REASONING_ALLOWANCE)
    return _estimate(system) + _estimate(_payload(segments, lo, hi)) + reply


def _worker(client, model: str, system: str, segments: list[dict],
            jobs: list, lock, state: dict, budget: TokenBudget) -> None:
    """Drain the shared job list using one model, paced by a shared budget.

    Several of these run per model. They share one TokenBudget because the rate
    limit is per model, not per connection - so the streams pace against each
    other rather than each believing it has the whole allowance.

    A model whose reasoning cannot be switched off spends a fixed overhead on
    every request, so it takes bigger bites: the system prompt and the thinking
    are then amortised over more lines instead of being paid per small batch -
    but only as far as the per-minute budget can actually carry in one request.
    """
    handed_back = 0
    while True:
        with lock:
            if state.get("error"):
                return
            # This bucket has nothing left today. Nothing it can do here but
            # get out of the way - every request it sends comes back 429, and
            # while it is sending them it is not translating anything.
            if budget.exhausted:
                return
            if not jobs:
                # Empty is not the same as finished. Another stream may still
                # be holding a batch it is about to hand back - a model that
                # runs out of daily allowance mid-request does exactly that.
                # Leaving here on sight of an empty list is what stranded that
                # batch with nobody to pick it up, and translate() then failed
                # the whole video for "daily token limit used up" while this
                # model still had budget for it. Measured on a simulation of
                # the old exit: 152 of 200 runs stranded work.
                if not state["holders"]:
                    return
                waiting = True
            else:
                waiting = False
                lo, hi = jobs.pop(0)
            if not waiting:
                # Merge in following batches when this model wants larger bites,
                # while the result still fits in one request.
                #
                # Without the fit test this livelocks, and silently: a reasoning
                # model's max_tokens is doubled to leave room for the thinking, so
                # even a plain 30-line batch projects past the 8000-token minute
                # (measured: 8891). _run_batch then raises TooLarge, the handler
                # below splits the batch in two and puts both halves back - and
                # this loop immediately rejoins them, because they are adjacent.
                # Split, merge, split, merge, with no request ever sent and no
                # error ever raised. Observed as a dub frozen at "Translating line
                # 91 of 150" with the process still burning CPU.
                if _REASONING.get(model) == "low":
                    while jobs and jobs[0][0] == hi and (hi - lo) < BATCH * 2:
                        grown = jobs[0][1]
                        if _projected_cost(system, segments, lo, grown,
                                           model) > budget.headroom():
                            break
                        hi = jobs.pop(0)[1]

                # ...and split back down to what one request can actually carry.
                # For a reasoning model this is the normal path rather than an edge
                # case: the doubled reply allowance puts even a plain 30-line batch
                # at 8891 tokens against an 8000-token minute, and only about 15
                # lines fit. Deciding it here costs nothing; leaving it to the 413
                # costs a wasted round trip per split.
                while (hi - lo) > 1 and _projected_cost(
                        system, segments, lo, hi, model) > budget.headroom():
                    mid = lo + (hi - lo) // 2
                    jobs.insert(0, (mid, hi))
                    hi = mid

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
                if not wait_a_moment:
                    # Held by this stream now: another stream must not read the
                    # empty list as "finished" while this one can still hand
                    # the batch back.
                    state["holders"] += 1
            else:
                wait_a_moment = False

        if waiting:
            time.sleep(0.2)      # a handback is still possible
            continue

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
                state["holders"] -= 1
            return
        except TooLarge:
            if hi - lo > 1:
                mid = lo + (hi - lo) // 2
                with lock:
                    jobs.insert(0, (mid, hi))
                    jobs.insert(0, (lo, mid))
                    state["holders"] -= 1
                continue
            with lock:
                state["holders"] -= 1
                state["error"] = RuntimeError(
                    f"A single subtitle line exceeds the {budget.headroom()} "
                    "tokens-per-minute limit.")
            return
        except Exception as e:
            with lock:
                state["holders"] -= 1
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
            state["holders"] -= 1
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
    # "holders" is how many streams are holding a batch right now. A stream
    # that finds the queue empty waits on this rather than leaving, because a
    # held batch can still come back (see _worker).
    state = {"done": resumed, "error": None, "checkpoint": checkpoint,
             "holders": 0}

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
