"""Find the right number of concurrent requests per model.

The translator sends one request at a time per model, waits for the whole reply,
then sends the next. Each reply takes a second or more, so most of the wall
clock is spent waiting on the network rather than on the rate limit - and the
token bucket refills the entire time, unused.

This measures a workload shaped like a real translation batch at several
concurrency levels. Every level completes the same amount of work: a request
refused with 429 is retried rather than dropped, so the comparison is honest.

    python tools/concurrency.py
"""
from __future__ import annotations

import concurrent.futures as cf
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import load_settings          # noqa: E402

URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "qwen/qwen3.6-27b"

# Shaped like a real batch: 30 subtitle lines in, ~1200 tokens of translation out.
BATCHES = 10
MAX_TOKENS = 1200
LEVELS = (1, 3, 6)

SYSTEM = "You translate Chinese subtitles to English. Output id|translation lines."
BODY = "\n".join(f"{i}|40|这是第{i}行字幕，讲述一个关于海洋的故事。" for i in range(1, 31))


def one(key: str) -> tuple[int, int, int]:
    """One batch, retrying a rate-limit refusal. Returns (ok, tokens, retries)."""
    retries = 0
    for _ in range(6):
        try:
            r = httpx.post(
                URL,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"model": MODEL,
                      "messages": [{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": BODY}],
                      "max_tokens": MAX_TOKENS, "reasoning_effort": "none"},
                timeout=180)
        except Exception:
            retries += 1
            time.sleep(2)
            continue
        if r.status_code == 200:
            used = 0
            try:
                used = r.json().get("usage", {}).get("total_tokens", 0)
            except Exception:
                pass
            return 1, used, retries
        if r.status_code == 429:
            retries += 1
            time.sleep(3)
            continue
        return 0, 0, retries
    return 0, 0, retries


def budget(key: str) -> int:
    try:
        r = httpx.post(URL, headers={"Authorization": f"Bearer {key}"},
                       json={"model": MODEL,
                             "messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 1, "reasoning_effort": "none"},
                       timeout=30)
        rem = r.headers.get("x-ratelimit-remaining-tokens", "")
        return int(rem) if rem.isdigit() else -1
    except Exception:
        return -1


def wait_full(key: str) -> None:
    print("   refilling", end="", flush=True)
    for _ in range(80):
        if budget(key) >= 7300:
            print(" ok")
            return
        print(".", end="", flush=True)
        time.sleep(4)
    print(" (proceeding anyway)")


def main() -> int:
    key = load_settings().get("groq_api_key", "").strip()
    if not key:
        print("No Groq key saved in Settings.")
        return 2

    print(f"\nModel    : {MODEL}")
    print(f"Workload : {BATCHES} batches, 30 lines each, {MAX_TOKENS} max_tokens")
    print(f"Levels   : {', '.join(map(str, LEVELS))} concurrent request(s)\n")

    rows = []
    for w in LEVELS:
        wait_full(key)
        print(f"   {w} concurrent...", end="", flush=True)
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=w) as pool:
            res = list(pool.map(lambda _: one(key), range(BATCHES)))
        took = time.time() - t0
        ok = sum(r[0] for r in res)
        tok = sum(r[1] for r in res)
        rt = sum(r[2] for r in res)
        print(f" {took:.1f}s  ({ok}/{BATCHES} ok, {rt} retries)")
        rows.append((w, took, ok, tok, rt))

    print(f"\n{'streams':>8}{'seconds':>10}{'done':>6}{'retries':>9}"
          f"{'tokens':>9}{'vs serial':>11}")
    print("-" * 53)
    base = rows[0][1]
    for w, took, ok, tok, rt in rows:
        print(f"{w:>8}{took:>10.1f}{ok:>6}{rt:>9}{tok:>9}{base / took:>10.2f}x")

    best = min(rows, key=lambda r: r[1])
    print("\n" + "=" * 60)
    print(f"  Fastest: {best[0]} concurrent, {base / best[1]:.2f}x quicker than serial.")
    if best[4] > BATCHES * 0.3:
        print("  But it needed a lot of retries - back off one level.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
