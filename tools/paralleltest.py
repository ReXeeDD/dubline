"""Does more parallelism against ONE model bucket finish work faster?

This is the question behind "two keys hitting the same pool must still be
quicker". It has nothing to do with how many keys there are - a key is just a
credential, and two keys pointed at one bucket send exactly the same requests
two connections could send with one key. So the honest test is: hold the key
fixed, vary only concurrency, and time an identical workload.

The workload is deliberately larger than the bucket's banked capacity. A short
burst finishes at whatever speed the bank allows and tells you nothing; only
work that outlasts the bank measures the sustained refill rate, which is what a
real episode runs into.

    python tools/paralleltest.py
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

REQUESTS = 18            # total requests in the workload
TOKENS = 600             # max_tokens each -> ~10800, comfortably over the bank
LEVELS = (1, 4)          # concurrency settings to compare
PROMPT = ("Write a detailed paragraph about the sea, its depth and its moods. "
          "Keep writing until you run out of room.")


def ask(key: str) -> tuple[int, int, int]:
    """Returns (status, completion_tokens, remaining_budget)."""
    try:
        r = httpx.post(
            URL,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
                  "max_tokens": TOKENS, "reasoning_effort": "none"},
            timeout=180)
        used = 0
        try:
            used = r.json().get("usage", {}).get("total_tokens", 0)
        except Exception:
            pass
        rem = r.headers.get("x-ratelimit-remaining-tokens", "")
        return r.status_code, used, (int(rem) if rem.isdigit() else -1)
    except Exception:
        return 0, 0, -1


def wait_full(key: str, target: float = 0.92) -> None:
    """Let the bucket refill, so every run starts from the same place."""
    print("   waiting for the budget to refill", end="", flush=True)
    limit = None
    for _ in range(90):
        st, _, rem = ask_probe(key)
        if limit is None:
            limit = 8000
        if rem >= limit * target:
            print(f"  ({rem} tokens)")
            return
        print(".", end="", flush=True)
        time.sleep(4)
    print("  (gave up waiting)")


def ask_probe(key: str) -> tuple[int, int, int]:
    try:
        r = httpx.post(
            URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1, "reasoning_effort": "none"}, timeout=30)
        rem = r.headers.get("x-ratelimit-remaining-tokens", "")
        return r.status_code, 0, (int(rem) if rem.isdigit() else -1)
    except Exception:
        return 0, 0, -1


def run(key: str, workers: int) -> dict:
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(lambda _: ask(key), range(REQUESTS)))
    took = time.time() - t0
    ok = sum(1 for s, _, _ in res if s == 200)
    r429 = sum(1 for s, _, _ in res if s == 429)
    used = sum(u for _, u, _ in res)
    return {"workers": workers, "seconds": took, "ok": ok, "429": r429,
            "tokens": used, "tok_per_sec": used / took if took else 0}


def main() -> int:
    key = load_settings().get("groq_api_key", "").strip()
    if not key:
        print("No Groq key saved in Settings.")
        return 2

    print(f"\nModel        : {MODEL}")
    print(f"Workload     : {REQUESTS} requests x {TOKENS} max_tokens "
          f"(~{REQUESTS * TOKENS} tokens, larger than the 8000 bank)")
    print(f"Comparing concurrency: {', '.join(map(str, LEVELS))}\n")

    rows = []
    for w in LEVELS:
        wait_full(key)
        print(f"   running with {w} parallel stream(s)...", end="", flush=True)
        r = run(key, w)
        print(f" {r['seconds']:.1f}s")
        rows.append(r)

    print(f"\n{'streams':>8}{'seconds':>10}{'ok':>5}{'429':>5}"
          f"{'tokens':>9}{'tok/sec':>9}")
    print("-" * 47)
    for r in rows:
        print(f"{r['workers']:>8}{r['seconds']:>10.1f}{r['ok']:>5}{r['429']:>5}"
              f"{r['tokens']:>9}{r['tok_per_sec']:>9.0f}")

    if len(rows) >= 2:
        a, b = rows[0], rows[-1]
        speedup = a["seconds"] / b["seconds"] if b["seconds"] else 0
        print("\n" + "=" * 60)
        print(f"  {b['workers']} streams vs {a['workers']}: {speedup:.2f}x")
        if speedup > 1.25:
            print("  Parallelism DOES help - the bucket was not the limit.")
        else:
            print("  Parallelism does not help. The work is capped by the")
            print("  bucket's refill rate, which more connections cannot raise.")
        print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
