"""Do two Groq API keys have separate rate-limit budgets?

Groq's docs say limits are metered per organization, so keys from one org share
a bucket - but that is worth proving rather than believing, because if it is
wrong the translation stage could run several times faster.

A naive version of this test does not work. The budget is a token bucket that
refills continuously at limit/60 per second (about 133 tokens/sec on an 8000 TPM
model), so a request followed by a read shows almost nothing: the bucket has
already topped itself back up. `x-ratelimit-reset-tokens` reads in milliseconds.

So the only way to see the bucket is to drain it faster than it refills. This
fires a concurrent burst on key A, and while that burst is still in flight it
asks key B for its remaining budget. If B sees a bucket drained by traffic it
never sent, the two keys share it.

    python tools/keytest.py <second_key>

Key A is the one already saved in Settings, so only the new key is typed. Keys
are never stored or logged by this script. The burst uses the qwen model's
budget for a few seconds; it refills within a minute.
"""
from __future__ import annotations

import concurrent.futures as cf
import sys
import threading
import time

import httpx

URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "qwen/qwen3.6-27b"
BURST = 12               # concurrent requests, enough to outrun the refill
BURST_TOKENS = 1200      # each

LONG = ("Write an extremely detailed multi-paragraph essay about the ocean, "
        "its currents, its depths and its history. Do not stop early.")


def ask(key: str, max_tokens: int, prompt: str, timeout: float = 90.0):
    """Returns (status, remaining_tokens or -1)."""
    try:
        r = httpx.post(
            URL,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "reasoning_effort": "none"},
            timeout=timeout)
        rem = r.headers.get("x-ratelimit-remaining-tokens", "")
        return r.status_code, (int(rem) if rem.isdigit() else -1)
    except Exception:
        return 0, -1


def main() -> int:
    if len(sys.argv) == 2:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
        from app.config import load_settings
        a, b = load_settings().get("groq_api_key", "").strip(), sys.argv[1].strip()
        if not a:
            print("No key saved in Settings - pass both keys instead.")
            return 2
    elif len(sys.argv) >= 3:
        a, b = sys.argv[1].strip(), sys.argv[2].strip()
    else:
        print(__doc__)
        return 2
    if a == b:
        print("Those are the same key - nothing to compare.")
        return 2

    print(f"\nModel under test: {MODEL}")
    st, base_a = ask(a, 1, "hi")
    st_b, base_b = ask(b, 1, "hi")
    print(f"\n1. Idle budgets")
    print(f"   key A: {base_a:>5} tokens   (http {st})")
    print(f"   key B: {base_b:>5} tokens   (http {st_b})")
    if base_a < 0 or base_b < 0:
        print("\n   One of the keys was rejected - check they are both valid.")
        return 1

    # Sample key B repeatedly while key A is hammering the model.
    samples: list[tuple[int, int]] = []
    stop = threading.Event()

    def watch_b():
        while not stop.is_set():
            samples.append(ask(b, 1, "hi"))
            time.sleep(0.4)

    print(f"\n2. Bursting {BURST} concurrent requests on key A only,")
    print(f"   while key B watches its own budget...")

    watcher = threading.Thread(target=watch_b, daemon=True)
    watcher.start()
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=BURST) as pool:
        results = list(pool.map(lambda _: ask(a, BURST_TOKENS, LONG), range(BURST)))
    stop.set()
    watcher.join(timeout=5)
    took = time.time() - t0

    a_429 = sum(1 for st, _ in results if st == 429)
    a_ok = sum(1 for st, _ in results if st == 200)
    a_low = min((rem for _, rem in results if rem >= 0), default=-1)
    b_low = min((rem for _, rem in samples if rem >= 0), default=-1)
    b_429 = sum(1 for st, _ in samples if st == 429)

    print(f"\n3. Result after {took:.1f}s")
    print(f"   key A: {a_ok} ok, {a_429} rate-limited, lowest budget seen {a_low}")
    print(f"   key B: {len(samples)} probes, {b_429} rate-limited, "
          f"lowest budget seen {b_low}")

    print("\n" + "=" * 64)
    if b_low < 0:
        print("  Inconclusive - key B never returned a readable budget.")
        return 1
    drained = base_b - b_low
    if b_429 > 0 or drained > base_b * 0.3:
        print(f"  SHARED BUCKET. Key B's budget fell by {drained} tokens, and it")
        print(f"  was rate-limited {b_429} time(s), without sending any real work.")
        print("  Adding keys to this organization adds no capacity.")
    elif drained < base_b * 0.05 and b_429 == 0:
        print(f"  SEPARATE BUCKETS. Key B stayed at {b_low} throughout and was")
        print("  never rate-limited while key A was saturated.")
        print("  Multi-key parallelism is real - worth building.")
    else:
        print(f"  Unclear: key B fell by {drained} tokens with {b_429} refusals.")
        print("  Re-run when nothing else is using the account.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
