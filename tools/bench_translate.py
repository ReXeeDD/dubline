"""Time the real translator on real subtitle lines, serial vs concurrent.

The synthetic benchmarks measured raw API behaviour. This measures the thing
that actually matters: app/translate.py doing its own job, with its batching,
its pacing and its retries, on lines taken from the library.

Every run translates identical lines and reports how many came back non-empty,
so a run that went faster by quietly dropping work cannot look like a win.

    python tools/bench_translate.py [line_count]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import llm, translate                        # noqa: E402
from app.config import LIBRARY, load_settings         # noqa: E402


def sample_lines(n: int) -> list[dict]:
    for f in LIBRARY.glob("*/segments.json"):
        segs = json.loads(f.read_text(encoding="utf-8"))
        out = [{"id": s["id"], "text": s["text"], "start": s["start"],
                "end": s["end"], "en": ""} for s in segs if s.get("text")][:n]
        if out:
            return out
    return []


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    lines = sample_lines(n)
    if not lines:
        print("No transcript in the library to benchmark against.")
        return 2

    cfg = load_settings()
    client = llm.make_translation_client(cfg)
    model = llm.translation_model(cfg)
    print(f"\nModel : {model}")
    print(f"Lines : {len(lines)}  ({translate.BATCH} per batch, "
          f"{-(-len(lines) // translate.BATCH)} batches)\n")

    results = []
    for streams in (1, translate.STREAMS_PER_MODEL):
        translate.STREAMS_PER_MODEL_OVERRIDE = streams
        original = translate.STREAMS_PER_MODEL
        translate.STREAMS_PER_MODEL = streams

        segs = [dict(s) for s in lines]
        print(f"   {streams} stream(s)...", end="", flush=True)
        t0 = time.time()
        try:
            translate.translate(client, segs, model, cfg["source_language"])
        except Exception as e:
            print(f" failed: {e}")
            translate.STREAMS_PER_MODEL = original
            continue
        took = time.time() - t0
        done = sum(1 for s in segs if s.get("en", "").strip())
        print(f" {took:.1f}s   ({done}/{len(segs)} lines translated)")
        results.append((streams, took, done))
        translate.STREAMS_PER_MODEL = original

        if streams == 1:
            print("   letting the bucket refill", end="", flush=True)
            for _ in range(15):
                print(".", end="", flush=True)
                time.sleep(4)
            print()

    if len(results) == 2:
        (s1, t1, d1), (s2, t2, d2) = results
        print(f"\n{'streams':>8}{'seconds':>10}{'lines':>8}{'lines/min':>11}")
        print("-" * 37)
        for s, t, d in results:
            print(f"{s:>8}{t:>10.1f}{d:>8}{d / t * 60:>11.0f}")
        print("\n" + "=" * 56)
        if d1 and d2:
            fair = (d2 / t2) / (d1 / t1)      # compare rates, not wall clock
            print(f"  {s2} streams vs {s1}: {fair:.2f}x on lines per second")
            if abs(d1 - d2) > max(2, 0.05 * d1):
                print(f"  (note: unequal work - {d1} vs {d2} lines)")
        print("=" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
