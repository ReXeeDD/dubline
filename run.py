"""Start the Dubline server.

    python run.py            -> http://localhost:8000
    python run.py --port 9000 --open
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Local video dubbing studio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open the browser on start")
    ap.add_argument("--lan", action="store_true",
                    help="also serve to other devices on your network")
    args = ap.parse_args()

    host = "0.0.0.0" if args.lan else args.host

    try:
        import uvicorn
    except ImportError:
        sys.exit("uvicorn is not installed. Run:  pip install \"uvicorn[standard]\"")

    from app.config import BIN, LIBRARY, ffmpeg_ready

    if not ffmpeg_ready():
        sys.exit(f"ffmpeg was not found in {BIN}.\nRun:  python setup.py")

    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{args.port}"
    print()
    print("  Dubline is running")
    print(f"    {url}")
    if args.lan:
        print("    (also reachable from other devices on your network)")
    print(f"    library: {LIBRARY}")
    print("    press Ctrl+C to stop")
    print()

    if args.open:
        import threading

        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run("app.main:app", host=host, port=args.port,
                log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
