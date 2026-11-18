"""Start the Dubline server.

    python run.py            -> http://localhost:8000
    python run.py --port 9000 --open
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
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
        sys.exit(f"Working ffmpeg and ffprobe are required in {BIN} or on PATH.\n"
                 "Run:  python setup.py")

    class DublineServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started:
                print(f"  Dubline is running at {url}", flush=True)
                if args.lan:
                    print("    (also reachable from other devices on your network)")
                print(f"    library: {LIBRARY}\n    press Ctrl+C to stop", flush=True)
                if args.open:
                    threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()

    # Reserve the port before application startup resumes any queued jobs.
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as listener:
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, args.port))
        except (OSError, OverflowError) as exc:
            sys.exit(f"Could not listen on {host} port {args.port}: {exc}")
        address = "localhost" if host in ("127.0.0.1", "0.0.0.0", "::") else host
        if ":" in address:
            address = f"[{address}]"
        url = f"http://{address}:{listener.getsockname()[1]}"
        server = DublineServer(uvicorn.Config("app.main:app", host=host, port=args.port,
                                               log_level="warning", access_log=False))
        server.run(sockets=[listener])
        if not server.started:
            sys.exit("Dubline failed to start.")


if __name__ == "__main__":
    main()
