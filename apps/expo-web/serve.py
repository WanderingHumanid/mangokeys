"""Serve the expo demo as static files.

    python apps/expo-web/serve.py --port 8000

The demo needs a real http server because the browser has to fetch the wasm
binaries and the model. Nothing is uploaded, the server only reads files.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    handler = functools.partial(Handler, directory=str(ROOT))
    with socketserver.TCPServer(("0.0.0.0", args.port), handler) as server:
        print(f"serving {ROOT} on http://localhost:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
