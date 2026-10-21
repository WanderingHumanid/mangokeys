"""Lightweight Async HTTP & WebSocket Server for MangoKeys Transformer Visualizer.

Serves static UI files (HTML, CSS, JS) and streams authentic real-time
cross-attention and beam search telemetry frames over WebSockets and HTTP.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Auto-reexec with local virtualenv python if torch or aiohttp are not in system python
try:
    import aiohttp
    import torch
except ImportError:
    venv_py = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    if venv_py.exists() and sys.executable != str(venv_py):
        os.execv(str(venv_py), [str(venv_py)] + sys.argv)
    raise

# Ensure workspace root is in sys.path so model.src is importable
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from aiohttp import web, WSMsgType
from visualizer.extractor import TensorExtractor

STATIC_DIR = Path(__file__).resolve().parent


class VisualizerServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        self.host = host
        self.port = port
        self.extractor = TensorExtractor(
            checkpoint_path=WORKSPACE_ROOT / "runs/ml_mini/best.pt",
            tokenizer_path=WORKSPACE_ROOT / "runs/ml_mini/tokenizer.json",
        )
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self) -> None:
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/styles.css", self.handle_css)
        self.app.router.add_get("/visualizer.js", self.handle_js)
        self.app.router.add_get("/api/health", self.handle_health)
        self.app.router.add_get("/api/extract", self.handle_extract_get)
        self.app.router.add_post("/api/extract", self.handle_extract_post)
        self.app.router.add_get("/ws", self.handle_websocket)

        assets_dir = STATIC_DIR / "assets"
        if assets_dir.exists():
            self.app.router.add_static("/assets", assets_dir)

    async def handle_index(self, request: web.Request) -> web.Response:
        index_file = STATIC_DIR / "index.html"
        if not index_file.exists():
            return web.Response(text="index.html not found", status=404)
        return web.FileResponse(index_file)

    async def handle_css(self, request: web.Request) -> web.Response:
        css_file = STATIC_DIR / "styles.css"
        if not css_file.exists():
            return web.Response(text="/* styles.css */", content_type="text/css")
        return web.FileResponse(css_file)

    async def handle_js(self, request: web.Request) -> web.Response:
        js_file = STATIC_DIR / "visualizer.js"
        if not js_file.exists():
            return web.Response(text="// visualizer.js", content_type="application/javascript")
        return web.FileResponse(js_file)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "ok",
            "model": "Seq2SeqTransformer",
            "params": "5.37M",
            "device": str(self.extractor.device),
            "layers": {"encoder": 4, "decoder": 4, "heads": 4, "d_model": 256},
            "checkpoint_loaded": self.extractor.checkpoint_path.exists(),
        }, headers={"Access-Control-Allow-Origin": "*"})

    async def handle_extract_get(self, request: web.Request) -> web.Response:
        text = request.query.get("text", "namaskaram")
        beam = int(request.query.get("beam", 4))
        max_steps = int(request.query.get("max_steps", 24))

        telemetry = await asyncio.to_thread(self.extractor.trace, text, beam, max_steps)
        return web.json_response(telemetry, headers={"Access-Control-Allow-Origin": "*"})

    async def handle_extract_post(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            data = {}
        text = data.get("text", "namaskaram")
        beam = int(data.get("beam", 4))
        max_steps = int(data.get("max_steps", 24))

        telemetry = await asyncio.to_thread(self.extractor.trace, text, beam, max_steps)
        return web.json_response(telemetry, headers={"Access-Control-Allow-Origin": "*"})

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)

        # Send greeting / readiness info
        await ws.send_json({
            "type": "ready",
            "model": "Seq2SeqTransformer",
            "params": "5.37M",
            "default_word": "namaskaram",
        })

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                        req_type = payload.get("type", "input")
                        if req_type in ("input", "trace"):
                            text = payload.get("text", "")
                            beam = int(payload.get("beam", 4))
                            max_steps = int(payload.get("max_steps", 24))

                            telemetry = await asyncio.to_thread(
                                self.extractor.trace, text, beam, max_steps
                            )
                            await ws.send_json(telemetry)
                        elif req_type == "ping":
                            await ws.send_json({"type": "pong"})
                    except Exception as err:
                        await ws.send_json({"type": "error", "message": str(err)})
                elif msg.type == WSMsgType.ERROR:
                    print(f"[WebSocket] Connection closed with exception: {ws.exception()}")
        finally:
            await ws.close()

        return ws

    def run(self) -> None:
        print(f"============================================================")
        print(f" MangoKeys Real-Time Neural Transformer Visualizer Server    ")
        print(f" Listening at: http://{self.host}:{self.port}              ")
        print(f" WebSocket:   ws://{self.host}:{self.port}/ws              ")
        print(f" Model:       Seq2Seq (5.37M Params, 4+4 Layers, d=256)     ")
        print(f"============================================================")
        web.run_app(self.app, host=self.host, port=self.port, print=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="MangoKeys Neural Transformer Visualizer Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    args = parser.parse_args()

    server = VisualizerServer(host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
