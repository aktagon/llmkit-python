"""






"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from llmkit.builders import anthropic, google


class _HeaderCapturingServer:
    def __init__(self, response_body: dict[str, Any]) -> None:
        self.response_body = response_body
        self.last_headers: dict[str, str] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):
                pass

            def _serve(self) -> None:
                outer.last_headers = {k.lower(): v for k, v in self.headers.items()}
                length = int(self.headers.get("Content-Length", "0"))
                if length:
                    self.rfile.read(length)
                payload = json.dumps(outer.response_body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                self._serve()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_HeaderCapturingServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


_ANTHROPIC_RESP = {
    "content": [{"type": "text", "text": "pong"}],
    "usage": {"input_tokens": 5, "output_tokens": 1},
}

FLASH_MODEL = "gemini-3.1-flash-image-preview"


def _flash_response(encoded: str) -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": encoded}}]}}
        ],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 1290},
    }


def test_add_header_reaches_wire_text_path() -> None:
    with _HeaderCapturingServer(_ANTHROPIC_RESP) as server:
        c = anthropic("test-key").base_url(server.url).add_header(
            "cf-aig-authorization", "Bearer gw-token"
        )
        resp = asyncio.run(c.text.prompt("ping"))
        assert resp.text == "pong"
        assert server.last_headers["x-api-key"] == "test-key"
        assert server.last_headers["cf-aig-authorization"] == "Bearer gw-token"


def test_add_header_reaches_wire_image_path() -> None:
    encoded = base64.b64encode(b"PNGDATA").decode("ascii")
    with _HeaderCapturingServer(_flash_response(encoded)) as server:
        c = google("test-key").base_url(server.url).add_header(
            "cf-aig-authorization", "Bearer gw-token"
        )
        resp = asyncio.run(c.image.model(FLASH_MODEL).generate("A nano banana dish"))
        assert len(resp.images) == 1
        assert server.last_headers["cf-aig-authorization"] == "Bearer gw-token"


def test_add_header_does_not_clobber_provider_auth() -> None:
    with _HeaderCapturingServer(_ANTHROPIC_RESP) as server:
        c = anthropic("test-key").base_url(server.url).add_header(
            "x-api-key", "attacker-override"
        )
        asyncio.run(c.text.prompt("ping"))
        assert server.last_headers["x-api-key"] == "test-key"


def test_add_header_different_cased_collision_cannot_clobber_auth() -> None:
    #
    #
    with _HeaderCapturingServer(_ANTHROPIC_RESP) as server:
        c = anthropic("test-key").base_url(server.url).add_header(
            "X-API-KEY", "attacker-override"
        )
        asyncio.run(c.text.prompt("ping"))
        assert server.last_headers["x-api-key"] == "test-key"
