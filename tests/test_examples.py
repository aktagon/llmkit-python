"""Smoke runner for python/examples/*.py.

Each example imports a real provider factory from llmkit.builders, builds
a Client, and calls a chain → terminal. The test substitutes the
factory with one that pins the Client's base_url at a mock HTTP server
serving a canned response, then runs the example's main() coroutine.

This catches the README/example bug class that py_compile and import-only
checks miss:
  * builder access form — `c.text()` vs `c.text` (TypeError at runtime)
  * Response field naming — `resp.usage` vs `resp.usage` (AttributeError)
  * builder surface — calling `Agent.history(...)` (AttributeError)

The mock servers reuse the same shapes as tests/test_builders.py and
tests/test_image.py.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"


# ---------- module loader -----------------------------------------------------


def _load(name: str):
    """Load an example as a fresh module so per-test monkey-patches are
    isolated."""
    path = EXAMPLES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_example_{name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- mock servers ------------------------------------------------------


class _JSONServer:
    """Replies to every POST/GET with the same canned JSON body."""

    def __init__(self, body: dict[str, Any]) -> None:
        outer = self
        self.body = body
        self.last_path = ""

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self):
                outer.last_path = self.path
                length = int(self.headers.get("Content-Length") or "0")
                if length:
                    self.rfile.read(length)
                payload = json.dumps(outer.body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_POST = _send
            do_GET = _send

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )

    def __enter__(self) -> "_JSONServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


class _SSEServer:
    """Streams a canned SSE event sequence on POST."""

    def __init__(self, events: list[str]) -> None:
        outer = self
        self.events = events

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or "0")
                if length:
                    self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for line in outer.events:
                    self.wfile.write((line + "\n").encode("utf-8"))
                    self.wfile.flush()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )

    def __enter__(self) -> "_SSEServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


# ---------- canned response bodies --------------------------------------------


_ANTHROPIC_OK = {
    "content": [{"type": "text", "text": "ok"}],
    "usage": {"input_tokens": 1, "output_tokens": 1},
    "stop_reason": "end_turn",
}

_OPENAI_FILE_OK = {"id": "file-zzz", "object": "file"}


def _google_image_response() -> dict[str, Any]:
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\n<fake>").decode("ascii")
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": encoded}},
                    ]
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10},
    }


_ANTHROPIC_SSE = [
    "event: content_block_delta",
    'data: {"delta":{"text":"Hi"}}',
    "",
    "event: message_delta",
    'data: {"usage":{"output_tokens":1}}',
    "",
    "event: message_stop",
    'data: {"type":"message_stop","stop_reason":"end_turn"}',
    "",
]


# ---------- patch helper ------------------------------------------------------


def _redirect(module, factory_name: str, base_url: str) -> None:
    """Replace `module.<factory_name>` with a wrapper that pins
    `provider.base_url` to the mock before returning the Client."""
    from llmkit.builders import (  # local import to avoid cycle on collection
        anthropic,
        google,
        openai,
    )

    real = {
        "anthropic": anthropic,
        "openai": openai,
        "google": google,
    }[factory_name]

    def patched(key: str):
        client = real(key)
        client.provider.base_url = base_url
        return client

    setattr(module, factory_name, patched)


# ---------- tests -------------------------------------------------------------


def test_quickstart_runs() -> None:
    ex = _load("quickstart")
    with _JSONServer(_ANTHROPIC_OK) as server:
        _redirect(ex, "anthropic", server.url)
        asyncio.run(ex.main())


def test_agent_runs() -> None:
    ex = _load("agent")
    with _JSONServer(_ANTHROPIC_OK) as server:
        _redirect(ex, "anthropic", server.url)
        asyncio.run(ex.main())


def test_streaming_runs() -> None:
    ex = _load("streaming")
    with _SSEServer(_ANTHROPIC_SSE) as server:
        _redirect(ex, "anthropic", server.url)
        asyncio.run(ex.main())


def test_image_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # example writes ./out.png
    ex = _load("image")
    with _JSONServer(_google_image_response()) as server:
        _redirect(ex, "google", server.url)
        asyncio.run(ex.main())
    assert (tmp_path / "out.png").exists()


def test_upload_runs(tmp_path, monkeypatch) -> None:
    # Path branch reads ./data.pdf from CWD; pre-create it in tmp.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.pdf").write_bytes(b"%PDF-1.4 stub")
    ex = _load("upload")
    with _JSONServer(_OPENAI_FILE_OK) as server:
        _redirect(ex, "openai", server.url)
        asyncio.run(ex.main())


def test_middleware_runs() -> None:
    ex = _load("middleware")
    with _JSONServer(_ANTHROPIC_OK) as server:
        _redirect(ex, "anthropic", server.url)
        asyncio.run(ex.main())
