"""














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
from urllib.parse import urlparse

import pytest

EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"


#


def _load(name: str):
    """
"""
    path = EXAMPLES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_example_{name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#


class _JSONServer:
    """"""

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


class _GrokVideoServer:
    """

"""

    def __init__(self, request_id: str, done_body: dict[str, Any]) -> None:
        self.request_id = request_id
        self.done_body = done_body
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or "0")
                if length:
                    self.rfile.read(length)
                self._send({"request_id": outer.request_id})

            def do_GET(self):
                self._send(outer.done_body)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )

    def __enter__(self) -> "_GrokVideoServer":
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
    """"""

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


class _AnthropicBatchServer:
    """
"""

    def __init__(self, batch_id: str, texts: list[str]) -> None:
        self.batch_id = batch_id
        self.texts = texts
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self, payload: bytes, ctype: str = "application/json"):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or "0")
                if length:
                    self.rfile.read(length)
                body = {
                    "id": outer.batch_id,
                    "type": "message_batch",
                    "processing_status": "in_progress",
                }
                self._send(json.dumps(body).encode("utf-8"))

            def do_GET(self):
                path = urlparse(self.path).path
                if path == f"/v1/messages/batches/{outer.batch_id}":
                    body = {"id": outer.batch_id, "processing_status": "ended"}
                    return self._send(json.dumps(body).encode("utf-8"))
                if path == f"/v1/messages/batches/{outer.batch_id}/results":
                    lines = []
                    for i, text in enumerate(outer.texts):
                        lines.append(
                            json.dumps(
                                {
                                    "custom_id": f"req-{i}",
                                    "result": {
                                        "type": "succeeded",
                                        "message": {
                                            "content": [{"type": "text", "text": text}],
                                            "stop_reason": "end_turn",
                                            "usage": {
                                                "input_tokens": 5,
                                                "output_tokens": 7,
                                            },
                                        },
                                    },
                                }
                            )
                        )
                    return self._send(
                        "\n".join(lines).encode("utf-8"), ctype="application/x-ndjson"
                    )
                self._send(b'{"error":"not found"}')

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )

    def __enter__(self) -> "_AnthropicBatchServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


#


_ANTHROPIC_OK = {
    "content": [{"type": "text", "text": "ok"}],
    "usage": {"input_tokens": 1, "output_tokens": 1},
    "stop_reason": "end_turn",
}

_OPENAI_FILE_OK = {"id": "file-zzz", "object": "file"}


#
#
_ANTHROPIC_CACHED = {
    "content": [{"type": "text", "text": "The API is fast."}],
    "usage": {
        "input_tokens": 12,
        "output_tokens": 6,
        "cache_creation_input_tokens": 1100,
        "cache_read_input_tokens": 0,
    },
    "stop_reason": "end_turn",
}

#
#
_OPENAI_REASONING = {
    "choices": [{"message": {"content": "9 sheep remain."}}],
    "usage": {
        "prompt_tokens": 40,
        "completion_tokens": 25,
        "completion_tokens_details": {"reasoning_tokens": 17},
    },
}


def _vertex_music_response() -> dict[str, Any]:
    encoded = base64.b64encode(b"RIFF\x01\x02\x03<fake-wav>").decode("ascii")
    return {"predictions": [{"audioContent": encoded, "mimeType": "audio/wav"}]}


_ANTHROPIC_MODELS = {
    "data": [
        {
            "type": "model",
            "id": "claude-opus-4-7",
            "display_name": "Claude Opus 4.7",
            "created_at": "2026-04-14T00:00:00Z",
            "max_input_tokens": 1000000,
            "max_tokens": 128000,
        }
    ],
    "has_more": False,
    "last_id": "claude-opus-4-7",
}


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


#


def _redirect(module, factory_name: str, base_url: str) -> None:
    """
"""
    from llmkit.builders import (  # local import to avoid cycle on collection
        anthropic,
        google,
        grok,
        openai,
        vertex,
    )

    real = {
        "anthropic": anthropic,
        "openai": openai,
        "google": google,
        "vertex": vertex,
        "grok": grok,
    }[factory_name]

    def patched(key: str):
        client = real(key)
        client.provider.base_url = base_url
        return client

    setattr(module, factory_name, patched)


#


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
    #
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


def test_catalogue_runs() -> None:
    ex = _load("catalogue")
    with _JSONServer(_ANTHROPIC_MODELS) as server:
        _redirect(ex, "anthropic", server.url)
        asyncio.run(ex.main())


def test_music_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # example writes ./out.wav
    ex = _load("music")
    with _JSONServer(_vertex_music_response()) as server:
        _redirect(ex, "vertex", server.url)
        asyncio.run(ex.main())
    assert (tmp_path / "out.wav").exists()


def test_video_runs() -> None:
    ex = _load("video")
    done = {
        "status": "done",
        "video": {"url": "https://vidgen.x.ai/abc/video.mp4", "duration": 8},
        "model": "grok-imagine-video",
    }
    with _GrokVideoServer("vid-123", done) as server:
        _redirect(ex, "grok", server.url)
        asyncio.run(ex.main())


def test_batch_runs() -> None:
    ex = _load("batch")
    texts = ["Bonjour", "Hola", "Hallo"]
    with _AnthropicBatchServer("batch_ex", texts) as server:
        _redirect(ex, "anthropic", server.url)
        asyncio.run(ex.main())


def test_caching_runs() -> None:
    ex = _load("caching")
    with _JSONServer(_ANTHROPIC_CACHED) as server:
        _redirect(ex, "anthropic", server.url)
        asyncio.run(ex.main())


def test_reasoning_runs() -> None:
    ex = _load("reasoning")
    with _JSONServer(_OPENAI_REASONING) as server:
        _redirect(ex, "openai", server.url)
        asyncio.run(ex.main())
