"""HTTP runtime tests for the catalogue (ADR-019 Phase 3).

Each test spins up an ``http.server`` thread, points the Client at its
URL via ``with_base_url``, and asserts pagination / parser / error
classification behaviour. Mirror of go/models_test.go and
ts/tests/catalogue_http.test.ts.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse

import pytest

from llmkit.builders import anthropic, cohere, google, openai
from llmkit.models import (
    ErrModelsNotSupported,
    ErrModelsScope,
    ErrModelsUnavailable,
)
from llmkit.types import Provider


HandlerFn = Callable[[str, str, dict[str, list[str]], dict[str, str]], tuple[int, bytes]]


class _StubHandler(BaseHTTPRequestHandler):
    handler_fn: HandlerFn  # set per test
    calls: list[tuple[str, dict[str, list[str]]]] = []

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler protocol)
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        self.__class__.calls.append((parsed.path, query))
        headers = {k.lower(): v for k, v in self.headers.items()}
        status, body = self.__class__.handler_fn(parsed.path, parsed.query, query, headers)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs) -> None:  # silence test noise
        return


def _start_server(handler_fn: HandlerFn) -> tuple[HTTPServer, str, list[tuple[str, dict[str, list[str]]]]]:
    _StubHandler.handler_fn = handler_fn
    _StubHandler.calls = []
    srv = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv, f"http://127.0.0.1:{srv.server_port}", _StubHandler.calls


def test_scoped_list_anthropic_cursor_pagination() -> None:
    page1 = json.dumps({
        "data": [
            {"type": "model", "id": "claude-opus-4-7", "display_name": "Claude Opus 4.7",
             "created_at": "2026-04-14T00:00:00Z", "max_input_tokens": 1000000, "max_tokens": 128000},
            {"type": "model", "id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6",
             "created_at": "2026-04-14T00:00:00Z", "max_input_tokens": 1000000, "max_tokens": 128000},
        ],
        "has_more": True,
        "last_id": "claude-sonnet-4-6",
    }).encode()
    page2 = json.dumps({
        "data": [
            {"type": "model", "id": "claude-haiku-4-5-20251001", "display_name": "Claude Haiku 4.5",
             "created_at": "2026-04-14T00:00:00Z", "max_input_tokens": 200000, "max_tokens": 64000},
        ],
        "has_more": False,
        "last_id": "claude-haiku-4-5-20251001",
    }).encode()

    def handler(path: str, _raw_query: str, query: dict[str, list[str]], headers: dict[str, str]) -> tuple[int, bytes]:
        if headers.get("x-api-key") != "test-key":
            return 401, b'{"error":"missing key"}'
        if query.get("after_id") == ["claude-sonnet-4-6"]:
            return 200, page2
        return 200, page1

    srv, base, calls = _start_server(handler)
    try:
        c = anthropic("test-key").with_base_url(base)
        models = asyncio.run(c.models.provider(Provider(name="anthropic", api_key="test-key")).list())
        assert len(models) == 3
        assert len(calls) == 2
        assert calls[1][1]["after_id"] == ["claude-sonnet-4-6"]
        opus = next(m for m in models if m.id == "claude-opus-4-7")
        assert len(opus.capabilities) > 0
    finally:
        srv.shutdown()


def test_scoped_list_google_opaque_token_pagination() -> None:
    page1 = json.dumps({
        "models": [
            {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash",
             "description": "Stable", "inputTokenLimit": 1048576, "outputTokenLimit": 65536},
        ],
        "nextPageToken": "opaque-cursor-xyz",
    }).encode()
    page2 = json.dumps({
        "models": [
            {"name": "models/gemini-2.5-pro", "displayName": "Gemini 2.5 Pro",
             "description": "Stable", "inputTokenLimit": 1048576, "outputTokenLimit": 65536},
        ],
    }).encode()

    def handler(path: str, _raw_query: str, query: dict[str, list[str]], headers: dict[str, str]) -> tuple[int, bytes]:
        if query.get("key") != ["test-key"]:
            return 401, b'{"error":"missing key"}'
        if query.get("pageToken") == ["opaque-cursor-xyz"]:
            return 200, page2
        return 200, page1

    srv, base, calls = _start_server(handler)
    try:
        c = google("test-key").with_base_url(base)
        models = asyncio.run(c.models.provider(Provider(name="google", api_key="test-key")).list())
        assert len(models) == 2
        assert len(calls) == 2
        # Google parser strips "models/" prefix.
        assert models[0].id == "gemini-2.5-flash"
    finally:
        srv.shutdown()


def test_scoped_list_openai_non_paginated() -> None:
    body = json.dumps({
        "object": "list",
        "data": [
            {"id": "gpt-5", "object": "model", "created": 1715367049, "owned_by": "system"},
            {"id": "gpt-4o", "object": "model", "created": 1715367049, "owned_by": "system"},
        ],
    }).encode()

    def handler(path: str, raw_query: str, query: dict[str, list[str]], headers: dict[str, str]) -> tuple[int, bytes]:
        if headers.get("authorization") != "Bearer test-key":
            return 401, b'{"error":"missing key"}'
        if raw_query != "":
            return 400, b'{"error":"unexpected query"}'
        return 200, body

    srv, base, calls = _start_server(handler)
    try:
        c = openai("test-key").with_base_url(base)
        models = asyncio.run(c.models.provider(Provider(name="openai", api_key="test-key")).list())
        assert len(calls) == 1
        assert len(models) == 2
    finally:
        srv.shutdown()


def test_scoped_list_403_scope_maps_to_err_models_scope() -> None:
    def handler(*args, **kwargs) -> tuple[int, bytes]:
        return 403, b'{"error":{"message":"Missing scopes: api.model.read"}}'

    srv, base, _ = _start_server(handler)
    try:
        c = openai("test-key").with_base_url(base)
        with pytest.raises(ErrModelsScope):
            asyncio.run(c.models.provider(Provider(name="openai", api_key="test-key")).list())
    finally:
        srv.shutdown()


def test_scoped_list_503_maps_to_err_models_unavailable() -> None:
    def handler(*args, **kwargs) -> tuple[int, bytes]:
        return 503, b'{"error":"down"}'

    srv, base, _ = _start_server(handler)
    try:
        c = anthropic("test-key").with_base_url(base)
        with pytest.raises(ErrModelsUnavailable):
            asyncio.run(c.models.provider(Provider(name="anthropic", api_key="test-key")).list())
    finally:
        srv.shutdown()


def test_scoped_list_endpointless_provider_keeps_not_supported() -> None:
    # No server needed — runtime short-circuits before any HTTP call.
    c = cohere("test-key")
    with pytest.raises(ErrModelsNotSupported):
        asyncio.run(c.models.provider(Provider(name="cohere", api_key="k")).list())


def test_scoped_get_anthropic_single_record() -> None:
    body = json.dumps({
        "type": "model", "id": "claude-opus-4-7", "display_name": "Claude Opus 4.7",
        "created_at": "2026-04-14T00:00:00Z", "max_input_tokens": 1000000, "max_tokens": 128000,
    }).encode()

    def handler(path: str, *args, **kwargs) -> tuple[int, bytes]:
        if path != "/v1/models/claude-opus-4-7":
            return 404, b'{"error":"not found"}'
        return 200, body

    srv, base, calls = _start_server(handler)
    try:
        c = anthropic("test-key").with_base_url(base)
        m = asyncio.run(
            c.models.provider(Provider(name="anthropic", api_key="test-key"))
            .get("claude-opus-4-7")
        )
        assert len(calls) == 1
        assert m.id == "claude-opus-4-7"
        assert len(m.capabilities) > 0
    finally:
        srv.shutdown()


def test_models_live_partial_success_typed_provider_error() -> None:
    def handler(*args, **kwargs) -> tuple[int, bytes]:
        return 503, b'{"error":"down"}'

    srv, base, _ = _start_server(handler)
    try:
        c = openai("test-key").with_base_url(base)
        res = asyncio.run(c.models.live())
        assert res.models == []
        err = res.errors.get("openai")
        assert err is not None
        assert err.kind == "unavailable"
    finally:
        srv.shutdown()
