"""Unit tests for llmkit.caching — apply_caching dispatch across the
three caching modes (automatic, explicit, resource). Anthropic exercises
the in-place body mutation path; Google exercises the pre-flight POST +
body-rewrite path via a mock /v1beta/cachedContents server; OpenAI is
the automatic (no-op) sentinel."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

import pytest

from llmkit.caching import apply_caching
from llmkit.errors import APIError
from llmkit.providers.generated.providers import PROVIDERS
from llmkit.types import Options, Provider


# ---------- mock server (Google resource-caching pre-flight) ----------


class _CacheCreateServer:
    """Single-shot HTTPS-shaped mock for the cachedContents create endpoint."""

    def __init__(self, response_body: dict[str, Any], status_code: int = 200) -> None:
        self.response_body = response_body
        self.status_code = status_code
        self.received_path = ""
        self.received_body: dict[str, Any] | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):  # silence noise
                pass

            def do_POST(self):
                outer.received_path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                outer.received_body = json.loads(self.rfile.read(length).decode("utf-8"))
                payload = json.dumps(outer.response_body).encode("utf-8")
                self.send_response(outer.status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_CacheCreateServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


# ---------- automatic caching (OpenAI) ----------


def test_apply_caching_openai_is_noop() -> None:
    # OpenAI uses AutomaticCaching: body is unchanged; no pre-flight request.
    body: dict[str, Any] = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    snapshot = json.dumps(body, sort_keys=True)
    apply_caching(body, Provider(name="openai", api_key="k"), Options(), PROVIDERS["openai"])
    assert json.dumps(body, sort_keys=True) == snapshot


# ---------- explicit caching (Anthropic, TopLevelField) ----------


def test_apply_caching_anthropic_wraps_system_string_in_blocks() -> None:
    body: dict[str, Any] = {"system": "You are a helpful assistant.", "messages": []}
    apply_caching(body, Provider(name="anthropic", api_key="k"), Options(), PROVIDERS["anthropic"])
    # system: str → system: [{type:"text", text, cache_control:{type:"ephemeral"}}]
    assert isinstance(body["system"], list)
    assert body["system"][0]["type"] == "text"
    assert body["system"][0]["text"] == "You are a helpful assistant."
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_apply_caching_anthropic_skips_when_no_system() -> None:
    # No system prompt → nothing to cache; body unchanged.
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}
    snapshot = json.dumps(body, sort_keys=True)
    apply_caching(body, Provider(name="anthropic", api_key="k"), Options(), PROVIDERS["anthropic"])
    assert json.dumps(body, sort_keys=True) == snapshot


def test_apply_caching_anthropic_skips_when_system_is_empty_string() -> None:
    body: dict[str, Any] = {"system": "", "messages": []}
    apply_caching(body, Provider(name="anthropic", api_key="k"), Options(), PROVIDERS["anthropic"])
    # Empty string → no rewrite (truthiness check in _apply_explicit).
    assert body["system"] == ""


# ---------- resource caching (Google, pre-flight POST + reference rewrite) ----------


def test_apply_caching_google_creates_resource_and_rewrites_body() -> None:
    response = {"name": "cachedContents/abc123", "model": "models/gemini-2.5-pro"}
    with _CacheCreateServer(response) as server:
        body: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": "You are a helpful assistant."}]},
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
        provider = Provider(
            name="google",
            api_key="test-key",
            model="gemini-2.5-pro",
            base_url=server.url,
        )
        apply_caching(body, provider, Options(), PROVIDERS["google"])

    # Pre-flight POST hit the cachedContents endpoint.
    assert server.received_path == "/v1beta/cachedContents"
    # Create body carried the system_instruction + model + ttl.
    assert server.received_body is not None
    assert server.received_body["model"] == "models/gemini-2.5-pro"
    assert "ttl" in server.received_body
    assert server.received_body["systemInstruction"] == {
        "parts": [{"text": "You are a helpful assistant."}]
    }
    # Body now references the cached resource and the inline system is gone.
    assert body["cachedContent"] == "cachedContents/abc123"
    assert "system_instruction" not in body


def test_apply_caching_google_surfaces_provider_error() -> None:
    # BUG-016: when the cachedContents create is rejected (e.g. Gemini's
    # "too small" 400 below its per-model token floor), the caller must get
    # a clean, typed APIError carrying the provider's OWN message — not an
    # opaque raw-body wrap. llmkit invents no size floor; it reports whatever
    # the provider rejected with. This structured error (provider + status +
    # message) is the substrate the opt-in capability telemetry reads.
    error_envelope = {
        "error": {
            "code": 400,
            "message": "Cached content is too small. total_token_count=653, min_total_token_count=1024",
            "status": "INVALID_ARGUMENT",
        }
    }
    with _CacheCreateServer(error_envelope, status_code=400) as server:
        body: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": "small system prompt"}]},
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        }
        provider = Provider(
            name="google",
            api_key="test-key",
            model="gemini-2.5-pro",
            base_url=server.url,
        )
        with pytest.raises(APIError) as exc_info:
            apply_caching(body, provider, Options(), PROVIDERS["google"])

    err = exc_info.value
    # parse_error ran: provider name set, provider's own message extracted.
    assert err.provider == "google"
    assert err.status_code == 400
    assert "min_total_token_count=1024" in err.message


def test_apply_caching_google_respects_explicit_cache_ttl() -> None:
    response = {"name": "cachedContents/ttl-test"}
    with _CacheCreateServer(response) as server:
        body: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": "x"}]},
            "contents": [],
        }
        provider = Provider(
            name="google", api_key="k", model="gemini-2.5-pro", base_url=server.url
        )
        opts = Options(cache_ttl=600)
        apply_caching(body, provider, opts, PROVIDERS["google"])

    # cache_ttl=600 → "600s" string on the wire.
    assert server.received_body is not None
    assert server.received_body["ttl"] == "600s"
