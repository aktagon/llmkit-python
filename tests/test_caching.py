"""



"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

from llmkit.caching import apply_caching
from llmkit.providers.generated.providers import PROVIDERS
from llmkit.types import Options, Provider


#


class _CacheCreateServer:
    """"""

    def __init__(self, response_body: dict[str, Any]) -> None:
        self.response_body = response_body
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
                self.send_response(200)
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


#


def test_apply_caching_openai_is_noop() -> None:
    #
    body: dict[str, Any] = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    snapshot = json.dumps(body, sort_keys=True)
    apply_caching(body, Provider(name="openai", api_key="k"), Options(), PROVIDERS["openai"])
    assert json.dumps(body, sort_keys=True) == snapshot


#


def test_apply_caching_anthropic_wraps_system_string_in_blocks() -> None:
    body: dict[str, Any] = {"system": "You are a helpful assistant.", "messages": []}
    apply_caching(body, Provider(name="anthropic", api_key="k"), Options(), PROVIDERS["anthropic"])
    #
    assert isinstance(body["system"], list)
    assert body["system"][0]["type"] == "text"
    assert body["system"][0]["text"] == "You are a helpful assistant."
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_apply_caching_anthropic_skips_when_no_system() -> None:
    #
    body: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}
    snapshot = json.dumps(body, sort_keys=True)
    apply_caching(body, Provider(name="anthropic", api_key="k"), Options(), PROVIDERS["anthropic"])
    assert json.dumps(body, sort_keys=True) == snapshot


def test_apply_caching_anthropic_skips_when_system_is_empty_string() -> None:
    body: dict[str, Any] = {"system": "", "messages": []}
    apply_caching(body, Provider(name="anthropic", api_key="k"), Options(), PROVIDERS["anthropic"])
    #
    assert body["system"] == ""


#


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

    #
    assert server.received_path == "/v1beta/cachedContents"
    #
    assert server.received_body is not None
    assert server.received_body["model"] == "models/gemini-2.5-pro"
    assert "ttl" in server.received_body
    assert server.received_body["systemInstruction"] == {
        "parts": [{"text": "You are a helpful assistant."}]
    }
    #
    assert body["cachedContent"] == "cachedContents/abc123"
    assert "system_instruction" not in body


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

    #
    assert server.received_body is not None
    assert server.received_body["ttl"] == "600s"
