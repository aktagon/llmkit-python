"""ADR-055 Phase B: the OpenAI Responses chat protocol — response-parse + the
opt-in surface. Mirrors go/responses_test.go. The request-wire golden
(responses-openai) covers the outbound body; these tests cover the reply
envelope (output[] not choices[]), the endpoint switch, and the loud
ValidationError on a provider that lacks the protocol. Mock HTTP server, no
live API calls.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

import pytest

from llmkit import Responses, ValidationError, anthropic, openai


class _MockServer:
    """Single-shot HTTP server that records the request path and serves a canned response."""

    def __init__(self, response_body: dict[str, Any]):
        self.response_body = response_body
        self.received_path = ""

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):  # silence noise
                pass

            def do_POST(self):
                outer.received_path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                payload = json.dumps(outer.response_body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_MockServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


# A Responses reply: an output[] array whose message item carries content[]
# blocks of type output_text, plus input_tokens/output_tokens usage. Live-
# anchored shape 2026-07-02.
_RESPONSES_REPLY = {
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Helsinki."}],
        }
    ],
    "usage": {"input_tokens": 16, "output_tokens": 5},
}

# A default Chat Completions reply.
_CHAT_COMPLETIONS_REPLY = {
    "choices": [{"message": {"content": "Helsinki."}}],
    "usage": {"prompt_tokens": 16, "completion_tokens": 5},
}


def test_responses_parses_output_envelope() -> None:
    # A Responses reply (output[] with output_text + input_tokens/output_tokens)
    # parses into Response.text + usage — NOT the Chat Completions choices[] path
    # — and the request hits /v1/responses.
    with _MockServer(_RESPONSES_REPLY) as server:
        c = openai("key")
        c.provider.base_url = server.url
        resp = asyncio.run(
            c.text.protocol(Responses).model("gpt-4o-mini").prompt(
                "capital of Finland?"
            )
        )
        assert resp.text == "Helsinki."
        assert resp.usage.input == 16
        assert resp.usage.output == 5
        assert server.received_path.endswith("/v1/responses")


def test_responses_default_unchanged_hits_chat_completions() -> None:
    # WITHOUT protocol(Responses) the same client still POSTs to
    # /v1/chat/completions and parses the choices[] envelope — default pinned.
    with _MockServer(_CHAT_COMPLETIONS_REPLY) as server:
        c = openai("key")
        c.provider.base_url = server.url
        resp = asyncio.run(
            c.text.model("gpt-4o-mini").prompt("capital of Finland?")
        )
        assert resp.text == "Helsinki."
        assert server.received_path.endswith("/v1/chat/completions")


def test_responses_unsupported_provider_errors() -> None:
    # protocol(Responses) on a provider that does not expose it (Anthropic)
    # raises the uniform ValidationError(field="protocol") before any network call.
    c = anthropic("key")
    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(
            c.text.protocol(Responses).model("claude-sonnet-4-6").prompt("hi")
        )
    assert exc_info.value.field == "protocol"


def test_responses_unknown_protocol_errors() -> None:
    # An unknown protocol token raises ValidationError(field="protocol") rather
    # than silently falling back.
    c = openai("key")
    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(c.text.protocol("nonexistent").model("gpt-4o-mini").prompt("hi"))
    assert exc_info.value.field == "protocol"
