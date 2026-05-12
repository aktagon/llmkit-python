"""Unit tests for llmkit.batch (prompt_batch / submit_batch / wait_batch)
and the typed-builder Text.batch / text_batch wire. Uses a stateful
Anthropic mock that handles the three-endpoint batch lifecycle:

  POST /v1/messages/batches           → create
  GET  /v1/messages/batches/{id}      → poll status
  GET  /v1/messages/batches/{id}/results → JSONL results

Anthropic chosen over OpenAI because its lifecycle is single-shape
(no two-hop file_id retrieval) — the bookkeeping is in one wire shape,
which makes the cross-symbol coverage clear."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

from llmkit.batch import BatchHandle, prompt_batch, submit_batch, wait_batch
from llmkit.builders import new_client
from llmkit.types import Provider, Request


# ---------- stateful Anthropic batch mock ----------


class _AnthropicBatchServer:
    """Mimics the Anthropic batch lifecycle. ``status_progression`` controls
    how many polls return `in_progress` before flipping to `ended`."""

    def __init__(
        self,
        batch_id: str,
        result_lines: list[dict[str, Any]],
        polls_before_ended: int = 0,
    ) -> None:
        self.batch_id = batch_id
        self.result_lines = result_lines
        self.polls_before_ended = polls_before_ended
        self.poll_count = 0
        self.create_body: dict[str, Any] | None = None
        self.create_headers: dict[str, str] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self, payload: bytes, status: int = 200, ctype: str = "application/json"):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if path == "/v1/messages/batches":
                    outer.create_body = json.loads(raw.decode("utf-8"))
                    outer.create_headers = {k.lower(): v for k, v in self.headers.items()}
                    body = {
                        "id": outer.batch_id,
                        "type": "message_batch",
                        "processing_status": "in_progress",
                    }
                    return self._send(json.dumps(body).encode("utf-8"))
                self._send(b'{"error":"unexpected POST"}', status=404)

            def do_GET(self):
                path = urlparse(self.path).path
                if path == f"/v1/messages/batches/{outer.batch_id}":
                    outer.poll_count += 1
                    status = (
                        "ended"
                        if outer.poll_count > outer.polls_before_ended
                        else "in_progress"
                    )
                    body = {
                        "id": outer.batch_id,
                        "processing_status": status,
                    }
                    return self._send(json.dumps(body).encode("utf-8"))
                if path == f"/v1/messages/batches/{outer.batch_id}/results":
                    lines = "\n".join(json.dumps(line) for line in outer.result_lines)
                    return self._send(lines.encode("utf-8"), ctype="application/x-ndjson")
                self._send(b'{"error":"not found"}', status=404)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

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


def _anthropic_result_line(custom_id: str, text: str, input_tokens: int = 5, output_tokens: int = 7) -> dict[str, Any]:
    """An Anthropic batch result wraps the message at result.message."""
    return {
        "custom_id": custom_id,
        "result": {
            "type": "succeeded",
            "message": {
                "id": "msg_" + custom_id,
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            },
        },
    }


def _anthropic_provider(base_url: str) -> Provider:
    return Provider(
        name="anthropic",
        api_key="test-key",
        model="claude-sonnet-4-6",
        base_url=base_url,
    )


# =============================================================================
# Free-function entry points (legacy surface — still exported, still tested)
# =============================================================================


def test_submit_batch_returns_handle_with_batch_id_and_provider() -> None:
    """submit_batch posts the create body and returns the parsed handle.
    Does not poll. Wire-body must carry the {custom_id, params: {...}}
    envelope under the `requests` key — the `params` key is load-bearing
    for Anthropic (the message body lives inside it)."""

    with _AnthropicBatchServer(batch_id="batch_anthro_1", result_lines=[]) as server:
        provider = _anthropic_provider(server.url)
        requests = [
            Request(user="What is 2+2?"),
            Request(user="What is 3+3?"),
        ]
        handle = submit_batch(provider, requests)

    assert isinstance(handle, BatchHandle)
    assert handle.id == "batch_anthro_1"
    assert handle.provider.name == "anthropic"

    # Verify the create body envelope.
    assert server.create_body is not None
    assert "requests" in server.create_body
    items = server.create_body["requests"]
    assert len(items) == 2
    # Each item: {"custom_id": "req-N", "params": {body...}}.
    assert items[0]["custom_id"] == "req-0"
    assert items[1]["custom_id"] == "req-1"
    assert "params" in items[0] and "params" in items[1]
    # The inner params body carries the message content.
    inner = items[0]["params"]
    assert inner["model"] == "claude-sonnet-4-6"
    assert any("2+2" in (m.get("content") or "") for m in inner.get("messages", []))


def test_wait_batch_polls_then_fetches_results_and_parses_via_result_body_path() -> None:
    """wait_batch polls until status=ended, fetches results, and unwraps
    each JSONL line via result.message (Anthropic's result_body_path)."""

    results = [
        _anthropic_result_line("req-0", "four"),
        _anthropic_result_line("req-1", "six", input_tokens=8, output_tokens=4),
    ]
    with _AnthropicBatchServer(batch_id="batch_wait_1", result_lines=results, polls_before_ended=1) as server:
        provider = _anthropic_provider(server.url)
        handle = BatchHandle(id="batch_wait_1", provider=provider)
        responses = wait_batch(handle, poll_interval=0.01)

    # Polled twice (1 in_progress + 1 ended).
    assert server.poll_count == 2
    assert len(responses) == 2
    assert responses[0].text == "four"
    assert responses[1].text == "six"
    # Usage extracted from the unwrapped message body, not the wrapper.
    assert responses[0].tokens.input == 5
    assert responses[0].tokens.output == 7
    assert responses[1].tokens.input == 8


def test_prompt_batch_end_to_end_submit_then_wait() -> None:
    """prompt_batch is the convenience wrapper: submits + waits."""

    results = [_anthropic_result_line("req-0", "the answer is 42")]
    with _AnthropicBatchServer(batch_id="batch_e2e", result_lines=results) as server:
        provider = _anthropic_provider(server.url)
        responses = prompt_batch(
            provider,
            [Request(user="answer me")],
            poll_interval=0.01,
        )

    assert len(responses) == 1
    assert responses[0].text == "the answer is 42"


# =============================================================================
# Typed-builder wire (Text.batch → text_batch → legacy prompt_batch)
# =============================================================================


def test_text_batch_through_typed_builder_round_trips_two_prompts() -> None:
    """Closes the Text.batch + text_batch coverage warnings. Builder's
    system + per-prompt user content must flow through the wire body."""

    # Known gap: legacy ``prompt_batch`` doesn't accept Options, so the
    # typed-builder's ``.max_tokens(...) / .temperature(...) / ...`` chain
    # state is silently dropped on the batch path. (Different bug class from
    # the ADR-011 silent-drops fixed in plan 018 D-stage — those were
    # text/agent helpers forgetting to read fields; this one is the
    # underlying free-function API missing the parameter altogether. Tracked
    # as a v1.1.0 follow-up.) The test asserts what *is* propagated.
    results = [
        _anthropic_result_line("req-0", "alpha"),
        _anthropic_result_line("req-1", "beta"),
    ]
    with _AnthropicBatchServer(batch_id="batch_typed", result_lines=results) as server:
        c = new_client("anthropic", "test-key")
        c.provider.base_url = server.url
        responses = asyncio.run(
            c.text.model("claude-sonnet-4-6")
            .system("You are terse.")
            .batch("first prompt", "second prompt")
        )

    assert [r.text for r in responses] == ["alpha", "beta"]

    # Wire body: the typed-builder's system reaches each request's params.
    assert server.create_body is not None
    items = server.create_body["requests"]
    inner_0 = items[0]["params"]
    assert inner_0["system"] == "You are terse."
    # User content carries the prompt.
    msgs_0 = inner_0.get("messages", [])
    assert any("first prompt" in (m.get("content") or "") for m in msgs_0)
    assert any(
        "second prompt" in (m.get("content") or "")
        for m in items[1]["params"].get("messages", [])
    )


def test_text_submit_batch_through_typed_builder_returns_typed_batch_handle() -> None:
    """Text.submit_batch returns a builders.batch.BatchHandle (typed-builder
    class) that exposes ``.wait()`` — distinct from the legacy free-function
    BatchHandle dataclass."""

    from llmkit.builders.batch import BatchHandle as TypedHandle

    with _AnthropicBatchServer(batch_id="batch_submit_typed", result_lines=[]) as server:
        c = new_client("anthropic", "test-key")
        c.provider.base_url = server.url
        handle = asyncio.run(c.text.model("claude-sonnet-4-6").submit_batch("hello"))

    assert isinstance(handle, TypedHandle)
    assert handle.id == "batch_submit_typed"
    assert hasattr(handle, "wait")
