"""Spike 036 (PIVOT wire-conformance): request-byte conformance, generalized
across capabilities (structured output, agent-path caching).

Asserts the OUTBOUND request body each SDK produces is value-equal to the shared
golden at codegen/testdata/wire/request/v1/<fixture>.json — the SAME golden
every SDK asserts against. These are the wires BUG-007 (Python malformed Google
body) and BUG-004 (agent-path caching dropped) broke. No API keys.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import llmkit
from llmkit import anthropic, openai
from llmkit.client import _build_request
from llmkit.providers.generated.providers import PROVIDERS

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "codegen" / "testdata" / "wire" / "request" / "v1"
ARTIFACT_ROOT = REPO_ROOT / "target" / "wire" / "request"


def _assert_wire_golden(fixture: str, body: dict[str, Any]) -> None:
    artifact = ARTIFACT_ROOT / fixture / "python.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(body, indent=2))
    golden = json.loads((GOLDEN_DIR / f"{fixture}.json").read_text())
    assert body == golden


class _CaptureServer:
    """Single-shot mock that records the outbound POST body and headers
    (headers feed the in-driver asserts for load-bearing headers, e.g.
    Anthropic's structured-output beta header)."""

    def __init__(self, response_body: dict[str, Any]):
        self.last_body: dict[str, Any] | None = None
        self.last_headers: dict[str, str] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if raw:
                    outer.last_body = json.loads(raw.decode("utf-8"))
                    outer.last_headers = {k.lower(): v for k, v in self.headers.items()}
                payload = json.dumps(response_body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._httpd.shutdown()
        self._httpd.server_close()


# Omits "required" so the goldens witness EnforceStrict normalization
# (auto-required); carries additionalProperties:false so Google's strip is
# witnessed too. See the Go driver comment (the minting reference).
_CANONICAL_SCHEMA = (
    '{"type":"object","properties":{"color":{"type":"string"}},'
    '"additionalProperties":false}'
)
_CANONICAL_PROMPT = "What color is a clear daytime sky?"


# Response shape valid for the text, agent, and batch-submit paths across
# providers (id is the batch-create handle; missing provider paths parse to
# empty text / zero usage, which the drivers never assert).
_CANNED_RESP = {
    "id": "msgbatch_test",
    "content": [{"type": "text", "text": "done"}],
    "usage": {"input_tokens": 2000, "output_tokens": 5},
}


def test_structured_output_google_matches_shared_golden() -> None:
    body, _ = _build_request(
        llmkit.Provider(name="google", api_key="AIza-test"),
        llmkit.Request(user=_CANONICAL_PROMPT, schema=_CANONICAL_SCHEMA),
        llmkit.Options(),
        PROVIDERS["google"],
    )
    _assert_wire_golden("structured-output-google", body)


def test_structured_output_openai_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(c.text.schema(_CANONICAL_SCHEMA).prompt(_CANONICAL_PROMPT))
        assert server.last_body is not None
        _assert_wire_golden("structured-output-openai", server.last_body)


def test_structured_output_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(c.text.schema(_CANONICAL_SCHEMA).prompt(_CANONICAL_PROMPT))
        assert server.last_body is not None
        # ADR-028 Open Questions: load-bearing headers assert in-driver.
        # Without this beta header Anthropic rejects output_format with a 400.
        assert (
            server.last_headers.get("anthropic-beta")
            == "structured-outputs-2025-11-13"
        )
        _assert_wire_golden("structured-output-anthropic", server.last_body)


def test_caching_agent_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.agent.system("a long stable system prefix").caching().prompt("hi")
        )
        assert server.last_body is not None
        _assert_wire_golden("caching-agent-anthropic", server.last_body)


def test_caching_text_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.system("a long stable system prefix").caching().prompt("hi")
        )
        assert server.last_body is not None
        _assert_wire_golden("caching-text-anthropic", server.last_body)


def test_caching_batch_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.system("a long stable system prefix").caching().submit_batch("hi")
        )
        assert server.last_body is not None
        _assert_wire_golden("caching-batch-anthropic", server.last_body)
