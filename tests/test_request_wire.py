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
from llmkit import anthropic
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
    """Single-shot mock that records the outbound POST body."""

    def __init__(self, response_body: dict[str, Any]):
        self.last_body: dict[str, Any] | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if raw:
                    outer.last_body = json.loads(raw.decode("utf-8"))
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


def test_structured_output_google_matches_shared_golden() -> None:
    body, _ = _build_request(
        llmkit.Provider(name="google", api_key="AIza-test"),
        llmkit.Request(
            user="What color is a clear daytime sky?",
            schema=(
                '{"type":"object","properties":{"color":{"type":"string"}},'
                '"required":["color"],"additionalProperties":false}'
            ),
        ),
        llmkit.Options(),
        PROVIDERS["google"],
    )
    _assert_wire_golden("structured-output-google", body)


# Response shape valid for the text, agent, and batch-submit paths (id is the
# batch-create handle).
_ANTHROPIC_RESP = {
    "id": "msgbatch_test",
    "content": [{"type": "text", "text": "done"}],
    "usage": {"input_tokens": 2000, "output_tokens": 5},
}


def test_caching_agent_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_ANTHROPIC_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.agent.system("a long stable system prefix").caching().prompt("hi")
        )
        assert server.last_body is not None
        _assert_wire_golden("caching-agent-anthropic", server.last_body)


def test_caching_text_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_ANTHROPIC_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.system("a long stable system prefix").caching().prompt("hi")
        )
        assert server.last_body is not None
        _assert_wire_golden("caching-text-anthropic", server.last_body)


def test_caching_batch_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_ANTHROPIC_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.system("a long stable system prefix").caching().submit_batch("hi")
        )
        assert server.last_body is not None
        _assert_wire_golden("caching-batch-anthropic", server.last_body)
