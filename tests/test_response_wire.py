"""Cross-SDK RESPONSE-body conformance driver — Python (ADR-065 / prompt 045).

Sibling of test_lifecycle_wire.py, on the response-BODY side. Where the lifecycle
suite asserts the poll CLASSIFICATION agrees across SDKs, this asserts the body
PARSE agrees: given the same anchored provider reply, every SDK's public
c.text.prompt() normalizes it to the SAME projection (Usage dims + finish reason
+ content). Response parsing is handwritten per SDK (ADR-028 behavior, not
generated data); this is its parity floor.

The parser INPUT lives at codegen/testdata/wire/response/v1/bodies/<shape>.json;
this driver serves it verbatim from a single-hop mock, drives one prompt, projects
the parsed Response, drops target/wire/response/<shape>/python.json, and asserts it
value-equals the EXPECTED golden codegen/testdata/wire/response/v1/<shape>.json.
codegen/test_cross_sdk_response.py compares all four SDK artifacts to that golden.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from llmkit.builders import new_client
from llmkit.structs import Response

REPO_ROOT = Path(__file__).resolve().parents[2]
BODY_DIR = REPO_ROOT / "codegen" / "testdata" / "wire" / "response" / "v1" / "bodies"
GOLDEN_DIR = REPO_ROOT / "codegen" / "testdata" / "wire" / "response" / "v1"
ARTIFACT_ROOT = REPO_ROOT / "target" / "wire" / "response"


class _ResponseMockServer:
    """Serves the anchored provider reply verbatim on any method/path — the parse
    path is single-hop, so a catch-all is enough. The parser dispatches on the
    client's provider, not the URL."""

    def __init__(self, body: bytes) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # chat requests are POST
                self._send()

            def do_GET(self):
                self._send()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_ResponseMockServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def _artifact_from(resp: Response) -> dict:
    """Normalized, cross-SDK-comparable projection of a parsed Response — the
    contract-bearing parse output only (Usage dims + finish reason + content)."""
    u = resp.usage
    return {
        "usage": {
            "input": u.input,
            "output": u.output,
            "cacheRead": u.cache_read,
            "cacheWrite": u.cache_write,
            "reasoning": u.reasoning,
            "cost": u.cost,
        },
        "finishReason": resp.finish_reason,
        "content": resp.text,
        "error": None,
    }


def _run_fixture(shape: str, provider: str) -> None:
    body = (BODY_DIR / f"{shape}.json").read_bytes()
    with _ResponseMockServer(body) as server:
        c = new_client(provider, "k")
        c.provider.base_url = server.url
        resp = asyncio.run(c.text.prompt("ping"))

    artifact = _artifact_from(resp)
    out_dir = ARTIFACT_ROOT / shape
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "python.json").write_text(json.dumps(artifact, indent=2))

    golden = json.loads((GOLDEN_DIR / f"{shape}.json").read_text())
    assert artifact == golden


def test_response_chat_openai() -> None:
    _run_fixture("chat-openai", "openai")


def test_response_chat_anthropic() -> None:
    _run_fixture("chat-anthropic", "anthropic")


def test_response_chat_google() -> None:
    _run_fixture("chat-google", "google")
