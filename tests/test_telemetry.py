"""ADR-054 opt-in telemetry: OTLP builder parity + exporter behaviour.

The two parity tests assert the pure ``build_otlp_traces`` output is
value-identical to the shared cross-SDK goldens at
``codegen/testdata/wire/telemetry/v1/`` (the same goldens every SDK asserts
against). The remaining tests cover the exporter wiring: it POSTs to the
collector, empty endpoints fail loud, and a bad endpoint is fail-open.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from llmkit import openai
from llmkit.errors import ValidationError
from llmkit.providers.generated.middleware import (
    Event,
    MiddlewareOp,
    MiddlewarePhase,
    Usage,
)
from llmkit.telemetry import (
    Telemetry,
    build_otlp_traces,
    http_export,
    make_telemetry_middleware,
    with_telemetry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "codegen" / "testdata" / "wire" / "telemetry" / "v1"
ARTIFACT_ROOT = REPO_ROOT / "target" / "wire" / "telemetry"

# Fixed span identity + timing so the builder output is deterministic and
# value-comparable against the goldens.
_TRACE_ID = "5b8efff798038103d269b633813fc60c"
_SPAN_ID = "eee19b7ec3c1b174"
_START = "1700000000000000000"
_END = "1700000001000000000"


def _assert_telemetry_golden(fixture: str, payload: bytes) -> None:
    obj = json.loads(payload)
    artifact = ARTIFACT_ROOT / fixture / "python.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(obj, indent=2))
    golden = json.loads((GOLDEN_DIR / f"{fixture}.json").read_text())
    assert obj == golden


def test_build_otlp_traces_success_matches_golden() -> None:
    payload = build_otlp_traces(
        "chat", "openai", "gpt-4o", 10, 20, "", _TRACE_ID, _SPAN_ID, _START, _END
    )
    _assert_telemetry_golden("telemetry-success", payload)


def test_build_otlp_traces_rejection_matches_golden() -> None:
    payload = build_otlp_traces(
        "chat",
        "openai",
        "gpt-4o",
        0,
        0,
        "rate_limit_exceeded",
        _TRACE_ID,
        _SPAN_ID,
        _START,
        _END,
    )
    _assert_telemetry_golden("telemetry-rejection", payload)


class _CollectorServer:
    """Single-shot OTLP collector mock: records the request path, headers, and
    decoded JSON body of one export POST."""

    def __init__(self) -> None:
        self.path: str | None = None
        self.headers: dict[str, str] = {}
        self.body: dict | None = None
        # Export is now fire-and-forget on a daemon thread (FU-2); tests wait on
        # this before asserting rather than racing the background POST.
        self.received = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                outer.path = self.path
                outer.headers = {k.lower(): v for k, v in self.headers.items()}
                outer.body = json.loads(raw.decode("utf-8"))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                outer.received.set()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._httpd.shutdown()
        self._httpd.server_close()


class _ProviderServer:
    """Minimal provider mock: returns an empty JSON body (parses to empty text
    and zero usage, no error) so the chat path completes and fires post-phase
    middleware."""

    def __init__(self) -> None:
        outer = self
        self.hit = False

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                outer.hit = True
                payload = b"{}"
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


def _post_event() -> Event:
    return Event(
        op=MiddlewareOp.LLM_REQUEST,
        phase=MiddlewarePhase.POST,
        provider="openai",
        model="gpt-4o",
        usage=Usage(input=10, output=20),
    )


def test_export_callback_invoked_synchronously() -> None:
    # ADR-059: the post phase hands finished OTLP bytes to the callback exactly
    # once, synchronously — no thread is spawned. The pre phase never exports.
    got: list[bytes] = []
    mw = make_telemetry_middleware(Telemetry(export=got.append))

    pre = Event(op=MiddlewareOp.LLM_REQUEST, phase=MiddlewarePhase.PRE, provider="openai", model="gpt-4o")
    assert mw(pre) is None
    assert got == []

    before = threading.active_count()
    for _ in range(50):
        assert mw(_post_event()) is None
    assert threading.active_count() <= before, "export must spawn no thread"
    assert len(got) == 50
    assert isinstance(got[0], bytes)
    assert "resourceSpans" in json.loads(got[0])


def test_export_throwing_callback_fails_open() -> None:
    def boom(_payload: bytes) -> None:
        raise RuntimeError("callback blew up")

    mw = make_telemetry_middleware(Telemetry(export=boom))
    assert mw(_post_event()) is None


def test_http_export_posts_to_collector() -> None:
    with _CollectorServer() as collector:
        mw = make_telemetry_middleware(
            Telemetry(
                export=http_export(collector.url, {"authorization": "Bearer tok-123"})
            )
        )
        assert mw(_post_event()) is None
        assert collector.received.wait(timeout=2.0), "export did not reach collector"
        assert collector.path == "/v1/traces"
        assert collector.headers.get("authorization") == "Bearer tok-123"
        assert collector.body is not None
        assert "resourceSpans" in collector.body


def test_exporter_wired_on_chat_path() -> None:
    # End-to-end: with_telemetry attaches the exporter to the Text builder seam,
    # so a real prompt call emits one span to the collector on the post phase.
    with _ProviderServer() as provider, _CollectorServer() as collector:
        client = openai("key")
        client.provider.base_url = provider.url
        with_telemetry(client, Telemetry(export=http_export(collector.url)))
        asyncio.run(client.text.prompt("hello"))
        assert provider.hit
        assert collector.received.wait(timeout=2.0), "export did not reach collector"
        assert collector.body is not None
        span = collector.body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        attrs = {a["key"]: a["value"] for a in span["attributes"]}
        assert attrs["gen_ai.system"] == {"stringValue": "openai"}


def test_with_telemetry_missing_export_raises() -> None:
    # TEL-017 honest-contract guard: enabled telemetry with no sink fails loud.
    with pytest.raises(ValidationError) as exc:
        with_telemetry(openai("key"), Telemetry(export=None))  # type: ignore[arg-type]
    assert exc.value.field == "telemetry.export"


def test_http_export_fail_open_on_bad_endpoint() -> None:
    # Port 1 is not listening -> connection refused -> swallowed, never raised.
    mw = make_telemetry_middleware(Telemetry(export=http_export("http://127.0.0.1:1")))
    assert mw(_post_event()) is None
