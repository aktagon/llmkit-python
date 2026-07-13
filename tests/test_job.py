"""Job engine (ADR-062 / ADR-063) tests — mirror of go/job_test.go.

The engine is proven end-to-end by the migrated batch + transcription paths
(test_batch / test_transcription). These cover the new public surface the
migration adds: poll (one normalized round-trip, ADR-063), the batch deadline
backstop + poll_deadline override (ADR-062 OQ-1), and the failed-vs-timeout
distinction (POLL-008).
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

import pytest

from llmkit import JobState, PollTimeoutError, audio
from llmkit.batch import BatchHandle, wait_batch
from llmkit.builders import new_client
from llmkit.errors import APIError
from llmkit.types import Provider

ASSEMBLYAI_AUDIO_URL = "https://storage.example.com/meeting.mp3"
_FAST = {"poll_interval": 0.01}


# ============================ transcription poll ============================


def _completed_transcript() -> dict[str, Any]:
    return {
        "id": "transcript-7c2",
        "status": "completed",
        "text": "The quarterly review is scheduled for Tuesday.",
        "words": [{"text": "The", "start": 120, "end": 280, "speaker": "A"}],
    }


class _AssemblyAIServer:
    """Serves the AssemblyAI submit + poll endpoints. The poll returns
    ``processing`` for the first ``pending_polls`` GET calls, then ``done_body``."""

    def __init__(self, pending_polls: int, done_body: dict[str, Any]) -> None:
        self.pending_polls = pending_polls
        self.done_body = done_body
        self.polls = 0
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
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                if path.endswith("/v2/transcript"):
                    return self._send({"id": "transcript-7c2", "status": "queued"})
                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                path = urlparse(self.path).path
                if "/v2/transcript/transcript-7c2" in path:
                    outer.polls += 1
                    if outer.polls <= outer.pending_polls:
                        return self._send(
                            {"id": "transcript-7c2", "status": "processing"}
                        )
                    return self._send(outer.done_body)
                self.send_response(404)
                self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_AssemblyAIServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def _assemblyai_handle(base_url: str):
    c = new_client("assemblyai", "test-key")
    c.provider.base_url = base_url
    h = asyncio.run(c.transcription.submit([audio(ASSEMBLYAI_AUDIO_URL)]))
    return h


def test_transcription_poll_succeeded() -> None:
    """poll on a completed job returns SUCCEEDED with the result populated inline
    (the result decode is the terminal capability tail) and no failure cause."""
    with _AssemblyAIServer(pending_polls=0, done_body=_completed_transcript()) as server:
        h = _assemblyai_handle(server.url)
        st = asyncio.run(h.poll())
    assert st.state is JobState.SUCCEEDED
    assert st.raw_status == "completed"
    assert st.cause is None
    assert st.result is not None
    assert st.result.text == "The quarterly review is scheduled for Tuesday."


def test_transcription_poll_running() -> None:
    """poll on an in-progress job returns RUNNING with no result and no cause —
    one round-trip, no loop."""
    with _AssemblyAIServer(pending_polls=5, done_body=_completed_transcript()) as server:
        h = _assemblyai_handle(server.url)
        st = asyncio.run(h.poll())
    assert st.state is JobState.RUNNING
    assert st.raw_status == "processing"
    assert st.result is None
    assert st.cause is None


def test_transcription_poll_failed() -> None:
    """poll on a failed job returns FAILED with the provider error message on the
    normalized cause (the same message wait surfaces — S02), and no result."""
    failed = {
        "id": "transcript-7c2",
        "status": "error",
        "error": "Download error, unable to download https://storage.example.com/meeting.mp3",
    }
    with _AssemblyAIServer(pending_polls=0, done_body=failed) as server:
        h = _assemblyai_handle(server.url)
        st = asyncio.run(h.poll())
    assert st.state is JobState.FAILED
    assert st.result is None
    assert st.cause is not None
    assert st.cause.status == "error"
    assert "Download error" in st.cause.message
    assert st.cause.timed_out is False


def test_transcription_wait_failed_error_message() -> None:
    """The wait path (not just poll) formats a failed job as
    "<noun> failed: <provider message>" — and it is NOT the timeout sentinel."""
    failed = {
        "id": "transcript-7c2",
        "status": "error",
        "error": "Download error, unable to download the source audio",
    }
    with _AssemblyAIServer(pending_polls=0, done_body=failed) as server:
        h = _assemblyai_handle(server.url)
        with pytest.raises(APIError) as exc:
            asyncio.run(h.wait(**_FAST))
    msg = exc.value.message
    assert msg.startswith("transcription failed: ")
    assert "Download error" in msg
    # POLL-008: a provider-reported failure is NOT the timeout sentinel.
    assert not isinstance(exc.value, PollTimeoutError)


# ================================ batch poll ================================


class _BatchPollServer:
    """Serves the OpenAI batch poll endpoint with a fixed status — enough for the
    RUNNING + FAILED + deadline paths (no result hop needed)."""

    def __init__(self, status: str) -> None:
        self.status = status
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def do_GET(self):
                path = urlparse(self.path).path
                if path.startswith("/v1/batches/"):
                    body = json.dumps(
                        {"id": "batch_1", "status": outer.status}
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(404)
                self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_BatchPollServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def _openai_batch_handle(base_url: str) -> BatchHandle:
    return BatchHandle(
        id="batch_1",
        provider=Provider(name="openai", api_key="test-key", base_url=base_url),
    )


def test_batch_poll_running() -> None:
    """BatchHandle.poll on an in-progress batch returns RUNNING without attempting
    the two-hop result fetch."""
    from llmkit.builders.batch import BatchHandle as TypedHandle

    with _BatchPollServer("in_progress") as server:
        h = TypedHandle(id="batch_1", provider=_openai_batch_handle(server.url).provider)
        st = asyncio.run(h.poll())
    assert st.state is JobState.RUNNING
    assert st.raw_status == "in_progress"
    assert st.result is None


def test_batch_poll_failed() -> None:
    """A batch the provider reports as terminally failed (OpenAI status "failed",
    carried by polling_error_values) classifies as FAILED on the FIRST poll — it
    does not hang to the deadline backstop."""
    from llmkit.builders.batch import BatchHandle as TypedHandle

    with _BatchPollServer("failed") as server:
        h = TypedHandle(id="batch_1", provider=_openai_batch_handle(server.url).provider)
        st = asyncio.run(h.poll())
    assert st.state is JobState.FAILED
    assert st.result is None
    assert st.cause is not None
    assert st.cause.status == "failed"
    assert st.cause.timed_out is False


def test_batch_wait_failed_error() -> None:
    """wait_batch on a failed batch returns a provider-failure error (not the
    timeout sentinel) — the deadline backstop is never reached."""
    with _BatchPollServer("expired") as server:
        handle = _openai_batch_handle(server.url)
        with pytest.raises(APIError) as exc:
            wait_batch(handle, poll_interval=0.001, poll_deadline=3600.0)
    assert exc.value.message.startswith("batch failed: ")
    assert not isinstance(exc.value, PollTimeoutError)


def test_batch_wait_times_out_at_backstop() -> None:
    """A batch that never completes must terminate at the deadline backstop rather
    than loop forever (ADR-062 OQ-1) — as a distinguishable PollTimeoutError."""
    with _BatchPollServer("in_progress") as server:
        handle = _openai_batch_handle(server.url)
        with pytest.raises(PollTimeoutError):
            wait_batch(handle, poll_interval=0.005, poll_deadline=0.02)


def test_batch_wait_async_cancellable() -> None:
    """The async wait loop honors cancellation — an asyncio.CancelledError raised
    while the loop awaits its cancellable sleep propagates (S06)."""
    from llmkit.builders.batch import BatchHandle as TypedHandle

    async def run() -> None:
        with _BatchPollServer("in_progress") as server:
            h = TypedHandle(
                id="batch_1", provider=_openai_batch_handle(server.url).provider
            )
            task = asyncio.ensure_future(h.wait(poll_interval=0.05, poll_deadline=3600.0))
            await asyncio.sleep(0.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
