"""Transcription (speech-to-text) tests (ADR-048) — mock HTTP server, no live
API calls.

Mirror of go/transcription_test.go. Transcription is asynchronous: submit
returns a handle, then handle.wait() polls until terminal. Slice 1 wires the
AssemblyAI wire shape only: optional upload hop -> {audio_url} submit ->
poll -> {text, words[]}.

The mock server returns `processing` for the first N polls, then the supplied
done body. Each wait() call passes a small poll_interval (mirroring
test_video.py) so tests run fast.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

import pytest

from llmkit import ValidationError, audio, audio_bytes
from llmkit.builders import new_client
from llmkit.builders.transcription import TranscriptionHandle
from llmkit.errors import APIError

ASSEMBLYAI_AUDIO_URL = "https://storage.example.com/meeting-2026-06-24.mp3"

_FAST = {"poll_interval": 0.01}


def _completed_transcript() -> dict[str, Any]:
    """AssemblyAI transcript object on terminal success: the full text plus
    word-level timing (start/end in milliseconds), with a diarized speaker
    label on the first word only."""
    return {
        "id": "transcript-7c2",
        "status": "completed",
        "text": "The quarterly review is scheduled for Tuesday.",
        "words": [
            {"text": "The", "start": 120, "end": 280, "speaker": "A"},
            {"text": "quarterly", "start": 280, "end": 760},
            {"text": "review", "start": 760, "end": 1100},
        ],
    }


class _AssemblyAIServer:
    """Serves the AssemblyAI upload + submit + poll endpoints. The poll returns
    `processing` for the first ``pending_polls`` GET calls, then the supplied
    done body. ``upload_url``, when non-empty, is returned from POST /v2/upload."""

    def __init__(
        self,
        pending_polls: int,
        done_body: dict[str, Any],
        upload_url: str = "",
    ) -> None:
        self.pending_polls = pending_polls
        self.done_body = done_body
        self.upload_url = upload_url
        self.polls = 0
        self.auth = ""
        self.upload_content_type = ""
        self.upload_byte_len = 0
        self.submit_body: dict[str, Any] | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):  # silence noise
                pass

            def _send(self, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                outer.auth = self.headers.get("Authorization", "")
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if path.endswith("/v2/upload"):
                    outer.upload_content_type = self.headers.get("Content-Type", "")
                    outer.upload_byte_len = len(raw)
                    return self._send({"upload_url": outer.upload_url})
                if path.endswith("/v2/transcript"):
                    outer.submit_body = json.loads(raw.decode("utf-8"))
                    return self._send({"id": "transcript-7c2", "status": "queued"})
                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                outer.auth = self.headers.get("Authorization", "")
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


# ===== submit + wait (processing -> completed) =====


def test_transcription_submit_and_wait_assemblyai() -> None:
    with _AssemblyAIServer(pending_polls=2, done_body=_completed_transcript()) as server:
        c = new_client("assemblyai", "test-key")
        c.provider.base_url = server.url
        h = asyncio.run(c.transcription.submit([audio(ASSEMBLYAI_AUDIO_URL)]))
        assert isinstance(h, TranscriptionHandle)
        assert h.id == "transcript-7c2"

        resp = asyncio.run(h.wait(**_FAST))

    # AssemblyAI auth: the raw key with no Bearer prefix (HeaderAPIKey).
    assert server.auth == "test-key"
    assert server.submit_body == {"audio_url": ASSEMBLYAI_AUDIO_URL}
    assert resp.text == "The quarterly review is scheduled for Tuesday."
    assert len(resp.segments) == 3
    assert resp.segments[0].text == "The"
    assert resp.segments[0].start == 120
    assert resp.segments[0].end == 280
    assert resp.segments[0].speaker == "A"
    assert resp.segments[1].speaker == ""
    assert resp.usage.input == 0


def test_transcription_audio_bytes_upload_hop() -> None:
    uploaded = "https://cdn.assemblyai.com/upload/abc123"
    wav = b"RIFF....WAVEfmt fake-audio-bytes"
    with _AssemblyAIServer(
        pending_polls=1, done_body=_completed_transcript(), upload_url=uploaded
    ) as server:
        c = new_client("assemblyai", "test-key")
        c.provider.base_url = server.url
        h = asyncio.run(c.transcription.submit([audio_bytes("audio/wav", wav)]))
        resp = asyncio.run(h.wait(**_FAST))

    assert server.upload_content_type == "application/octet-stream"
    assert server.upload_byte_len == len(wav)
    assert server.submit_body == {"audio_url": uploaded}
    assert resp.text == "The quarterly review is scheduled for Tuesday."


def test_transcription_error_status_surfaces_as_error() -> None:
    failed = {
        "id": "transcript-7c2",
        "status": "error",
        "error": "Download error, unable to download "
        + ASSEMBLYAI_AUDIO_URL,
    }
    with _AssemblyAIServer(pending_polls=1, done_body=failed) as server:
        c = new_client("assemblyai", "test-key")
        c.provider.base_url = server.url
        h = asyncio.run(c.transcription.submit([audio(ASSEMBLYAI_AUDIO_URL)]))
        with pytest.raises(APIError) as exc:
            asyncio.run(h.wait(**_FAST))
        assert "Download error" in exc.value.message


def test_transcription_rejects_non_audio_part() -> None:
    from llmkit import Part

    c = new_client("assemblyai", "test-key")
    with pytest.raises(ValidationError) as exc:
        asyncio.run(c.transcription.submit([Part(text="transcribe this please")]))
    assert "only audio parts" in exc.value.message


def test_transcription_requires_exactly_one_audio_part() -> None:
    c = new_client("assemblyai", "test-key")
    with pytest.raises(ValidationError) as exc:
        asyncio.run(
            c.transcription.submit(
                [
                    audio(ASSEMBLYAI_AUDIO_URL),
                    audio("https://storage.example.com/other.mp3"),
                ]
            )
        )
    assert "exactly one audio part" in exc.value.message

    with pytest.raises(ValidationError):
        asyncio.run(c.transcription.submit([]))


def test_transcription_unsupported_provider_rejected() -> None:
    c = new_client("openai", "test-key")
    with pytest.raises(ValidationError) as exc:
        asyncio.run(c.transcription.submit([audio(ASSEMBLYAI_AUDIO_URL)]))
    assert "does not support transcription" in exc.value.message
