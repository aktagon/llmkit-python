"""










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
    """

"""
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
    """

"""

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


#


def test_transcription_submit_and_wait_assemblyai() -> None:
    with _AssemblyAIServer(pending_polls=2, done_body=_completed_transcript()) as server:
        c = new_client("assemblyai", "test-key")
        c.provider.base_url = server.url
        h = asyncio.run(c.transcription.submit([audio(ASSEMBLYAI_AUDIO_URL)]))
        assert isinstance(h, TranscriptionHandle)
        assert h.id == "transcript-7c2"

        resp = asyncio.run(h.wait(**_FAST))

    #
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
    #
    c = new_client("anthropic", "test-key")
    with pytest.raises(ValidationError) as exc:
        asyncio.run(c.transcription.submit([audio(ASSEMBLYAI_AUDIO_URL)]))
    assert "does not support transcription" in exc.value.message


#

import email  # noqa: E402

FAKE_MP3 = bytes([0xFF, 0xFB, 0x90, 0x00, 0x6D, 0x70, 0x33])

OPENAI_VERBOSE = {
    "text": "The quarterly review is scheduled for Tuesday.",
    "segments": [
        {"start": 0.0, "end": 1.5, "text": "The quarterly review"},
        {"start": 1.5, "end": 2.84, "text": " is scheduled for Tuesday."},
    ],
}


class _OpenAITranscriptionServer:
    """
"""

    def __init__(self, resp_body: dict[str, Any]) -> None:
        self.resp_body = resp_body
        self.path = ""
        self.auth = ""
        self.fields: dict[str, str] = {}
        self.file_name = ""
        self.file_ctype = ""
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def do_POST(self):
                outer.path = urlparse(self.path).path
                outer.auth = self.headers.get("Authorization", "")
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                ctype = self.headers.get("Content-Type", "")
                msg = email.message_from_bytes(
                    b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw
                )
                for part in msg.get_payload():
                    name = part.get_param("name", header="content-disposition")
                    fn = part.get_filename()
                    if fn:
                        outer.file_name = fn
                        outer.file_ctype = part.get_content_type()
                    else:
                        outer.fields[name] = part.get_payload(decode=True).decode()
                body = json.dumps(outer.resp_body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def test_transcribe_sync_openai_segments_sec_to_ms() -> None:
    with _OpenAITranscriptionServer(OPENAI_VERBOSE) as server:
        c = new_client("openai", "test-key")
        c.provider.base_url = server.url
        resp = asyncio.run(
            c.transcription.model("whisper-1").transcribe(
                [audio_bytes("audio/mpeg", FAKE_MP3)]
            )
        )
    assert server.path == "/v1/audio/transcriptions"
    assert server.auth == "Bearer test-key"
    assert server.fields["model"] == "whisper-1"
    assert server.fields["response_format"] == "verbose_json"
    assert server.file_name == "audio.mp3"
    assert server.file_ctype == "audio/mpeg"
    assert resp.text == "The quarterly review is scheduled for Tuesday."
    assert len(resp.segments) == 2
    assert resp.segments[0].end == 1500
    assert resp.segments[1].end == 2840


def test_transcribe_openai_empty_segments() -> None:
    with _OpenAITranscriptionServer({"text": "Hello there."}) as server:
        c = new_client("openai", "test-key")
        c.provider.base_url = server.url
        resp = asyncio.run(
            c.transcription.model("whisper-1").transcribe(
                [audio_bytes("audio/mpeg", FAKE_MP3)]
            )
        )
    assert resp.text == "Hello there."
    assert resp.segments == []


def test_submit_on_sync_provider_rejected() -> None:
    c = new_client("openai", "test-key")
    with pytest.raises(ValidationError) as exc:
        asyncio.run(
            c.transcription.model("whisper-1").submit(
                [audio_bytes("audio/mpeg", FAKE_MP3)]
            )
        )
    assert exc.value.field == "interaction"
    assert "Transcribe" in exc.value.message


def test_transcribe_on_async_provider_rejected() -> None:
    c = new_client("assemblyai", "test-key")
    with pytest.raises(ValidationError) as exc:
        asyncio.run(
            c.transcription.model("best").transcribe(
                [audio_bytes("audio/mpeg", FAKE_MP3)]
            )
        )
    assert exc.value.field == "interaction"
    assert "Submit/Wait" in exc.value.message


def test_transcribe_rejects_audio_url() -> None:
    c = new_client("openai", "test-key")
    with pytest.raises(ValidationError) as exc:
        asyncio.run(
            c.transcription.model("whisper-1").transcribe([audio(ASSEMBLYAI_AUDIO_URL)])
        )
    assert exc.value.field.startswith("parts")


def test_transcribe_requires_model() -> None:
    c = new_client("openai", "test-key")
    with pytest.raises(ValidationError) as exc:
        asyncio.run(
            c.transcription.transcribe([audio_bytes("audio/mpeg", FAKE_MP3)])
        )
    assert exc.value.field == "model"
