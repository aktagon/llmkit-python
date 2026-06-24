"""Speech generation tests (ADR-049) — mock HTTP server, no live API calls.

Mirror of go/speech_test.go: the Inworld SpeechInworld wire shape (flat-JSON
body, Basic auth with the key sent verbatim, base64 audioContent round-trip)
plus the pre-flight rejections (unknown voice / model, missing voice,
unsupported provider).
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from llmkit import ValidationError
from llmkit.builders import new_client

INWORLD_TTS2 = "inworld-tts-2"

# Distinct bytes so the round-trip assert is real (RIFF/WAVE-ish header).
FAKE_AUDIO = bytes([0x52, 0x49, 0x46, 0x46, 0x01, 0x57, 0x41])


class _MockServer:
    """Single-shot HTTP server that captures one request and serves a canned response."""

    def __init__(self, response_body: dict[str, Any]):
        self.response_body = response_body
        self.received_path = ""
        self.received_body: dict[str, Any] | None = None
        self.received_headers: dict[str, str] = {}
        self.hit = False

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):
                pass

            def do_POST(self):
                outer.hit = True
                outer.received_path = self.path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                outer.received_body = json.loads(raw.decode("utf-8"))
                outer.received_headers = dict(self.headers.items())

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


def test_speech_generate_inworld_round_trips_wav() -> None:
    encoded = base64.b64encode(FAKE_AUDIO).decode("ascii")
    response = {
        "audioContent": encoded,
        "usage": {"processedCharactersCount": 18, "modelId": INWORLD_TTS2},
    }
    with _MockServer(response) as server:
        c = new_client("inworld", "test-token")
        c.provider.base_url = server.url
        resp = asyncio.run(
            c.speech.model(INWORLD_TTS2).voice("Dennis").generate("Hello from llmkit.")
        )

    assert server.received_path == "/tts/v1/voice"
    assert server.received_headers.get("Authorization") == "Basic test-token"
    assert server.received_body == {
        "text": "Hello from llmkit.",
        "voiceId": "Dennis",
        "modelId": INWORLD_TTS2,
        "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 22050},
        "deliveryMode": "BALANCED",
    }
    assert resp.audio.mime_type == "audio/wav"
    assert resp.audio.bytes == FAKE_AUDIO


def test_speech_generate_unknown_voice_rejected_preflight() -> None:
    with _MockServer({}) as server:
        c = new_client("inworld", "test-token")
        c.provider.base_url = server.url
        with pytest.raises(ValidationError) as exc:
            asyncio.run(
                c.speech.model(INWORLD_TTS2).voice("Nonexistent").generate("Hi")
            )
        assert exc.value.field == "voice"
        assert server.hit is False


def test_speech_generate_unknown_model_rejected() -> None:
    c = new_client("inworld", "test-token")
    with pytest.raises(ValidationError) as exc:
        asyncio.run(c.speech.model("inworld-tts-99").voice("Dennis").generate("Hi"))
    assert exc.value.field == "model"


def test_speech_generate_missing_voice_rejected() -> None:
    c = new_client("inworld", "test-token")
    with pytest.raises(ValidationError) as exc:
        asyncio.run(c.speech.model(INWORLD_TTS2).generate("Hi"))
    assert exc.value.field == "voice"


def test_speech_generate_unsupported_provider_rejected() -> None:
    c = new_client("openai", "test-token")
    with pytest.raises(ValidationError) as exc:
        asyncio.run(c.speech.model(INWORLD_TTS2).voice("Dennis").generate("Hi"))
    assert exc.value.field == "provider"
