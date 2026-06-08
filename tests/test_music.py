"""Music generation tests (ADR-033) — mock HTTP server, no live API calls.

Mirror of test_image.py. Covers all three wire shapes:
  - Vertex Lyria 2  (MusicPredict)       — base64 predictions[], instrumental
  - Google Lyria 3  (MusicGenerateContent) — base64 inlineData, lyrics inline
  - MiniMax 2.6     (MusicMinimax)        — hex data.audio, absolute URL

MiniMax's gen_endpoint is an absolute https URL, so its end-to-end test
monkeypatches llmkit.music.do_post rather than overriding base_url.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

import llmkit.music as music_mod
from llmkit import MiddlewareVetoError, ValidationError
from llmkit.builders import new_client

VERTEX_LYRIA = "lyria-002"
GOOGLE_LYRIA_PRO = "lyria-3-pro-preview"
GOOGLE_LYRIA_CLIP = "lyria-3-clip-preview"
MINIMAX_MUSIC = "music-2.6"

# Distinct bytes so round-trip asserts are real (RIFF-ish header).
FAKE_AUDIO = bytes([0x52, 0x49, 0x46, 0x46, 0x01, 0x02, 0x03])


class _MockServer:
    """Single-shot HTTP server that captures one request and serves a canned response."""

    def __init__(self, response_body: dict[str, Any]):
        self.response_body = response_body
        self.received_path = ""
        self.received_query: dict[str, list[str]] = {}
        self.received_body: dict[str, Any] | None = None
        self.received_headers: dict[str, str] = {}
        self.hit = False

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):  # silence noise
                pass

            def do_POST(self):
                outer.hit = True
                parsed = urlparse(self.path)
                outer.received_path = parsed.path
                outer.received_query = parse_qs(parsed.query)
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


# ===== Vertex Lyria 2 (MusicPredict wire shape) =====


def _vertex_music_response(b64: str, mime: str = "audio/wav") -> dict[str, Any]:
    return {"predictions": [{"audioContent": b64, "mimeType": mime}]}


def test_music_generate_vertex_predict_round_trips_wav() -> None:
    encoded = base64.b64encode(FAKE_AUDIO).decode("ascii")
    with _MockServer(_vertex_music_response(encoded)) as server:
        c = new_client("vertex", "test-token")
        c.provider.base_url = server.url
        resp = asyncio.run(
            c.music.model(VERTEX_LYRIA).generate("Ambient piano, slow tempo")
        )

    assert server.received_path == f"/{VERTEX_LYRIA}:predict"
    assert server.received_headers.get("Authorization") == "Bearer test-token"
    body = server.received_body or {}
    assert body["instances"] == [{"prompt": "Ambient piano, slow tempo"}]
    assert body["parameters"] == {"sampleCount": 1}

    assert len(resp.audio) == 1
    assert resp.audio[0].mime_type == "audio/wav"
    assert resp.audio[0].bytes == FAKE_AUDIO
    assert resp.text == ""
    assert resp.usage.output == 0


def test_music_generate_raw_opt_in_populates_raw() -> None:
    encoded = base64.b64encode(FAKE_AUDIO).decode("ascii")
    with _MockServer(_vertex_music_response(encoded)) as server:
        c = new_client("vertex", "test-token")
        c.provider.base_url = server.url
        resp = asyncio.run(
            c.music.model(VERTEX_LYRIA).raw().generate("Ambient piano, slow tempo")
        )

    assert resp.raw is not None
    assert resp.raw["predictions"][0]["audioContent"] == encoded


def test_music_generate_vertex_surfaces_rai_filtered_reason() -> None:
    response = {"predictions": [{"raiFilteredReason": "Audio filtered by safety system"}]}
    with _MockServer(response) as server:
        c = new_client("vertex", "test-token")
        c.provider.base_url = server.url
        resp = asyncio.run(c.music.model(VERTEX_LYRIA).generate("blocked"))
    assert resp.audio == []
    assert resp.finish_reason == "Audio filtered by safety system"


def test_music_generate_vertex_rejects_lyrics_instrumental_only() -> None:
    c = new_client("vertex", "test-token")
    c.provider.base_url = "http://unused"
    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(
            c.music.model(VERTEX_LYRIA).lyrics("la la la").generate("a melody")
        )
    assert exc_info.value.field == "parts"
    assert "instrumental-only" in exc_info.value.message


# ===== Google Lyria 3 (MusicGenerateContent wire shape) =====


def _gemini_music_response(
    b64: str, mime: str = "audio/mpeg", text: str | None = None
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"inlineData": {"mimeType": mime, "data": b64}}]
    if text is not None:
        parts.insert(0, {"text": text})
    return {"candidates": [{"content": {"parts": parts}}]}


def test_music_generate_google_generate_content_round_trips() -> None:
    encoded = base64.b64encode(FAKE_AUDIO).decode("ascii")
    with _MockServer(_gemini_music_response(encoded)) as server:
        c = new_client("google", "test-key")
        c.provider.base_url = server.url
        resp = asyncio.run(
            c.music.model(GOOGLE_LYRIA_PRO).generate("Upbeat jazz fusion")
        )

    assert f"/v1beta/models/{GOOGLE_LYRIA_PRO}:generateContent" == server.received_path
    assert server.received_query.get("key") == ["test-key"]
    body = server.received_body or {}
    assert body["contents"] == [{"parts": [{"text": "Upbeat jazz fusion"}]}]
    assert body["generationConfig"] == {"responseModalities": ["AUDIO"]}

    assert len(resp.audio) == 1
    assert resp.audio[0].mime_type == "audio/mpeg"
    assert resp.audio[0].bytes == FAKE_AUDIO


def test_music_generate_google_lyrics_fold_into_text_parts_in_order() -> None:
    encoded = base64.b64encode(FAKE_AUDIO).decode("ascii")
    with _MockServer(_gemini_music_response(encoded)) as server:
        c = new_client("google", "test-key")
        c.provider.base_url = server.url
        asyncio.run(
            c.music.model(GOOGLE_LYRIA_CLIP)
            .text("Pop ballad, 90 BPM")
            .lyrics("When the night falls")
            .generate("")
        )
    body = server.received_body or {}
    assert body["contents"][0]["parts"] == [
        {"text": "Pop ballad, 90 BPM"},
        {"text": "When the night falls"},
    ]


def test_music_generate_google_captures_text_part() -> None:
    encoded = base64.b64encode(FAKE_AUDIO).decode("ascii")
    response = _gemini_music_response(encoded, text="Generated lyrics: la la la")
    with _MockServer(response) as server:
        c = new_client("google", "test-key")
        c.provider.base_url = server.url
        resp = asyncio.run(c.music.model(GOOGLE_LYRIA_PRO).generate("x"))
    assert resp.text == "Generated lyrics: la la la"
    assert len(resp.audio) == 1


# ===== MiniMax Music 2.6 (MusicMinimax wire shape) =====
#
# MiniMax's gen_endpoint is an absolute https URL, so we monkeypatch
# llmkit.music.do_post to capture the call and serve a hex payload.


class _Captured:
    def __init__(self) -> None:
        self.url = ""
        self.headers: dict[str, str] = {}
        self.body: dict[str, Any] = {}


def _patch_minimax(monkeypatch, response_body: dict[str, Any]) -> _Captured:
    cap = _Captured()

    def fake_do_post(url, body, headers, timeout=600.0):
        cap.url = url
        cap.headers = headers
        cap.body = json.loads(body.decode("utf-8"))
        return json.dumps(response_body).encode("utf-8")

    monkeypatch.setattr(music_mod, "do_post", fake_do_post)
    return cap


def test_music_generate_minimax_hex_round_trip_and_absolute_url(monkeypatch) -> None:
    hex_audio = FAKE_AUDIO.hex()
    cap = _patch_minimax(
        monkeypatch,
        {"data": {"audio": hex_audio}, "base_resp": {"status_code": 0, "status_msg": "success"}},
    )

    c = new_client("minimax", "mm-key")
    resp = asyncio.run(
        c.music.model(MINIMAX_MUSIC)
        .lyrics("Sing a song of sixpence")
        .generate("A nursery rhyme melody")
    )

    assert cap.url == "https://api.minimax.io/v1/music_generation"
    assert cap.headers.get("Authorization") == "Bearer mm-key"
    assert cap.body["model"] == MINIMAX_MUSIC
    assert cap.body["prompt"] == "A nursery rhyme melody"
    assert cap.body["lyrics"] == "Sing a song of sixpence"
    assert cap.body["output_format"] == "hex"
    assert cap.body["audio_setting"] == {
        "sample_rate": 44100,
        "bitrate": 128000,
        "format": "mp3",
    }

    assert len(resp.audio) == 1
    assert resp.audio[0].mime_type == "audio/mpeg"
    assert resp.audio[0].bytes == FAKE_AUDIO
    assert resp.finish_message == ""


def test_music_generate_minimax_surfaces_non_success_status_msg(monkeypatch) -> None:
    cap = _patch_minimax(
        monkeypatch,
        {"data": {"audio": ""}, "base_resp": {"status_code": 1004, "status_msg": "rate limited"}},
    )
    c = new_client("minimax", "mm-key")
    resp = asyncio.run(c.music.model(MINIMAX_MUSIC).generate("x"))
    assert cap.body["prompt"] == "x"
    assert "lyrics" not in cap.body  # no lyrics part → key omitted
    assert resp.audio == []
    assert resp.finish_message == "rate limited"


# ===== Cross-shape validation =====


def test_music_generate_requires_model() -> None:
    with pytest.raises(ValidationError) as exc_info:
        c = new_client("google", "k")
        asyncio.run(c.music.generate("x"))
    assert exc_info.value.field == "model"


def test_music_generate_rejects_image_part() -> None:
    from llmkit import MediaRef, Part

    c = new_client("google", "test-key")
    c.provider.base_url = "http://unused"
    # Construct a Part list directly with an image part via the runtime,
    # since the Music builder has no image() method by design.
    req = music_mod.MusicRequest(
        model=GOOGLE_LYRIA_PRO,
        parts=[Part(image=MediaRef(mime_type="image/png", bytes=b"\x89PNG"))],
    )
    from llmkit.types import Provider

    with pytest.raises(ValidationError) as exc_info:
        music_mod.generate_music(
            Provider(name="google", api_key="test-key", base_url="http://unused"),
            req,
        )
    assert exc_info.value.field == "parts[0]"
    assert "image parts" in exc_info.value.message


def test_music_generate_rejects_neither_prompt_nor_parts() -> None:
    with pytest.raises(ValidationError) as exc_info:
        c = new_client("google", "test-key")
        c.provider.base_url = "http://unused"
        asyncio.run(c.music.model(GOOGLE_LYRIA_PRO).generate(""))
    assert exc_info.value.field == "prompt"


def test_music_generate_rejects_unknown_model() -> None:
    with pytest.raises(ValidationError) as exc_info:
        c = new_client("google", "test-key")
        c.provider.base_url = "http://unused"
        asyncio.run(c.music.model("lyria-999").generate("x"))
    assert exc_info.value.field == "model"


def test_music_generate_rejects_unsupported_provider() -> None:
    with pytest.raises(ValidationError) as exc_info:
        c = new_client("openai", "test-key")
        c.provider.base_url = "http://unused"
        asyncio.run(c.music.model("music-2.6").generate("x"))
    assert exc_info.value.field == "provider"
    assert "does not support music generation" in exc_info.value.message


# ===== Middleware =====


def test_music_generate_middleware_fires_pre_then_post() -> None:
    encoded = base64.b64encode(FAKE_AUDIO).decode("ascii")
    ops: list[str] = []
    phases: list[str] = []

    def mw(event):
        ops.append(event.op.value)
        phases.append(event.phase.value)
        return None

    with _MockServer(_vertex_music_response(encoded)) as server:
        c = new_client("vertex", "test-token")
        c.provider.base_url = server.url
        asyncio.run(
            c.music.model(VERTEX_LYRIA).add_middleware(mw).generate("x")
        )
    assert ops == ["music_generation", "music_generation"]
    assert phases == ["pre", "post"]


def test_music_generate_middleware_pre_phase_can_veto() -> None:
    def mw(event):
        if event.phase.value == "pre":
            return RuntimeError("no music today")
        return None

    with _MockServer(_vertex_music_response("")) as server:
        c = new_client("vertex", "test-token")
        c.provider.base_url = server.url
        with pytest.raises(MiddlewareVetoError):
            asyncio.run(
                c.music.model(VERTEX_LYRIA).add_middleware(mw).generate("x")
            )
        assert server.hit is False
