"""Image generation tests — mock HTTP server, no live API calls."""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

import llmkit
from llmkit import (
    Image,
    ImageRequest,
    MiddlewareVetoError,
    Text,
    ValidationError,
    generate_image,
)

FLASH_MODEL = "gemini-3.1-flash-image-preview"
PRO_MODEL = "gemini-3-pro-image-preview"

FAKE_PNG = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


class _MockServer:
    """Single-shot HTTP server that captures one request and serves a canned response."""

    def __init__(self, response_body: dict[str, Any]):
        self.response_body = response_body
        self.received_path = ""
        self.received_query: dict[str, list[str]] = {}
        self.received_body: dict[str, Any] | None = None
        self.received_headers: dict[str, str] = {}

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):  # silence noise
                pass

            def do_POST(self):
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
        port = self._httpd.server_port
        return f"http://127.0.0.1:{port}"


def _flash_response(encoded: str, prompt_tokens: int = 12, output_tokens: int = 1290) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": encoded}},
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": output_tokens,
        },
    }


def test_generate_image_google_flash_round_trips_png() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _MockServer(_flash_response(encoded)) as server:
        resp = generate_image(
            llmkit.Provider(name="google", api_key="test-key", base_url=server.url),
            ImageRequest(prompt="A nano banana dish", model=FLASH_MODEL),
            aspect_ratio="16:9",
            image_size="2K",
        )

    assert FLASH_MODEL + ":generateContent" in server.received_path
    assert server.received_query.get("key") == ["test-key"]
    body = server.received_body
    assert body is not None
    assert body["generationConfig"]["responseModalities"] == ["IMAGE"]
    assert body["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"
    assert body["generationConfig"]["imageConfig"]["imageSize"] == "2K"

    assert len(resp.images) == 1
    assert resp.images[0].mime_type == "image/png"
    assert resp.images[0].data == FAKE_PNG
    assert resp.tokens.input == 12
    assert resp.tokens.output == 1290
    assert resp.text == ""


def test_generate_image_with_include_text_captures_text_part() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Here is your image:"},
                        {"inlineData": {"mimeType": "image/png", "data": encoded}},
                    ]
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 100},
    }
    with _MockServer(response) as server:
        resp = generate_image(
            llmkit.Provider(name="google", api_key="k", base_url=server.url),
            ImageRequest(prompt="x", model=FLASH_MODEL),
            include_text=True,
        )
    assert server.received_body is not None
    assert server.received_body["generationConfig"]["responseModalities"] == ["TEXT", "IMAGE"]
    assert resp.text == "Here is your image:"


def test_generate_image_parts_interleaved_compositional() -> None:
    # ADR-008's motivating scenario: text and reference images interleaved
    # so the model attends to the description-image pairing as intended.
    ref_a = b"\x89PNGA"
    ref_b = b"\x89PNGB"
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _MockServer(_flash_response(encoded)) as server:
        generate_image(
            llmkit.Provider(name="google", api_key="k", base_url=server.url),
            ImageRequest(
                model=FLASH_MODEL,
                parts=[
                    Text("Person:"),
                    Image("image/png", ref_a),
                    Text("Outfit:"),
                    Image("image/png", ref_b),
                    Text("Generate the person wearing the outfit."),
                ],
            ),
        )
    body = server.received_body
    assert body is not None
    parts = body["contents"][0]["parts"]
    assert len(parts) == 5
    assert parts[0] == {"text": "Person:"}
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == ref_a
    assert parts[2] == {"text": "Outfit:"}
    assert base64.b64decode(parts[3]["inlineData"]["data"]) == ref_b
    assert parts[4] == {"text": "Generate the person wearing the outfit."}


def test_generate_image_rejects_unsupported_aspect_on_pro() -> None:
    with pytest.raises(ValidationError) as exc_info:
        generate_image(
            llmkit.Provider(name="google", api_key="k", base_url="http://unused"),
            ImageRequest(prompt="x", model=PRO_MODEL),
            aspect_ratio="8:1",
        )
    assert exc_info.value.field == "aspect_ratio"


def test_generate_image_rejects_512_size_on_pro() -> None:
    with pytest.raises(ValidationError) as exc_info:
        generate_image(
            llmkit.Provider(name="google", api_key="k", base_url="http://unused"),
            ImageRequest(prompt="x", model=PRO_MODEL),
            image_size="512",
        )
    assert exc_info.value.field == "image_size"


def test_generate_image_rejects_too_many_image_parts() -> None:
    too_many = [Text("describe and edit:")] + [
        Image("image/png", FAKE_PNG) for _ in range(15)
    ]
    with pytest.raises(ValidationError) as exc_info:
        generate_image(
            llmkit.Provider(name="google", api_key="k", base_url="http://unused"),
            ImageRequest(model=FLASH_MODEL, parts=too_many),
        )
    assert exc_info.value.field == "parts"


def test_generate_image_rejects_both_prompt_and_parts() -> None:
    with pytest.raises(ValidationError) as exc_info:
        generate_image(
            llmkit.Provider(name="google", api_key="k", base_url="http://unused"),
            ImageRequest(model=FLASH_MODEL, prompt="x", parts=[Text("y")]),
        )
    assert exc_info.value.field == "parts"


def test_generate_image_rejects_both_empty() -> None:
    with pytest.raises(ValidationError) as exc_info:
        generate_image(
            llmkit.Provider(name="google", api_key="k", base_url="http://unused"),
            ImageRequest(model=FLASH_MODEL),
        )
    assert exc_info.value.field == "prompt"


def test_generate_image_requires_model() -> None:
    with pytest.raises(ValidationError) as exc_info:
        generate_image(
            llmkit.Provider(name="google", api_key="k"),
            ImageRequest(prompt="x"),
        )
    assert exc_info.value.field == "model"


def test_generate_image_middleware_fires_pre_then_post() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    ops: list[str] = []
    phases: list[str] = []

    def mw(event):
        ops.append(event.op.value)
        phases.append(event.phase.value)
        return None

    with _MockServer(_flash_response(encoded, prompt_tokens=1, output_tokens=2)) as server:
        generate_image(
            llmkit.Provider(name="google", api_key="k", base_url=server.url),
            ImageRequest(prompt="x", model=FLASH_MODEL),
            middleware=[mw],
        )
    assert ops == ["image_generation", "image_generation"]
    assert phases == ["pre", "post"]


def test_generate_image_middleware_pre_phase_can_veto() -> None:
    def mw(event):
        if event.phase.value == "pre":
            return RuntimeError("no images today")
        return None

    with pytest.raises(MiddlewareVetoError):
        generate_image(
            llmkit.Provider(name="google", api_key="k", base_url="http://unused"),
            ImageRequest(prompt="x", model=FLASH_MODEL),
            middleware=[mw],
        )
