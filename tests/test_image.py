"""Image generation tests — mock HTTP server, no live API calls."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from llmkit import MiddlewareVetoError, ValidationError
from llmkit.builders import new_client

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


def _client(server_url: str | None = None):
    c = new_client("google", "test-key" if server_url else "k")
    if server_url:
        c.provider.base_url = server_url
    else:
        c.provider.base_url = "http://unused"
    return c


def test_image_generate_google_flash_round_trips_png() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _MockServer(_flash_response(encoded)) as server:
        c = _client(server.url)
        resp = asyncio.run(
            c.image.model(FLASH_MODEL).aspect_ratio("16:9").image_size("2K").generate("A nano banana dish")
        )

    assert FLASH_MODEL + ":generateContent" in server.received_path
    assert server.received_query.get("key") == ["test-key"]
    # Body-shape asserts (generationConfig/imageConfig) migrated to the
    # image-gen-google-flash wire fixture (ADR-028 M2); URL/auth shape and
    # response parsing remain this test's subjects.

    assert len(resp.images) == 1
    assert resp.images[0].mime_type == "image/png"
    assert resp.images[0].bytes == FAKE_PNG
    assert resp.usage.input == 12
    assert resp.usage.output == 1290
    assert resp.text == ""


def test_image_generate_with_include_text_captures_text_part() -> None:
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
        c = _client(server.url)
        resp = asyncio.run(
            c.image.model(FLASH_MODEL).include_text().generate("x")
        )
    # The [TEXT, IMAGE] modality body assert migrated to the
    # image-gen-google-pro wire fixture (ADR-028 M2).
    assert resp.text == "Here is your image:"


# The Parts positional-ordering wire test (ADR-008) migrated to the
# wire-conformance suite: the image-edit-google-flash fixture witnesses
# inlineData encoding and caller-order preservation byte-for-byte
# (ADR-028 M2, falsification class d2).


def test_image_generate_rejects_unsupported_aspect_on_pro() -> None:
    with pytest.raises(ValidationError) as exc_info:
        c = _client()
        asyncio.run(
            c.image.model(PRO_MODEL).aspect_ratio("8:1").generate("x")
        )
    assert exc_info.value.field == "aspect_ratio"


def test_image_generate_rejects_512_size_on_pro() -> None:
    with pytest.raises(ValidationError) as exc_info:
        c = _client()
        asyncio.run(
            c.image.model(PRO_MODEL).image_size("512").generate("x")
        )
    assert exc_info.value.field == "image_size"


def test_image_generate_rejects_too_many_image_parts() -> None:
    with pytest.raises(ValidationError) as exc_info:
        c = _client()
        img = c.image.model(FLASH_MODEL).text("describe and edit:")
        for _ in range(15):
            img = img.image("image/png", FAKE_PNG)
        asyncio.run(img.generate(""))
    assert exc_info.value.field == "parts"


# The "both prompt and parts set" XOR test from the legacy free-function
# surface is no longer reachable via typed-builder: chain methods either
# accumulate parts or pass a final-text msg, never both as a free-form pair.


def test_image_generate_rejects_neither_set() -> None:
    with pytest.raises(ValidationError) as exc_info:
        c = _client()
        asyncio.run(c.image.model(FLASH_MODEL).generate(""))
    assert exc_info.value.field == "prompt"


def test_image_generate_requires_model() -> None:
    with pytest.raises(ValidationError) as exc_info:
        c = new_client("google", "k")
        asyncio.run(c.image.generate("x"))
    assert exc_info.value.field == "model"


def test_image_generate_middleware_fires_pre_then_post() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    ops: list[str] = []
    phases: list[str] = []

    def mw(event):
        ops.append(event.op.value)
        phases.append(event.phase.value)
        return None

    with _MockServer(_flash_response(encoded, prompt_tokens=1, output_tokens=2)) as server:
        c = _client(server.url)
        asyncio.run(
            c.image.model(FLASH_MODEL).add_middleware(mw).generate("x")
        )
    assert ops == ["image_generation", "image_generation"]
    assert phases == ["pre", "post"]


def test_image_generate_middleware_pre_phase_can_veto() -> None:
    def mw(event):
        if event.phase.value == "pre":
            return RuntimeError("no images today")
        return None

    with pytest.raises(MiddlewareVetoError):
        c = _client()
        asyncio.run(
            c.image.model(FLASH_MODEL).add_middleware(mw).generate("x")
        )


# ===== OpenAI Image API (plan 020 phase 4) =====
#
# Two endpoints: /v1/images/generations (JSON; no image parts) and
# /v1/images/edits (multipart/form-data; one or more image parts).
# Output is forced to b64_json so the response shape stays uniform.

OPENAI_IMAGE_2 = "gpt-image-2"


def _openai_image_response(encoded: str, n: int = 1) -> dict[str, Any]:
    return {
        "created": 1700000000,
        "data": [{"b64_json": encoded} for _ in range(n)],
        "usage": {"input_tokens": 7, "output_tokens": 1500},
    }


class _OpenAIMockServer:
    """Mock that captures either JSON (generations) or multipart (edits)
    bodies. Parses multipart with stdlib ``email`` so test assertions can
    walk fields and image[] files in caller order.
    """

    def __init__(self, response_body: dict[str, Any]):
        self.response_body = response_body
        self.received_path = ""
        self.received_headers: dict[str, str] = {}
        self.received_json: dict[str, Any] | None = None
        self.received_form_fields: dict[str, str] = {}
        self.received_form_files: dict[str, list[dict[str, Any]]] = {}
        self.hit = False

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):
                pass

            def do_POST(self):
                outer.hit = True
                outer.received_path = urlparse(self.path).path
                outer.received_headers = dict(self.headers.items())
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                ctype = self.headers.get("Content-Type", "")
                if ctype.startswith("multipart/form-data"):
                    import email
                    full = b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw
                    msg = email.message_from_bytes(full)
                    for part in msg.walk():
                        if part.is_multipart():
                            continue
                        cd = part.get("Content-Disposition", "")
                        if not cd or "form-data" not in cd:
                            continue
                        name = part.get_param("name", header="content-disposition")
                        filename = part.get_param("filename", header="content-disposition")
                        payload = part.get_payload(decode=True) or b""
                        if filename:
                            outer.received_form_files.setdefault(name, []).append(
                                {
                                    "filename": filename,
                                    "mime": part.get_content_type(),
                                    "bytes": payload,
                                }
                            )
                        else:
                            outer.received_form_fields[name] = payload.decode("utf-8")
                else:
                    outer.received_json = json.loads(raw.decode("utf-8")) if raw else {}

                payload = json.dumps(outer.response_body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_OpenAIMockServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def _openai_client(server_url: str | None = None):
    c = new_client("openai", "test-key")
    c.provider.base_url = server_url or "http://unused"
    return c


def test_image_generate_openai_generations_omits_response_format() -> None:
    """gpt-image-* always returns b64_json and rejects the
    response_format parameter — must be absent on the wire."""
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _OpenAIMockServer(_openai_image_response(encoded)) as server:
        c = _openai_client(server.url)
        resp = asyncio.run(c.image.model(OPENAI_IMAGE_2).generate("A red circle"))

    assert server.received_path == "/v1/images/generations"
    assert server.received_headers.get("Authorization") == "Bearer test-key"
    body = server.received_json
    assert body is not None
    assert body["model"] == OPENAI_IMAGE_2
    assert body["prompt"] == "A red circle"
    assert "response_format" not in body
    assert "size" not in body

    assert len(resp.images) == 1
    assert resp.images[0].bytes == FAKE_PNG
    assert resp.usage.input == 7
    assert resp.usage.output == 1500


def test_image_generate_openai_edits_single_reference() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    ref_bytes = bytes([0x89, 0x50, 0x4E, 0x47, 0x41])
    with _OpenAIMockServer(_openai_image_response(encoded)) as server:
        c = _openai_client(server.url)
        resp = asyncio.run(
            c.image.model(OPENAI_IMAGE_2)
            .image("image/png", ref_bytes)
            .generate("Add a hat")
        )

    assert server.received_path == "/v1/images/edits"
    assert server.received_form_fields["model"] == OPENAI_IMAGE_2
    assert server.received_form_fields["prompt"] == "Add a hat"
    files = server.received_form_files["image[]"]
    assert len(files) == 1
    assert files[0]["bytes"] == ref_bytes
    assert files[0]["mime"] == "image/png"
    assert len(resp.images) == 1


def test_image_generate_openai_edits_three_references_preserves_caller_order() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    ref_a = bytes([0x89, 0x50, 0x41])
    ref_b = bytes([0x89, 0x50, 0x42])
    ref_c = bytes([0x89, 0x50, 0x43])
    with _OpenAIMockServer(_openai_image_response(encoded)) as server:
        c = _openai_client(server.url)
        asyncio.run(
            c.image.model(OPENAI_IMAGE_2)
            .image("image/png", ref_a)
            .image("image/png", ref_b)
            .image("image/png", ref_c)
            .generate("Combine them")
        )

    files = server.received_form_files["image[]"]
    assert len(files) == 3
    assert files[0]["bytes"] == ref_a
    assert files[1]["bytes"] == ref_b
    assert files[2]["bytes"] == ref_c


def test_image_generate_openai_extra_fields_quality_propagates() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _OpenAIMockServer(_openai_image_response(encoded)) as server:
        c = _openai_client(server.url)
        asyncio.run(
            c.image.model(OPENAI_IMAGE_2)
            .extra_fields({"quality": "high"})
            .generate("x")
        )
    assert server.received_json is not None
    assert server.received_json["quality"] == "high"


def test_image_generate_openai_extra_fields_n_returns_n_images() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _OpenAIMockServer(_openai_image_response(encoded, n=4)) as server:
        c = _openai_client(server.url)
        resp = asyncio.run(
            c.image.model(OPENAI_IMAGE_2)
            .extra_fields({"n": 4})
            .generate("x")
        )
    assert server.received_json is not None
    assert server.received_json["n"] == 4
    assert len(resp.images) == 4


def test_image_generate_openai_arbitrary_size_accepted() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _OpenAIMockServer(_openai_image_response(encoded)) as server:
        c = _openai_client(server.url)
        asyncio.run(
            c.image.model(OPENAI_IMAGE_2).image_size("1536x1024").generate("x")
        )
    assert server.received_json is not None
    assert server.received_json["size"] == "1536x1024"


def test_image_generate_openai_middleware_fires_both_branches() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    for branch in ("generations", "edits"):
        ops: list[str] = []
        phases: list[str] = []

        def mw(event):
            ops.append(event.op.value)
            phases.append(event.phase.value)
            return None

        with _OpenAIMockServer(_openai_image_response(encoded)) as server:
            c = _openai_client(server.url)
            b = c.image.model(OPENAI_IMAGE_2).add_middleware(mw)
            if branch == "edits":
                b = b.image("image/png", bytes([0x89, 0x50, 0x4E]))
            asyncio.run(b.generate("x"))
        assert ops == ["image_generation", "image_generation"], branch
        assert phases == ["pre", "post"], branch


def test_image_generate_openai_middleware_veto_skips_http() -> None:
    def mw(event):
        if event.phase.value == "pre":
            return RuntimeError("blocked")
        return None

    with _OpenAIMockServer(_openai_image_response("")) as server:
        c = _openai_client(server.url)
        with pytest.raises(MiddlewareVetoError):
            asyncio.run(
                c.image.model(OPENAI_IMAGE_2).add_middleware(mw).generate("x")
            )
        assert server.hit is False


# ===== xAI Grok Imagine =====
#
# JSON throughout — both endpoints. Image refs travel as data URLs in the
# body. response_format must be forced to b64_json (xAI defaults to URL).

GROK_IMAGINE_QUALITY = "grok-imagine-image-quality"


def _grok_image_response(
    encoded: str, n: int = 1, mime: str | None = "image/png"
) -> dict[str, Any]:
    data: list[dict[str, Any]] = []
    for _ in range(n):
        entry: dict[str, Any] = {"b64_json": encoded}
        if mime:
            entry["mime_type"] = mime
        data.append(entry)
    return {"data": data, "usage": {"cost_in_usd_ticks": 1234567}}


def _grok_client(server_url: str | None = None):
    c = new_client("grok", "test-key")
    c.provider.base_url = server_url or "http://unused"
    return c


def test_image_generate_grok_generations_forces_b64_json() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _OpenAIMockServer(_grok_image_response(encoded)) as server:
        c = _grok_client(server.url)
        resp = asyncio.run(c.image.model(GROK_IMAGINE_QUALITY).generate("A red circle"))

    assert server.received_path == "/v1/images/generations"
    body = server.received_json
    assert body is not None
    assert body["model"] == GROK_IMAGINE_QUALITY
    assert body["prompt"] == "A red circle"
    # xAI defaults to URL — we must force b64_json on the wire.
    assert body["response_format"] == "b64_json"
    assert "image" not in body
    assert "images" not in body
    assert len(resp.images) == 1
    assert resp.images[0].bytes == FAKE_PNG
    assert resp.images[0].mime_type == "image/png"
    # xAI reports cost_in_usd_ticks, not tokens. Both should remain zero.
    assert resp.usage.input == 0
    assert resp.usage.output == 0


def test_image_generate_grok_aspect_ratio_and_resolution() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _OpenAIMockServer(_grok_image_response(encoded)) as server:
        c = _grok_client(server.url)
        asyncio.run(
            c.image.model(GROK_IMAGINE_QUALITY)
            .aspect_ratio("16:9")
            .image_size("2k")
            .generate("x")
        )
    body = server.received_json
    assert body is not None
    assert body["aspect_ratio"] == "16:9"
    # image_size maps to xAI's `resolution` field (different name from OpenAI's `size`).
    assert body["resolution"] == "2k"


def test_image_generate_grok_rejects_unsupported_aspect_ratio() -> None:
    c = _grok_client()
    with pytest.raises(ValidationError):
        asyncio.run(
            c.image.model(GROK_IMAGINE_QUALITY)
            .aspect_ratio("4:5")
            .generate("x")
        )


def test_image_generate_grok_accepts_auto_aspect_ratio() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _OpenAIMockServer(_grok_image_response(encoded)) as server:
        c = _grok_client(server.url)
        asyncio.run(
            c.image.model(GROK_IMAGINE_QUALITY)
            .aspect_ratio("auto")
            .generate("x")
        )
    body = server.received_json
    assert body is not None
    assert body["aspect_ratio"] == "auto"


def test_image_generate_grok_edits_single_reference_as_data_url() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    ref_bytes = bytes([0x89, 0x50, 0x4E, 0x47, 0x41])
    expected_data_url = (
        "data:image/png;base64," + base64.b64encode(ref_bytes).decode("ascii")
    )
    with _OpenAIMockServer(_grok_image_response(encoded)) as server:
        c = _grok_client(server.url)
        asyncio.run(
            c.image.model(GROK_IMAGINE_QUALITY)
            .image("image/png", ref_bytes)
            .generate("Add a hat")
        )

    assert server.received_path == "/v1/images/edits"
    body = server.received_json
    assert body is not None
    # Single ref → `image: {url: "data:..."}` (not `images: [...]`).
    assert body["image"] == {"url": expected_data_url}
    assert "images" not in body


def test_image_generate_grok_edits_three_references_as_images_array_in_order() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    ref_a = bytes([0x89, 0x41])
    ref_b = bytes([0x89, 0x42])
    ref_c = bytes([0x89, 0x43])
    with _OpenAIMockServer(_grok_image_response(encoded)) as server:
        c = _grok_client(server.url)
        asyncio.run(
            c.image.model(GROK_IMAGINE_QUALITY)
            .image("image/png", ref_a)
            .image("image/png", ref_b)
            .image("image/png", ref_c)
            .generate("Combine them")
        )
    body = server.received_json
    assert body is not None
    images = body["images"]
    assert len(images) == 3
    assert images[0]["url"].endswith(base64.b64encode(ref_a).decode("ascii"))
    assert images[1]["url"].endswith(base64.b64encode(ref_b).decode("ascii"))
    assert images[2]["url"].endswith(base64.b64encode(ref_c).decode("ascii"))
    assert "image" not in body


def test_image_generate_grok_extra_fields_n_returns_n_images() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _OpenAIMockServer(_grok_image_response(encoded, n=4)) as server:
        c = _grok_client(server.url)
        resp = asyncio.run(
            c.image.model(GROK_IMAGINE_QUALITY)
            .extra_fields({"n": 4})
            .generate("x")
        )
    body = server.received_json
    assert body is not None
    assert body["n"] == 4
    assert len(resp.images) == 4


def test_image_generate_grok_middleware_fires_both_branches() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    for branch in ("generations", "edits"):
        ops: list[str] = []
        phases: list[str] = []

        def mw(event):
            ops.append(event.op.value)
            phases.append(event.phase.value)
            return None

        with _OpenAIMockServer(_grok_image_response(encoded)) as server:
            c = _grok_client(server.url)
            b = c.image.model(GROK_IMAGINE_QUALITY).add_middleware(mw)
            if branch == "edits":
                b = b.image("image/png", bytes([0x89, 0x50, 0x4E]))
            asyncio.run(b.generate("x"))
        assert ops == ["image_generation", "image_generation"], branch
        assert phases == ["pre", "post"], branch


# =============================================================================
# Plan 020 phase 2 — typed image-gen knob tests
# =============================================================================


# The quality/output_format/background JSON-body asserts migrated to the
# image-gen-openai wire fixture (ADR-028 M2, falsification class d3),
# which sets all five generations-branch knobs on one canonical call.
# The count test survives for its response-side subject (n=3 -> three
# decoded images), with the body assert dropped.


def test_image_openai_typed_count_yields_three_images() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _OpenAIMockServer(_openai_image_response(encoded, n=3)) as server:
        c = _openai_client(server.url)
        resp = asyncio.run(c.image.model(OPENAI_IMAGE_2).count(3).generate("x"))
    assert len(resp.images) == 3


def test_image_openai_typed_knobs_propagate_as_multipart_fields() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _OpenAIMockServer(_openai_image_response(encoded, n=2)) as server:
        c = _openai_client(server.url)
        asyncio.run(
            c.image.model(OPENAI_IMAGE_2)
            .quality("medium")
            .output_format("png")
            .background("auto")
            .count(2)
            .image("image/png", FAKE_PNG)
            .generate("edit it")
        )
    assert server.received_form_fields.get("quality") == "medium"
    assert server.received_form_fields.get("output_format") == "png"
    assert server.received_form_fields.get("background") == "auto"
    assert server.received_form_fields.get("n") == "2"


def test_image_google_rejects_openai_typed_knobs() -> None:
    c = new_client("google", "k")
    cases = [
        ("quality", lambda b: b.quality("high")),
        ("output_format", lambda b: b.output_format("png")),
        ("background", lambda b: b.background("auto")),
        ("count", lambda b: b.count(2)),
    ]
    for field, build in cases:
        builder = build(c.image.model(FLASH_MODEL))
        with pytest.raises(ValidationError) as excinfo:
            asyncio.run(builder.generate("x"))
        assert excinfo.value.field == field, f"{field}: {excinfo.value.field}"


def test_image_grok_rejects_quality_outputformat_background() -> None:
    c = new_client("grok", "k")
    cases = [
        ("quality", lambda b: b.quality("high")),
        ("output_format", lambda b: b.output_format("png")),
        ("background", lambda b: b.background("auto")),
    ]
    for field, build in cases:
        builder = build(c.image.model(GROK_IMAGINE_QUALITY))
        with pytest.raises(ValidationError) as excinfo:
            asyncio.run(builder.generate("x"))
        assert excinfo.value.field == field, f"{field}: {excinfo.value.field}"


def test_image_grok_typed_count_lands_as_n() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _OpenAIMockServer(_grok_image_response(encoded, n=2)) as server:
        c = _grok_client(server.url)
        resp = asyncio.run(c.image.model(GROK_IMAGINE_QUALITY).count(2).generate("x"))
    assert server.received_json is not None
    assert server.received_json.get("n") == 2
    assert len(resp.images) == 2


def test_image_openai_mask_attaches_to_edit_multipart() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    mask_bytes = bytes([0xDE, 0xAD, 0xBE, 0xEF])
    with _OpenAIMockServer(_openai_image_response(encoded)) as server:
        c = _openai_client(server.url)
        asyncio.run(
            c.image.model(OPENAI_IMAGE_2)
            .image("image/png", FAKE_PNG)
            .mask("image/png", mask_bytes)
            .generate("patch")
        )
    masks = server.received_form_files.get("mask") or []
    assert len(masks) == 1, f"expected one mask file, got {server.received_form_files}"
    assert masks[0]["bytes"] == mask_bytes
    assert masks[0]["mime"] == "image/png"


def test_image_openai_mask_without_image_parts_rejected() -> None:
    c = new_client("openai", "k")
    with pytest.raises(ValidationError) as excinfo:
        asyncio.run(
            c.image.model(OPENAI_IMAGE_2)
            .mask("image/png", bytes([0xDE, 0xAD]))
            .generate("x")
        )
    assert excinfo.value.field == "mask"


def test_image_google_and_grok_reject_mask() -> None:
    g = new_client("google", "k")
    with pytest.raises(ValidationError) as excinfo:
        asyncio.run(
            g.image.model(FLASH_MODEL)
            .mask("image/png", bytes([0xDE, 0xAD]))
            .generate("x")
        )
    assert excinfo.value.field == "mask"

    x = new_client("grok", "k")
    with pytest.raises(ValidationError) as excinfo:
        asyncio.run(
            x.image.model(GROK_IMAGINE_QUALITY)
            .mask("image/png", bytes([0xDE, 0xAD]))
            .generate("x")
        )
    assert excinfo.value.field == "mask"


# =============================================================================
# Vertex Imagen (plan 021) — JSONPredict input mode, bearer auth
# =============================================================================

VERTEX_IMAGEN_3 = "imagen-3.0-generate-002"


def _vertex_response(b64: str, n: int = 1, mime: str = "image/png") -> dict[str, Any]:
    preds = []
    for _ in range(n):
        entry: dict[str, Any] = {"bytesBase64Encoded": b64}
        if mime:
            entry["mimeType"] = mime
        preds.append(entry)
    return {"predictions": preds}


def test_image_vertex_generations_happy_path() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _MockServer(_vertex_response(encoded, 1, "image/png")) as server:
        c = new_client("vertex", "test-token")
        c.provider.base_url = server.url
        resp = asyncio.run(
            c.image.model(VERTEX_IMAGEN_3).generate("A red circle")
        )
        # Path: /{model}:predict
        assert server.received_path == f"/{VERTEX_IMAGEN_3}:predict"
        # Bearer auth header
        assert server.received_headers.get("Authorization") == "Bearer test-token"
        # Body shape
        body = server.received_body or {}
        assert isinstance(body["instances"], list) and len(body["instances"]) == 1
        instance = body["instances"][0]
        assert instance["prompt"] == "A red circle"
        # No image on generation path
        assert "image" not in instance
        # sampleCount defaults to 1
        assert body["parameters"]["sampleCount"] == 1
        # Response decode
        assert len(resp.images) == 1
        assert resp.images[0].bytes == FAKE_PNG
        assert resp.images[0].mime_type == "image/png"
        # Vertex predict does not return token counts
        assert resp.usage.input == 0
        assert resp.usage.output == 0


def test_image_vertex_edit_carries_image_on_instance() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    ref_bytes = bytes([0x01, 0x02, 0x03, 0x04])
    expected_b64 = base64.b64encode(ref_bytes).decode("ascii")
    with _MockServer(_vertex_response(encoded, 1)) as server:
        c = new_client("vertex", "test-token")
        c.provider.base_url = server.url
        asyncio.run(
            c.image.model(VERTEX_IMAGEN_3)
            .image("image/png", ref_bytes)
            .generate("Make it winter")
        )
        body = server.received_body or {}
        instance = body["instances"][0]
        assert instance["image"]["bytesBase64Encoded"] == expected_b64


def test_image_vertex_mask_attaches_to_instance() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    mask_bytes = bytes([0xAA, 0xBB, 0xCC])
    expected_mask_b64 = base64.b64encode(mask_bytes).decode("ascii")
    with _MockServer(_vertex_response(encoded, 1)) as server:
        c = new_client("vertex", "test-token")
        c.provider.base_url = server.url
        asyncio.run(
            c.image.model(VERTEX_IMAGEN_3)
            .image("image/png", bytes([0x01]))
            .mask("image/png", mask_bytes)
            .generate("Inpaint here")
        )
        body = server.received_body or {}
        instance = body["instances"][0]
        assert instance["mask"]["image"]["bytesBase64Encoded"] == expected_mask_b64


def test_image_vertex_count_maps_to_sample_count() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _MockServer(_vertex_response(encoded, 4)) as server:
        c = new_client("vertex", "test-token")
        c.provider.base_url = server.url
        resp = asyncio.run(
            c.image.model(VERTEX_IMAGEN_3).count(4).generate("x")
        )
        body = server.received_body or {}
        assert body["parameters"]["sampleCount"] == 4
        assert len(resp.images) == 4


def test_image_vertex_aspect_ratio_maps_to_parameters() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _MockServer(_vertex_response(encoded, 1)) as server:
        c = new_client("vertex", "test-token")
        c.provider.base_url = server.url
        asyncio.run(
            c.image.model(VERTEX_IMAGEN_3).aspect_ratio("16:9").generate("x")
        )
        body = server.received_body or {}
        assert body["parameters"]["aspectRatio"] == "16:9"


def test_image_vertex_extra_fields_spread_into_parameters() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _MockServer(_vertex_response(encoded, 1)) as server:
        c = new_client("vertex", "test-token")
        c.provider.base_url = server.url
        asyncio.run(
            c.image.model(VERTEX_IMAGEN_3)
            .extra_fields({"negativePrompt": "ugly", "safetySetting": "block_some"})
            .generate("x")
        )
        body = server.received_body or {}
        assert body["parameters"]["negativePrompt"] == "ugly"
        assert body["parameters"]["safetySetting"] == "block_some"


def test_image_vertex_rejects_quality_output_format_background() -> None:
    c = new_client("vertex", "test-token")
    c.provider.base_url = "http://unused"

    for chain, expected_field in [
        (lambda b: b.quality("high"), "quality"),
        (lambda b: b.output_format("png"), "output_format"),
        (lambda b: b.background("transparent"), "background"),
    ]:
        with pytest.raises(ValidationError) as excinfo:
            asyncio.run(chain(c.image.model(VERTEX_IMAGEN_3)).generate("x"))
        assert excinfo.value.field == expected_field


def test_image_generate_google_surfaces_finish_reason_when_blocked() -> None:
    """Gemini returns a candidate with finishReason + finishMessage but no
    parts when it declines to generate. Verify both fields land on
    ImageResponse so callers can show a useful message."""
    response = {
        "candidates": [
            {
                "finishReason": "IMAGE_OTHER",
                "finishMessage": "Could not generate image. Try rephrasing the prompt.",
            }
        ],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 0},
    }
    with _MockServer(response) as server:
        c = _client(server.url)
        resp = asyncio.run(c.image.model(FLASH_MODEL).generate("blocked"))

    assert resp.images == []
    assert resp.finish_reason == "IMAGE_OTHER"
    assert resp.finish_message == "Could not generate image. Try rephrasing the prompt."


def test_image_generate_google_omits_finish_reason_on_success() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _MockServer(_flash_response(encoded)) as server:
        c = _client(server.url)
        resp = asyncio.run(c.image.model(FLASH_MODEL).generate("a cat"))
    assert len(resp.images) == 1
    assert resp.finish_reason == ""
    assert resp.finish_message == ""


def test_image_generate_vertex_surfaces_rai_filtered_reason() -> None:
    response = {
        "predictions": [{"raiFilteredReason": "Image filtered by safety system"}],
    }
    with _MockServer(response) as server:
        c = new_client(
            "vertex",
            "Bearer fake-token-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        c.provider.base_url = server.url
        resp = asyncio.run(c.image.model(VERTEX_IMAGEN_3).generate("blocked"))
    assert resp.images == []
    assert resp.finish_reason == "Image filtered by safety system"
    assert resp.finish_message == ""


def test_image_vertex_safety_filter_maps_to_parameters() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode()
    with _MockServer(_vertex_response(encoded, 1)) as server:
        c = new_client("vertex", "test-token")
        c.provider.base_url = server.url
        asyncio.run(
            c.image.model(VERTEX_IMAGEN_3).safety_filter("block_few").generate("x")
        )
    params = server.received_body["parameters"]
    assert params["safetySetting"] == "block_few"


def test_image_safety_filter_rejected_on_non_vertex() -> None:
    c = new_client("google", "key")
    c.provider.base_url = "http://unused"
    with pytest.raises(ValidationError):
        asyncio.run(
            c.image.model(FLASH_MODEL).safety_filter("block_few").generate("x")
        )


def test_image_google_safety_settings_wire_body() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode()
    with _MockServer(_flash_response(encoded)) as server:
        c = new_client("google", "key")
        c.provider.base_url = server.url
        from llmkit.types import SafetySetting
        asyncio.run(
            c.image.model(FLASH_MODEL)
            .safety_settings([SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")])
            .generate("a cat")
        )
    ss = server.received_body.get("safetySettings")
    assert ss == [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]


def test_image_safety_settings_rejected_on_openai() -> None:
    c = new_client("openai", "key")
    c.provider.base_url = "http://unused"
    from llmkit.types import SafetySetting
    with pytest.raises(ValidationError):
        asyncio.run(
            c.image.model("gpt-image-1")
            .safety_settings([SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")])
            .generate("x")
        )


# === Recraft (JSONGenerations input mode) ===

RECRAFT_V3 = "recraftv3"
RECRAFT_V3_VECTOR = "recraftv3_vector"
FAKE_SVG = b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>'


def _recraft_client(server_url: str | None = None):
    c = new_client("recraft", "test-key")
    c.provider.base_url = server_url or "http://unused"
    return c


def test_image_generate_recraft_generations_happy_path() -> None:
    encoded = base64.b64encode(FAKE_PNG).decode("ascii")
    with _OpenAIMockServer({"data": [{"b64_json": encoded}]}) as server:
        c = _recraft_client(server.url)
        resp = asyncio.run(
            c.image.model(RECRAFT_V3).image_size("1024x1024").generate("A red circle")
        )

    assert server.received_path == "/v1/images/generations"
    assert server.received_headers.get("Authorization") == "Bearer test-key"
    body = server.received_json
    assert body is not None
    assert body["model"] == RECRAFT_V3
    assert body["prompt"] == "A red circle"
    # Recraft defaults to URL — we must force b64_json on the wire.
    assert body["response_format"] == "b64_json"
    assert body["size"] == "1024x1024"
    # Text-to-image only: no image/images fields.
    assert "image" not in body
    assert "images" not in body
    assert len(resp.images) == 1
    assert resp.images[0].bytes == FAKE_PNG
    assert resp.images[0].mime_type == "image/png"
    # Recraft returns no usage object; tokens stay zero (no fabricated values).
    assert resp.usage.input == 0
    assert resp.usage.output == 0


def test_image_generate_recraft_vector_sniffs_svg() -> None:
    encoded = base64.b64encode(FAKE_SVG).decode("ascii")
    with _OpenAIMockServer({"data": [{"b64_json": encoded}]}) as server:
        c = _recraft_client(server.url)
        resp = asyncio.run(
            c.image.model(RECRAFT_V3_VECTOR).generate("A sailboat logo")
        )

    body = server.received_json
    assert body is not None
    assert body["model"] == RECRAFT_V3_VECTOR
    assert len(resp.images) == 1
    assert resp.images[0].bytes == FAKE_SVG
    # Vector output: SVG bytes in the same b64_json slot, no mime echoed -> sniff.
    assert resp.images[0].mime_type == "image/svg+xml"


def test_image_generate_recraft_rejects_image_parts() -> None:
    c = _recraft_client()
    with pytest.raises(ValidationError) as exc:
        asyncio.run(
            c.image.model(RECRAFT_V3).image("image/png", FAKE_PNG).generate("edit this")
        )
    assert exc.value.field == "parts"


def test_image_generate_recraft_rejects_aspect_ratio() -> None:
    c = _recraft_client()
    with pytest.raises(ValidationError) as exc:
        asyncio.run(
            c.image.model(RECRAFT_V3).aspect_ratio("16:9").generate("A red circle")
        )
    assert exc.value.field == "aspect_ratio"
