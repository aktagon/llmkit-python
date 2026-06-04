"""Spike 036 (PIVOT wire-conformance): request-byte conformance, generalized
across capabilities (structured output, agent-path caching).

Asserts the OUTBOUND request body each SDK produces is value-equal to the shared
golden at codegen/testdata/wire/request/v1/<fixture>.json — the SAME golden
every SDK asserts against. These are the wires BUG-007 (Python malformed Google
body) and BUG-004 (agent-path caching dropped) broke. No API keys.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import llmkit
from llmkit import anthropic, google, openai
from llmkit.types import SafetySetting
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
    """Single-shot mock that records the outbound POST body and headers
    (headers feed the in-driver asserts for load-bearing headers, e.g.
    Anthropic's structured-output beta header)."""

    def __init__(self, response_body: dict[str, Any]):
        self.last_body: dict[str, Any] | None = None
        self.last_headers: dict[str, str] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if raw:
                    outer.last_body = json.loads(raw.decode("utf-8"))
                    outer.last_headers = {k.lower(): v for k, v in self.headers.items()}
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


# Omits "required" so the goldens witness EnforceStrict normalization
# (auto-required); carries additionalProperties:false so Google's strip is
# witnessed too. See the Go driver comment (the minting reference).
_CANONICAL_SCHEMA = (
    '{"type":"object","properties":{"color":{"type":"string"}},'
    '"additionalProperties":false}'
)
_CANONICAL_PROMPT = "What color is a clear daytime sky?"


# 69-byte 1x1 RGB PNG (single brick-red pixel) — the FIXED reference image
# for the image-edit fixture. SAME base64 constant in all four SDK drivers.
_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGM4YWQEAALyAS2s"
    "aifrAAAAAElFTkSuQmCC"
)


# Response shape valid for the text, agent, batch-submit, and image paths
# across providers (id is the batch-create handle; the inlineData part and the
# data[] array are the image-shaped fields for the Google and OpenAI image
# paths — ADR-028 two-helper rule: extend the canned response, don't add
# capture helpers; missing provider paths parse to empty text / zero usage,
# which the drivers never assert).
_CANNED_RESP = {
    "id": "msgbatch_test",
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": '{"color":"blue"}'},
                    {"inlineData": {"mimeType": "image/png", "data": _TINY_PNG_BASE64}},
                ]
            }
        }
    ],
    "content": [{"type": "text", "text": "done"}],
    "data": [{"b64_json": _TINY_PNG_BASE64}],
    "usage": {"input_tokens": 2000, "output_tokens": 5},
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
}


def test_structured_output_google_matches_shared_golden() -> None:
    body, _ = _build_request(
        llmkit.Provider(name="google", api_key="AIza-test"),
        llmkit.Request(user=_CANONICAL_PROMPT, schema=_CANONICAL_SCHEMA),
        llmkit.Options(),
        PROVIDERS["google"],
    )
    _assert_wire_golden("structured-output-google", body)


def test_structured_output_openai_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(c.text.schema(_CANONICAL_SCHEMA).prompt(_CANONICAL_PROMPT))
        assert server.last_body is not None
        _assert_wire_golden("structured-output-openai", server.last_body)


def test_structured_output_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(c.text.schema(_CANONICAL_SCHEMA).prompt(_CANONICAL_PROMPT))
        assert server.last_body is not None
        # ADR-028 Open Questions: load-bearing headers assert in-driver.
        # Without this beta header Anthropic rejects output_format with a 400.
        assert (
            server.last_headers.get("anthropic-beta")
            == "structured-outputs-2025-11-13"
        )
        _assert_wire_golden("structured-output-anthropic", server.last_body)


def test_caching_agent_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.agent.system("a long stable system prefix").caching().prompt("hi")
        )
        assert server.last_body is not None
        _assert_wire_golden("caching-agent-anthropic", server.last_body)


def test_caching_text_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.system("a long stable system prefix").caching().prompt("hi")
        )
        assert server.last_body is not None
        _assert_wire_golden("caching-text-anthropic", server.last_body)


def test_caching_batch_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.system("a long stable system prefix").caching().submit_batch("hi")
        )
        assert server.last_body is not None
        _assert_wire_golden("caching-batch-anthropic", server.last_body)


# === M2: options fixtures, one per model family (see the Go drivers — the
# minting reference — for WIRE-005 provenance and the live rejection matrix
# that shaped each option chain). ===


def test_options_openai_gpt5_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model("gpt-5").max_tokens(1024).reasoning_effort("low").seed(42)
            .prompt("Summarize the plot of Hamlet in two sentences.")
        )
        assert server.last_body is not None
        _assert_wire_golden("options-openai-gpt5", server.last_body)


def test_options_openai_o_series_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model("o4-mini").max_tokens(1024).reasoning_effort("medium").seed(7)
            .prompt("What is the capital of Finland?")
        )
        assert server.last_body is not None
        _assert_wire_golden("options-openai-o-series", server.last_body)


def test_options_openai_gpt4o_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model("gpt-4o").max_tokens(256).temperature(0.7).top_p(0.9)
            .stop_sequences("END_OF_LIST").seed(42)
            .frequency_penalty(0.25).presence_penalty(0.15)
            .prompt("List three primary colors, then write END_OF_LIST.")
        )
        assert server.last_body is not None
        _assert_wire_golden("options-openai-gpt4o", server.last_body)


def test_options_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model("claude-sonnet-4-6").max_tokens(2048).thinking_budget(1024)
            .stop_sequences("END_OF_ANSWER")
            .prompt(
                "Explain in one sentence why the sky appears blue at noon,"
                " then write END_OF_ANSWER."
            )
        )
        assert server.last_body is not None
        _assert_wire_golden("options-anthropic", server.last_body)


def test_options_google_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = google("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model("gemini-3.5-flash").max_tokens(1024).temperature(0.7)
            .top_p(0.9).top_k(40).stop_sequences("END_OF_ANSWER").seed(7)
            .safety_settings([
                SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ])
            .prompt("Name the two largest moons of Jupiter, then write END_OF_ANSWER.")
        )
        assert server.last_body is not None
        _assert_wire_golden("options-google", server.last_body)


def test_options_google_gemini25_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = google("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model("gemini-2.5-flash").max_tokens(1024).temperature(0.5)
            .thinking_budget(512)
            .prompt("How many planets orbit the Sun? Answer with a number.")
        )
        assert server.last_body is not None
        _assert_wire_golden("options-google-gemini25", server.last_body)


# === M2: image-generation fixtures (M5 pull-forward, JSON bodies only;
# multipart edits are a WIRE-008 documented exclusion). ===


def test_image_gen_google_flash_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = google("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.image.model("gemini-3.1-flash-image-preview")
            .aspect_ratio("16:9").image_size("2K")
            .generate("A lighthouse on a rocky coastline at dusk")
        )
        assert server.last_body is not None
        _assert_wire_golden("image-gen-google-flash", server.last_body)


def test_image_gen_google_pro_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = google("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.image.model("gemini-3-pro-image-preview")
            .aspect_ratio("4:3").image_size("1K").include_text()
            .generate("A watercolor map of the Baltic Sea")
        )
        assert server.last_body is not None
        _assert_wire_golden("image-gen-google-pro", server.last_body)


def test_image_gen_openai_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.image.model("gpt-image-2").image_size("1024x1024").quality("low")
            .output_format("png").background("opaque").count(1)
            .generate("A minimalist line drawing of a sailboat")
        )
        assert server.last_body is not None
        _assert_wire_golden("image-gen-openai", server.last_body)


def test_image_edit_google_flash_matches_shared_golden() -> None:
    png = base64.b64decode(_TINY_PNG_BASE64)
    with _CaptureServer(_CANNED_RESP) as server:
        c = google("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.image.model("gemini-3.1-flash-image-preview")
            .image("image/png", png)
            .generate("Recolor the square to deep blue")
        )
        assert server.last_body is not None
        _assert_wire_golden("image-edit-google-flash", server.last_body)
