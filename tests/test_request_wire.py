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
import wire_inputs as wi

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


# Canonical inputs are single-sourced from ontology/wire-fixtures.ttl (plan
# 039) via the generated wire_inputs.py consts. The schema omits "required"
# so the goldens witness EnforceStrict normalization (auto-required); it
# carries additionalProperties:false so Google's strip is witnessed too. See
# the Go driver comment (the minting reference).


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
                    {"inlineData": {"mimeType": "image/png", "data": wi.WIRE_IMAGE_EDIT_GOOGLE_FLASH_IMAGE_BASE64}},
                ]
            }
        }
    ],
    "content": [{"type": "text", "text": "done"}],
    "data": [{"b64_json": wi.WIRE_IMAGE_EDIT_GOOGLE_FLASH_IMAGE_BASE64}],
    "usage": {"input_tokens": 2000, "output_tokens": 5},
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
}


def test_structured_output_google_matches_shared_golden() -> None:
    body, _ = _build_request(
        llmkit.Provider(name="google", api_key="AIza-test"),
        llmkit.Request(user=wi.WIRE_STRUCTURED_OUTPUT_PROMPT, schema=wi.WIRE_STRUCTURED_OUTPUT_SCHEMA),
        llmkit.Options(),
        PROVIDERS["google"],
    )
    _assert_wire_golden("structured-output-google", body)


def test_structured_output_openai_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(c.text.schema(wi.WIRE_STRUCTURED_OUTPUT_SCHEMA).prompt(wi.WIRE_STRUCTURED_OUTPUT_PROMPT))
        assert server.last_body is not None
        _assert_wire_golden("structured-output-openai", server.last_body)


def test_structured_output_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(c.text.schema(wi.WIRE_STRUCTURED_OUTPUT_SCHEMA).prompt(wi.WIRE_STRUCTURED_OUTPUT_PROMPT))
        assert server.last_body is not None
        # ADR-028 Open Questions: load-bearing headers assert in-driver.
        # Without this beta header Anthropic rejects output_format with a 400.
        assert (
            server.last_headers.get("anthropic-beta")
            == "structured-outputs-2025-11-13"
        )
        _assert_wire_golden("structured-output-anthropic", server.last_body)


# === Plan 039: nested-schema fixtures — the recursive normalization walk
# (witness-lint first catch; see the Go drivers for the rationale). ===


def test_structured_output_nested_google_matches_shared_golden() -> None:
    body, _ = _build_request(
        llmkit.Provider(name="google", api_key="AIza-test"),
        llmkit.Request(
            user=wi.WIRE_STRUCTURED_OUTPUT_NESTED_PROMPT,
            schema=wi.WIRE_STRUCTURED_OUTPUT_NESTED_SCHEMA,
        ),
        llmkit.Options(),
        PROVIDERS["google"],
    )
    _assert_wire_golden("structured-output-nested-google", body)


def test_structured_output_nested_openai_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.schema(wi.WIRE_STRUCTURED_OUTPUT_NESTED_SCHEMA).prompt(
                wi.WIRE_STRUCTURED_OUTPUT_NESTED_PROMPT
            )
        )
        assert server.last_body is not None
        _assert_wire_golden("structured-output-nested-openai", server.last_body)


def test_structured_output_nested_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.schema(wi.WIRE_STRUCTURED_OUTPUT_NESTED_SCHEMA).prompt(
                wi.WIRE_STRUCTURED_OUTPUT_NESTED_PROMPT
            )
        )
        assert server.last_body is not None
        assert (
            server.last_headers.get("anthropic-beta")
            == "structured-outputs-2025-11-13"
        )
        _assert_wire_golden("structured-output-nested-anthropic", server.last_body)


def test_caching_agent_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.agent.system(wi.WIRE_CACHING_SYSTEM).caching().prompt(wi.WIRE_CACHING_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("caching-agent-anthropic", server.last_body)


def test_caching_text_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.system(wi.WIRE_CACHING_SYSTEM).caching().prompt(wi.WIRE_CACHING_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("caching-text-anthropic", server.last_body)


def test_caching_batch_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.system(wi.WIRE_CACHING_SYSTEM).caching().submit_batch(wi.WIRE_CACHING_PROMPT)
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
            c.text.model(wi.WIRE_OPTIONS_OPENAI_GPT5_MODEL).max_tokens(wi.WIRE_OPTIONS_OPENAI_GPT5_MAX_TOKENS).reasoning_effort(wi.WIRE_OPTIONS_OPENAI_GPT5_REASONING_EFFORT).seed(wi.WIRE_OPTIONS_OPENAI_GPT5_SEED)
            .prompt(wi.WIRE_OPTIONS_OPENAI_GPT5_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("options-openai-gpt5", server.last_body)


def test_options_openai_o_series_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model(wi.WIRE_OPTIONS_OPENAI_O_SERIES_MODEL).max_tokens(wi.WIRE_OPTIONS_OPENAI_O_SERIES_MAX_TOKENS).reasoning_effort(wi.WIRE_OPTIONS_OPENAI_O_SERIES_REASONING_EFFORT).seed(wi.WIRE_OPTIONS_OPENAI_O_SERIES_SEED)
            .prompt(wi.WIRE_OPTIONS_OPENAI_O_SERIES_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("options-openai-o-series", server.last_body)


def test_options_openai_gpt4o_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model(wi.WIRE_OPTIONS_OPENAI_GPT4O_MODEL).max_tokens(wi.WIRE_OPTIONS_OPENAI_GPT4O_MAX_TOKENS).temperature(wi.WIRE_OPTIONS_OPENAI_GPT4O_TEMPERATURE).top_p(wi.WIRE_OPTIONS_OPENAI_GPT4O_TOP_P)
            .stop_sequences(wi.WIRE_OPTIONS_OPENAI_GPT4O_STOP_SEQUENCES).seed(wi.WIRE_OPTIONS_OPENAI_GPT4O_SEED)
            .frequency_penalty(wi.WIRE_OPTIONS_OPENAI_GPT4O_FREQUENCY_PENALTY).presence_penalty(wi.WIRE_OPTIONS_OPENAI_GPT4O_PRESENCE_PENALTY)
            .prompt(wi.WIRE_OPTIONS_OPENAI_GPT4O_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("options-openai-gpt4o", server.last_body)


def test_options_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model(wi.WIRE_OPTIONS_ANTHROPIC_MODEL).max_tokens(wi.WIRE_OPTIONS_ANTHROPIC_MAX_TOKENS).thinking_budget(wi.WIRE_OPTIONS_ANTHROPIC_THINKING_BUDGET)
            .stop_sequences(wi.WIRE_OPTIONS_ANTHROPIC_STOP_SEQUENCES)
            .prompt(wi.WIRE_OPTIONS_ANTHROPIC_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("options-anthropic", server.last_body)


def test_options_anthropic_plain_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model(wi.WIRE_OPTIONS_ANTHROPIC_PLAIN_MODEL)
            .max_tokens(wi.WIRE_OPTIONS_ANTHROPIC_PLAIN_MAX_TOKENS)
            .temperature(wi.WIRE_OPTIONS_ANTHROPIC_PLAIN_TEMPERATURE)
            .top_k(wi.WIRE_OPTIONS_ANTHROPIC_PLAIN_TOP_K)
            .stop_sequences(wi.WIRE_OPTIONS_ANTHROPIC_PLAIN_STOP_SEQUENCES)
            .prompt(wi.WIRE_OPTIONS_ANTHROPIC_PLAIN_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("options-anthropic-plain", server.last_body)


def test_options_anthropic_adaptive_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model(wi.WIRE_OPTIONS_ANTHROPIC_ADAPTIVE_MODEL)
            .max_tokens(wi.WIRE_OPTIONS_ANTHROPIC_ADAPTIVE_MAX_TOKENS)
            .reasoning_effort(wi.WIRE_OPTIONS_ANTHROPIC_ADAPTIVE_REASONING_EFFORT)
            .stop_sequences(wi.WIRE_OPTIONS_ANTHROPIC_ADAPTIVE_STOP_SEQUENCES)
            .prompt(wi.WIRE_OPTIONS_ANTHROPIC_ADAPTIVE_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("options-anthropic-adaptive", server.last_body)


def test_options_google_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = google("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model(wi.WIRE_OPTIONS_GOOGLE_MODEL).max_tokens(wi.WIRE_OPTIONS_GOOGLE_MAX_TOKENS).temperature(wi.WIRE_OPTIONS_GOOGLE_TEMPERATURE)
            .top_p(wi.WIRE_OPTIONS_GOOGLE_TOP_P).top_k(wi.WIRE_OPTIONS_GOOGLE_TOP_K).stop_sequences(wi.WIRE_OPTIONS_GOOGLE_STOP_SEQUENCES).seed(wi.WIRE_OPTIONS_GOOGLE_SEED)
            .reasoning_effort(wi.WIRE_OPTIONS_GOOGLE_REASONING_EFFORT)
            .safety_settings([
                SafetySetting(
                    category=wi.WIRE_OPTIONS_GOOGLE_SAFETY_CATEGORY,
                    threshold=wi.WIRE_OPTIONS_GOOGLE_SAFETY_THRESHOLD,
                )
            ])
            .prompt(wi.WIRE_OPTIONS_GOOGLE_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("options-google", server.last_body)


def test_options_google_gemini25_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = google("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model(wi.WIRE_OPTIONS_GOOGLE_GEMINI25_MODEL).max_tokens(wi.WIRE_OPTIONS_GOOGLE_GEMINI25_MAX_TOKENS).temperature(wi.WIRE_OPTIONS_GOOGLE_GEMINI25_TEMPERATURE)
            .thinking_budget(wi.WIRE_OPTIONS_GOOGLE_GEMINI25_THINKING_BUDGET)
            .prompt(wi.WIRE_OPTIONS_GOOGLE_GEMINI25_PROMPT)
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
            c.image.model(wi.WIRE_IMAGE_GEN_GOOGLE_FLASH_MODEL)
            .aspect_ratio(wi.WIRE_IMAGE_GEN_GOOGLE_FLASH_ASPECT_RATIO).image_size(wi.WIRE_IMAGE_GEN_GOOGLE_FLASH_IMAGE_SIZE)
            .generate(wi.WIRE_IMAGE_GEN_GOOGLE_FLASH_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("image-gen-google-flash", server.last_body)


def test_image_gen_google_pro_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = google("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.image.model(wi.WIRE_IMAGE_GEN_GOOGLE_PRO_MODEL)
            .aspect_ratio(wi.WIRE_IMAGE_GEN_GOOGLE_PRO_ASPECT_RATIO).image_size(wi.WIRE_IMAGE_GEN_GOOGLE_PRO_IMAGE_SIZE).include_text()
            .generate(wi.WIRE_IMAGE_GEN_GOOGLE_PRO_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("image-gen-google-pro", server.last_body)


def test_image_gen_openai_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.image.model(wi.WIRE_IMAGE_GEN_OPENAI_MODEL).image_size(wi.WIRE_IMAGE_GEN_OPENAI_IMAGE_SIZE).quality(wi.WIRE_IMAGE_GEN_OPENAI_QUALITY)
            .output_format(wi.WIRE_IMAGE_GEN_OPENAI_OUTPUT_FORMAT).background(wi.WIRE_IMAGE_GEN_OPENAI_BACKGROUND).count(wi.WIRE_IMAGE_GEN_OPENAI_COUNT)
            .generate(wi.WIRE_IMAGE_GEN_OPENAI_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("image-gen-openai", server.last_body)


def test_image_edit_google_flash_matches_shared_golden() -> None:
    png = base64.b64decode(wi.WIRE_IMAGE_EDIT_GOOGLE_FLASH_IMAGE_BASE64)
    with _CaptureServer(_CANNED_RESP) as server:
        c = google("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.image.model(wi.WIRE_IMAGE_EDIT_GOOGLE_FLASH_MODEL)
            .image(wi.WIRE_IMAGE_EDIT_GOOGLE_FLASH_IMAGE_MIME, png)
            .generate(wi.WIRE_IMAGE_EDIT_GOOGLE_FLASH_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("image-edit-google-flash", server.last_body)
