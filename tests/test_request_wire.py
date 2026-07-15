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
from llmkit import (
    audio_bytes,
    anthropic,
    bedrock,
    google,
    grok,
    minimax,
    openai,
    qwen,
    together,
    zhipu,
)
from llmkit.builders import vertex  # not re-exported at top level (caller-base provider)
from llmkit.builders import workersai  # not re-exported at top level (prompt 043)
from llmkit.builders import recraft  # not re-exported at top level (prompt 043)
from llmkit.builders import vidu  # not re-exported at top level (prompt 043)
from llmkit.builders import pixverse  # not re-exported at top level (prompt 043)
from llmkit.builders import inworld  # not re-exported at top level (ADR-049)
from llmkit.builders import assemblyai  # not re-exported at top level (ADR-048)
from llmkit import audio  # transcription audio Part constructor (ADR-048)
from llmkit.types import SafetySetting, Tool
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


def _assert_wire_headers(fixture: str, headers: dict[str, str]) -> None:
    """Drop the per-SDK request-header artifact (lowercased keys) for the
    cross-SDK comparator's opt-in header subset-match (HANDOFF-028), closing
    BUG-017's deferred golden header lock. A fixture with a companion
    <fixture>.headers.json golden has each named header asserted value-equal
    across all four SDKs."""
    artifact = ARTIFACT_ROOT / fixture / "python.headers.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    flat = {k.lower(): v for k, v in headers.items()}
    artifact.write_text(json.dumps(flat, indent=2))


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
    "request_id": "vid_test",  # VID-007: Grok video-submit handle id
    "task_id": "vid_test",  # VideoMinimax: top-level task_id submit handle
    "name": "models/veo-test/operations/op_test",  # VideoVeo: operation-name submit handle
    "invocationArn": "arn:aws:bedrock:us-east-1:0:async-invoke/vid_test",  # VideoBedrock: invocationArn submit handle
    "output": {"task_id": "vid_test", "task_status": "PENDING"},  # VideoQwen: output.task_id submit handle
    "Resp": {"video_id": 318633193768896},  # VideoPixVerse: Resp.video_id submit handle (numeric)
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
    "audioContent": wi.WIRE_IMAGE_EDIT_GOOGLE_FLASH_IMAGE_BASE64,  # SpeechInworld: base64 synthesized audio
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
        _assert_wire_headers("structured-output-anthropic", server.last_headers)


def test_anthropic_schema_document_composes_both_betas() -> None:
    # BUG-017/HANDOFF-028: schema + file id in one request composes the
    # structured-output beta and the files-api beta into one anthropic-beta.
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model(wi.WIRE_ANTHROPIC_SCHEMA_DOCUMENT_MODEL)
            .schema(wi.WIRE_ANTHROPIC_SCHEMA_DOCUMENT_SCHEMA)
            .file(wi.WIRE_ANTHROPIC_SCHEMA_DOCUMENT_FILE_ID)
            .prompt(wi.WIRE_ANTHROPIC_SCHEMA_DOCUMENT_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("anthropic-schema-document", server.last_body)
        _assert_wire_headers("anthropic-schema-document", server.last_headers)


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
            c.text.system(wi.WIRE_CACHING_SYSTEM).caching().batch(wi.WIRE_CACHING_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("caching-batch-anthropic", server.last_body)


def test_batch_multimodal_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        data = base64.b64decode(wi.WIRE_BATCH_MULTIMODAL_ANTHROPIC_IMAGE_BASE64)
        asyncio.run(
            c.text.model(wi.WIRE_BATCH_MULTIMODAL_ANTHROPIC_MODEL)
            .image(wi.WIRE_BATCH_MULTIMODAL_ANTHROPIC_IMAGE_MIME, data)
            .file(wi.WIRE_BATCH_MULTIMODAL_ANTHROPIC_FILE_ID)
            .batch(wi.WIRE_BATCH_MULTIMODAL_ANTHROPIC_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("batch-multimodal-anthropic", server.last_body)
        # Referencing an uploaded file id in a batch item requires the files-api
        # beta on the batch CREATE request (batch-modality witness) — golden-locked
        # across all four SDKs via batch-multimodal-anthropic.headers.json.
        assert (
            server.last_headers.get("anthropic-beta") == "files-api-2025-04-14"
        )
        _assert_wire_headers("batch-multimodal-anthropic", server.last_headers)


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


# BUG-028: stream_options.include_usage on the OpenAI streaming request body.
def test_stream_openai_matches_shared_golden() -> None:
    async def _drive(c) -> None:
        stream = c.text.model(wi.WIRE_STREAM_OPENAI_MODEL).stream(wi.WIRE_STREAM_OPENAI_PROMPT)
        async for _ in stream:
            pass

    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(_drive(c))
        assert server.last_body is not None
        _assert_wire_golden("stream-openai", server.last_body)


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


def test_anthropic_text_document_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model(wi.WIRE_ANTHROPIC_TEXT_DOCUMENT_MODEL)
            .file(wi.WIRE_ANTHROPIC_TEXT_DOCUMENT_FILE_ID)
            .prompt(wi.WIRE_ANTHROPIC_TEXT_DOCUMENT_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("anthropic-text-document", server.last_body)
        # BUG-017 / HANDOFF-028: the Files API beta must ride on the Messages
        # request referencing an uploaded file — golden-locked across all four
        # SDKs via the companion anthropic-text-document.headers.json.
        _assert_wire_headers("anthropic-text-document", server.last_headers)


def test_openai_text_document_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model(wi.WIRE_OPENAI_TEXT_DOCUMENT_MODEL)
            .file(wi.WIRE_OPENAI_TEXT_DOCUMENT_FILE_ID)
            .prompt(wi.WIRE_OPENAI_TEXT_DOCUMENT_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("openai-text-document", server.last_body)


def test_anthropic_text_image_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        data = base64.b64decode(wi.WIRE_ANTHROPIC_TEXT_IMAGE_IMAGE_BASE64)
        asyncio.run(
            c.text.model(wi.WIRE_ANTHROPIC_TEXT_IMAGE_MODEL)
            .image(wi.WIRE_ANTHROPIC_TEXT_IMAGE_IMAGE_MIME, data)
            .prompt(wi.WIRE_ANTHROPIC_TEXT_IMAGE_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("anthropic-text-image", server.last_body)


def test_openai_text_image_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        data = base64.b64decode(wi.WIRE_OPENAI_TEXT_IMAGE_IMAGE_BASE64)
        asyncio.run(
            c.text.model(wi.WIRE_OPENAI_TEXT_IMAGE_MODEL)
            .image(wi.WIRE_OPENAI_TEXT_IMAGE_IMAGE_MIME, data)
            .prompt(wi.WIRE_OPENAI_TEXT_IMAGE_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("openai-text-image", server.last_body)


def test_google_text_image_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = google("key")
        c.provider.base_url = server.url
        data = base64.b64decode(wi.WIRE_GOOGLE_TEXT_IMAGE_IMAGE_BASE64)
        asyncio.run(
            c.text.model(wi.WIRE_GOOGLE_TEXT_IMAGE_MODEL)
            .image(wi.WIRE_GOOGLE_TEXT_IMAGE_IMAGE_MIME, data)
            .prompt(wi.WIRE_GOOGLE_TEXT_IMAGE_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("google-text-image", server.last_body)


def test_bedrock_text_image_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = bedrock("key")
        c.provider.base_url = server.url
        data = base64.b64decode(wi.WIRE_BEDROCK_TEXT_IMAGE_IMAGE_BASE64)
        asyncio.run(
            c.text.model(wi.WIRE_BEDROCK_TEXT_IMAGE_MODEL)
            .image(wi.WIRE_BEDROCK_TEXT_IMAGE_IMAGE_MIME, data)
            .prompt(wi.WIRE_BEDROCK_TEXT_IMAGE_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("bedrock-text-image", server.last_body)


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


def test_image_gen_recraft_matches_shared_golden() -> None:
    # Recraft generations JSON body (JSONGenerations shape): {model, prompt,
    # size, n} plus the forced response_format=b64_json (Recraft defaults to
    # URL delivery; the SDK forces b64_json for a uniform decode path).
    with _CaptureServer(_CANNED_RESP) as server:
        c = recraft("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.image.model(wi.WIRE_IMAGE_GEN_RECRAFT_MODEL).image_size(wi.WIRE_IMAGE_GEN_RECRAFT_IMAGE_SIZE).count(wi.WIRE_IMAGE_GEN_RECRAFT_COUNT)
            .generate(wi.WIRE_IMAGE_GEN_RECRAFT_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("image-gen-recraft", server.last_body)


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


# === ADR-034 / VID-007: video generation submit body ===


def test_video_grok_matches_shared_golden() -> None:
    # Grok video-submit body {model, prompt}. The async VideoHandle is
    # discarded — only the outbound submit bytes are asserted. The canned
    # response carries request_id so submit parses a handle.
    with _CaptureServer(_CANNED_RESP) as server:
        c = grok("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.video.model(wi.WIRE_VIDEO_GROK_MODEL).submit(wi.WIRE_VIDEO_GROK_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("video-grok", server.last_body)


def test_video_grok_i2v_matches_shared_golden() -> None:
    # BUG-010: Grok image-to-video submit body {model, prompt, image:{url}}.
    # The seed frame inlines as a data URL at image.url (the Grok image-EDIT
    # encoding); the text-to-video golden above has no image field.
    seed = base64.b64decode(wi.WIRE_VIDEO_GROK_I2V_IMAGE_BASE64)
    with _CaptureServer(_CANNED_RESP) as server:
        c = grok("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.video.model(wi.WIRE_VIDEO_GROK_I2V_MODEL)
            .image(wi.WIRE_VIDEO_GROK_I2V_IMAGE_MIME, seed)
            .submit(wi.WIRE_VIDEO_GROK_I2V_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("video-grok-i2v", server.last_body)


def test_video_zhipu_matches_shared_golden() -> None:
    # Zhipu CogVideoX video-submit body {model, prompt} — structurally
    # identical to Grok's (the shared {model, prompt} arm); the lifecycle
    # divergence is delivery-side, covered by the unit tests.
    with _CaptureServer(_CANNED_RESP) as server:
        c = zhipu("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.video.model(wi.WIRE_VIDEO_ZHIPU_MODEL).submit(wi.WIRE_VIDEO_ZHIPU_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("video-zhipu", server.last_body)


def test_video_vidu_matches_shared_golden() -> None:
    # Vidu (Shengshu) video-submit body {model, prompt} — structurally
    # identical to Grok's/Zhipu's (the shared {model, prompt} arm); the
    # lifecycle divergence is delivery-side, covered by the unit tests.
    with _CaptureServer(_CANNED_RESP) as server:
        c = vidu("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.video.model(wi.WIRE_VIDEO_VIDU_MODEL).submit(wi.WIRE_VIDEO_VIDU_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("video-vidu", server.last_body)


def test_speech_inworld_matches_shared_golden() -> None:
    # Inworld text-to-speech body {text, voiceId, modelId, audioConfig,
    # deliveryMode} (ADR-049 SPK-007).
    with _CaptureServer(_CANNED_RESP) as server:
        c = inworld("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.speech.model(wi.WIRE_SPEECH_INWORLD_MODEL)
            .voice(wi.WIRE_SPEECH_INWORLD_VOICE)
            .generate(wi.WIRE_SPEECH_INWORLD_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("speech-inworld", server.last_body)


def test_speech_openai_matches_shared_golden() -> None:
    # OpenAI text-to-speech body {model, input, voice, response_format}
    # (ADR-051). The response is raw audio bytes; only the outbound request
    # bytes are asserted here.
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.speech.model(wi.WIRE_SPEECH_OPENAI_MODEL)
            .voice(wi.WIRE_SPEECH_OPENAI_VOICE)
            .generate(wi.WIRE_SPEECH_OPENAI_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("speech-openai", server.last_body)


def test_transcription_assemblyai_matches_shared_golden() -> None:
    # AssemblyAI transcription submit body {audio_url} (ADR-048). The async
    # TranscriptionHandle is discarded; only the outbound submit bytes are
    # asserted. The upload hop is bytes-only and is not exercised here (URL
    # part skips it).
    with _CaptureServer(_CANNED_RESP) as server:
        c = assemblyai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.transcription.submit(
                [audio(wi.WIRE_TRANSCRIPTION_ASSEMBLYAI_AUDIO_U_R_L)]
            )
        )
        assert server.last_body is not None
        _assert_wire_golden("transcription-assemblyai", server.last_body)


class _MultipartCaptureServer:
    """Records the outbound multipart/form-data body and decodes it into the
    canonical descriptor (ADR-051 OQ-3): ordered fields, the file part keeping
    filename + content-type with a fixed bytes placeholder."""

    def __init__(self, response_body: dict[str, Any]):
        self.descriptor: dict[str, Any] | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def do_POST(self):
                import email

                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                ctype = self.headers.get("Content-Type", "")
                msg = email.message_from_bytes(
                    b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw
                )
                fields: list[dict[str, str]] = []
                for part in msg.get_payload():
                    name = part.get_param("name", header="content-disposition")
                    fn = part.get_filename()
                    if fn:
                        fields.append(
                            {
                                "name": name,
                                "filename": fn,
                                "contentType": part.get_content_type(),
                                "bytes": "<audio-bytes>",
                            }
                        )
                    else:
                        fields.append(
                            {
                                "name": name,
                                "value": part.get_payload(decode=True).decode(),
                            }
                        )
                outer.descriptor = {
                    "_encoding": "multipart/form-data",
                    "fields": fields,
                }
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


def test_transcription_openai_matches_shared_golden() -> None:
    # OpenAI SYNCHRONOUS transcription — the first multipart/form-data request
    # body (ADR-051). The golden is the canonical multipart descriptor (OQ-3);
    # the driver decodes its actual encoded multipart body into ordered fields.
    with _MultipartCaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.transcription.model(wi.WIRE_TRANSCRIPTION_OPENAI_MODEL).transcribe(
                [audio_bytes(wi.WIRE_TRANSCRIPTION_OPENAI_AUDIO_MIME, b"fake-audio")]
            )
        )
        assert server.descriptor is not None
        _assert_wire_golden("transcription-openai", server.descriptor)


def test_video_pixverse_matches_shared_golden() -> None:
    # PixVerse video-submit body {model, prompt, duration, quality,
    # aspect_ratio} — the dedicated PixVerse arm (all five fields required);
    # the dynamic Ai-trace-id header is omitted from the golden (it is a
    # per-request UUID) and asserted in the lifecycle unit tests.
    with _CaptureServer(_CANNED_RESP) as server:
        c = pixverse("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.video.model(wi.WIRE_VIDEO_PIXVERSE_MODEL).submit(
                wi.WIRE_VIDEO_PIXVERSE_PROMPT
            )
        )
        assert server.last_body is not None
        _assert_wire_golden("video-pixverse", server.last_body)


def test_video_together_matches_shared_golden() -> None:
    # Together video-submit body {model, prompt} — structurally identical to
    # Grok's/Zhipu's (the shared {model, prompt} arm); the lifecycle
    # divergence is delivery-side, covered by the unit tests.
    with _CaptureServer(_CANNED_RESP) as server:
        c = together("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.video.model(wi.WIRE_VIDEO_TOGETHER_MODEL).submit(
                wi.WIRE_VIDEO_TOGETHER_PROMPT
            )
        )
        assert server.last_body is not None
        _assert_wire_golden("video-together", server.last_body)


def test_video_qwen_matches_shared_golden() -> None:
    # Qwen (DashScope) video-submit body is the NESTED {model, input:{prompt}}
    # shape — the first divergent submit body. Also asserts the load-bearing
    # X-DashScope-Async: enable header in-driver (mirrors the Anthropic
    # beta-header assert).
    with _CaptureServer(_CANNED_RESP) as server:
        c = qwen("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.video.model(wi.WIRE_VIDEO_QWEN_MODEL).submit(wi.WIRE_VIDEO_QWEN_PROMPT)
        )
        assert server.last_body is not None
        assert server.last_headers.get("x-dashscope-async") == "enable"
        _assert_wire_golden("video-qwen", server.last_body)


def test_video_minimax_matches_shared_golden() -> None:
    # MiniMax video-submit body is the shared {model, prompt}. The two-hop
    # result (poll file_id -> file-retrieve download_url) is delivery-side,
    # covered by the unit tests.
    with _CaptureServer(_CANNED_RESP) as server:
        c = minimax("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.video.model(wi.WIRE_VIDEO_MINIMAX_MODEL).submit(
                wi.WIRE_VIDEO_MINIMAX_PROMPT
            )
        )
        assert server.last_body is not None
        _assert_wire_golden("video-minimax", server.last_body)


def test_video_veo_matches_shared_golden() -> None:
    # Google Veo video-submit body is the nested {instances:[{prompt}]} shape —
    # the first video-submit body with NO model field, because Veo carries the
    # model in the submit PATH (/v1beta/models/{model}:predictLongRunning). The
    # LRO lifecycle and ?key= query-param auth are delivery-side, covered by the
    # unit tests.
    with _CaptureServer(_CANNED_RESP) as server:
        c = google("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.video.model(wi.WIRE_VIDEO_GOOGLE_MODEL).submit(
                wi.WIRE_VIDEO_GOOGLE_PROMPT
            )
        )
        assert server.last_body is not None
        _assert_wire_golden("video-google", server.last_body)


def test_video_bedrock_matches_shared_golden() -> None:
    # Bedrock Nova Reel video-submit body is the nested {modelId, modelInput:
    # {taskType, textToVideoParams:{text}}, outputDataConfig:{s3OutputDataConfig:
    # {s3Uri}}} shape — the first video-submit body that carries the model in the
    # body AND a caller output S3 URI, and the first SigV4-signed video submit.
    # The ARN-signing/poll lifecycle is delivery-side, covered by the unit tests.
    with _CaptureServer(_CANNED_RESP) as server:
        c = bedrock("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.video.model(wi.WIRE_VIDEO_BEDROCK_MODEL)
            .output_uri("s3://llmkit-wire-fixtures/out/")
            .submit(wi.WIRE_VIDEO_BEDROCK_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("video-bedrock", server.last_body)


def test_video_vertex_matches_shared_golden() -> None:
    # Vertex Veo video-submit body is the nested {instances:[{prompt}]} shape —
    # byte-identical to the Veo golden (model in the PATH, not the body). The
    # POST-poll lifecycle (:fetchPredictOperation, inline-base64 download
    # delivery) is delivery-side, covered by the unit tests.
    with _CaptureServer(_CANNED_RESP) as server:
        c = vertex("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.video.model(wi.WIRE_VIDEO_VERTEX_MODEL).submit(
                wi.WIRE_VIDEO_VERTEX_PROMPT
            )
        )
        assert server.last_body is not None
        _assert_wire_golden("video-vertex", server.last_body)


# === Prompt 043: Cloudflare Workers AI (OpenAI-compatible chat) ===


def test_workersai_matches_shared_golden() -> None:
    # Workers AI's OpenAI-compatible chat-completions body {model, messages,
    # max_tokens, temperature, top_p} — structurally identical to the gpt-4o
    # options golden (OpenAI ArgsFormat, system-in-messages); the novel bit
    # (account-id-in-URL) is delivery-side, not request-body-side.
    with _CaptureServer(_CANNED_RESP) as server:
        c = workersai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.model(wi.WIRE_WORKERSAI_MODEL)
            .max_tokens(wi.WIRE_WORKERSAI_MAX_TOKENS)
            .temperature(wi.WIRE_WORKERSAI_TEMPERATURE)
            .top_p(wi.WIRE_WORKERSAI_TOP_P)
            .prompt(wi.WIRE_WORKERSAI_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("workersai", server.last_body)


# === ADR-055 Phase B: OpenAI Responses chat protocol ===


def test_responses_openai_matches_shared_golden() -> None:
    # OpenAI Responses body: the SAME flat message array as Chat Completions but
    # under "input" (not "messages"), max_tokens renamed to max_output_tokens,
    # POSTed to /v1/responses. Text.protocol("responses") is the opt-in. The
    # default Chat Completions goldens are untouched (default pinned).
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.protocol("responses")
            .model(wi.WIRE_RESPONSES_OPENAI_MODEL)
            .max_tokens(wi.WIRE_RESPONSES_OPENAI_MAX_TOKENS)
            .prompt(wi.WIRE_RESPONSES_OPENAI_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("responses-openai", server.last_body)


# === TASK-002: tool-definition fixtures across the four chat wire families ===
#
# Each driver builds the single canonical tool (name/description/schema from the
# generated wire-input consts), registers it on the agent via add_tool, and
# prompts it. The mock returns a plain text response, so the agent loop makes one
# request (carrying the tool defs) and terminates. Mirrors the Go drivers
# (TestRequestWire_ToolDef* / wireToolDef). NOT live-anchored — parity held by
# the cross-SDK comparator + mock body, like the keyless providers.


def _wire_tool_def() -> Tool:
    # The Run stub is never invoked: the mock returns plain text, so the agent
    # loop sends one request carrying the tool defs and terminates.
    return Tool(
        name=wi.WIRE_TOOL_TOOL_NAME,
        description=wi.WIRE_TOOL_TOOL_DESCRIPTION,
        schema=json.loads(wi.WIRE_TOOL_TOOL_SCHEMA),
        run=lambda _args: "",
    )


def test_tooldef_openai_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = openai("key")
        c.provider.base_url = server.url
        asyncio.run(c.agent.add_tool(_wire_tool_def()).prompt(wi.WIRE_TOOL_PROMPT))
        assert server.last_body is not None
        _assert_wire_golden("tooldef-openai", server.last_body)


def test_tooldef_anthropic_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = anthropic("key")
        c.provider.base_url = server.url
        asyncio.run(c.agent.add_tool(_wire_tool_def()).prompt(wi.WIRE_TOOL_PROMPT))
        assert server.last_body is not None
        _assert_wire_golden("tooldef-anthropic", server.last_body)


def test_tooldef_google_matches_shared_golden() -> None:
    with _CaptureServer(_CANNED_RESP) as server:
        c = google("key")
        c.provider.base_url = server.url
        asyncio.run(c.agent.add_tool(_wire_tool_def()).prompt(wi.WIRE_TOOL_PROMPT))
        assert server.last_body is not None
        _assert_wire_golden("tooldef-google", server.last_body)


def test_tooldef_bedrock_matches_shared_golden() -> None:
    # Bedrock captures the body before the SigV4 auth check, so a dummy key works.
    with _CaptureServer(_CANNED_RESP) as server:
        c = bedrock("key")
        c.provider.base_url = server.url
        asyncio.run(c.agent.add_tool(_wire_tool_def()).prompt(wi.WIRE_TOOL_PROMPT))
        assert server.last_body is not None
        _assert_wire_golden("tooldef-bedrock", server.last_body)


def test_bedrock_chat_matches_shared_golden() -> None:
    # Text prompt to Bedrock (no tools) — the Converse message-transform arm,
    # plus the inferenceConfig option surface (maxTokens/temperature/top_p/
    # stopSequences). Mirrors Go's TestRequestWire_BedrockChat.
    with _CaptureServer(_CANNED_RESP) as server:
        c = bedrock("key")
        c.provider.base_url = server.url
        asyncio.run(
            c.text.max_tokens(wi.WIRE_BEDROCK_CHAT_MAX_TOKENS)
            .temperature(wi.WIRE_BEDROCK_CHAT_TEMPERATURE)
            .top_p(wi.WIRE_BEDROCK_CHAT_TOP_P)
            .stop_sequences(wi.WIRE_BEDROCK_CHAT_STOP_SEQUENCES)
            .prompt(wi.WIRE_BEDROCK_CHAT_PROMPT)
        )
        assert server.last_body is not None
        _assert_wire_golden("bedrock-chat", server.last_body)
