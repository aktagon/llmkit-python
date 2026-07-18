"""Cross-SDK RESPONSE-body conformance driver — Python (ADR-065 / prompt 045).

Sibling of test_lifecycle_wire.py, on the response-BODY side. Where the lifecycle
suite asserts the poll CLASSIFICATION agrees across SDKs, this asserts the body
PARSE agrees: given the same anchored provider reply, every SDK's public
c.text.prompt() normalizes it to the SAME projection (Usage dims + finish reason
+ content). Response parsing is handwritten per SDK (ADR-028 behavior, not
generated data); this is its parity floor.

The parser INPUT lives at codegen/testdata/wire/response/v1/bodies/<shape>.json;
this driver serves it verbatim from a single-hop mock, drives one prompt, projects
the parsed Response, drops target/wire/response/<shape>/python.json, and asserts it
value-equals the EXPECTED golden codegen/testdata/wire/response/v1/<shape>.json.
codegen/test_cross_sdk_response.py compares all four SDK artifacts to that golden.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from llmkit.builders import new_client
from llmkit.builders.batch import BatchHandle
from llmkit.image import audio_bytes
from llmkit.types import Provider
from llmkit.providers.generated.models_parsers import (
    ParsedModelsPage,
    parse_anthropic_models_response,
    parse_google_models_response,
    parse_openai_cohort_models_response,
)
from llmkit.structs import ImageResponse, Response, SpeechResponse, TranscriptionResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
BODY_DIR = REPO_ROOT / "codegen" / "testdata" / "wire" / "response" / "v1" / "bodies"
GOLDEN_DIR = REPO_ROOT / "codegen" / "testdata" / "wire" / "response" / "v1"
ARTIFACT_ROOT = REPO_ROOT / "target" / "wire" / "response"


class _ResponseMockServer:
    """Serves the anchored provider reply verbatim on any method/path — the parse
    path is single-hop, so a catch-all is enough. The parser dispatches on the
    client's provider, not the URL."""

    def __init__(self, body: bytes, content_type: str = "application/json") -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # chat requests are POST
                self._send()

            def do_GET(self):
                self._send()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_ResponseMockServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def _artifact_from(resp: Response) -> dict:
    """Normalized, cross-SDK-comparable projection of a parsed Response — the
    contract-bearing parse output only (Usage dims + finish reason + content)."""
    u = resp.usage
    return {
        "usage": {
            "input": u.input,
            "output": u.output,
            "cacheRead": u.cache_read,
            "cacheWrite": u.cache_write,
            "reasoning": u.reasoning,
            "cost": u.cost,
        },
        "finishReason": resp.finish_reason,
        "content": resp.text,
        "error": None,
    }


def _image_artifact_from(resp: ImageResponse) -> dict:
    """Projection for image responses. Content is the media discriminant
    {kind,mimeType,byteLen,count} (RWR-004) — the four SDKs must agree the same
    body decodes to the same images (the BUG-024 parse-drift class)."""
    first = resp.images[0] if resp.images else None
    u = resp.usage
    return {
        "usage": {
            "input": u.input,
            "output": u.output,
            "cacheRead": u.cache_read,
            "cacheWrite": u.cache_write,
            "reasoning": u.reasoning,
            "cost": u.cost,
        },
        "finishReason": resp.finish_reason,
        "content": {
            "kind": "image",
            "mimeType": first.mime_type if first else "",
            "byteLen": len(first.bytes) if first else 0,
            "count": len(resp.images),
        },
        "error": None,
    }


def _speech_artifact_from(resp: SpeechResponse) -> dict:
    """Projection for speech (TTS) responses — the media discriminant
    {kind,mimeType,byteLen} (the ADR-018 bytes/mime accessor contract)."""
    u = resp.usage
    return {
        "usage": {
            "input": u.input,
            "output": u.output,
            "cacheRead": u.cache_read,
            "cacheWrite": u.cache_write,
            "reasoning": u.reasoning,
            "cost": u.cost,
        },
        "finishReason": "",
        "content": {
            "kind": "speech",
            "mimeType": resp.audio.mime_type,
            "byteLen": len(resp.audio.bytes),
        },
        "error": None,
    }


def _transcript_artifact_from(resp: TranscriptionResponse) -> dict:
    """Projection for transcription (STT) responses — {kind,text,segments}."""
    u = resp.usage
    return {
        "usage": {
            "input": u.input,
            "output": u.output,
            "cacheRead": u.cache_read,
            "cacheWrite": u.cache_write,
            "reasoning": u.reasoning,
            "cost": u.cost,
        },
        "finishReason": "",
        "content": {
            "kind": "transcript",
            "text": resp.text,
            "segments": len(resp.segments),
        },
        "error": None,
    }


def _models_artifact_from(page: ParsedModelsPage) -> dict:
    """Projection for catalogue (/models) responses. Content is the catalogue
    discriminant {kind:"models", count, firstId, lastId, nextCursor, first{...}}
    (ADR-067 Fix B) — the same body must decode to the same model list +
    pagination cursor across all five SDKs. No usage / finishReason: a catalogue
    is not a generation response."""
    first = page.records[0] if page.records else None
    last = page.records[-1] if page.records else None
    return {
        "content": {
            "count": len(page.records),
            "first": {
                "contextWindow": first.context_window if first else 0,
                "displayName": first.display_name if first else "",
                "maxOutput": first.max_output if first else 0,
            },
            "firstId": first.id if first else "",
            "kind": "models",
            "lastId": last.id if last else "",
            "nextCursor": page.next_cursor,
        },
        "error": None,
    }


def _write_and_assert(shape: str, artifact: dict) -> None:
    out_dir = ARTIFACT_ROOT / shape
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "python.json").write_text(json.dumps(artifact, indent=2))

    golden = json.loads((GOLDEN_DIR / f"{shape}.json").read_text())
    assert artifact == golden


def _run_fixture(shape: str, provider: str) -> None:
    body = (BODY_DIR / f"{shape}.json").read_bytes()
    with _ResponseMockServer(body) as server:
        c = new_client(provider, "k")
        c.provider.base_url = server.url
        resp = asyncio.run(c.text.prompt("ping"))
    _write_and_assert(shape, _artifact_from(resp))


def _run_image_fixture(shape: str, provider: str, model: str) -> None:
    body = (BODY_DIR / f"{shape}.json").read_bytes()
    with _ResponseMockServer(body) as server:
        c = new_client(provider, "k")
        c.provider.base_url = server.url
        resp = asyncio.run(c.image.model(model).generate("a cat"))
    _write_and_assert(shape, _image_artifact_from(resp))


def _run_speech_fixture(shape: str, provider: str, model: str, voice: str) -> None:
    body = (BODY_DIR / f"{shape}.json").read_bytes()
    with _ResponseMockServer(body) as server:
        c = new_client(provider, "k")
        c.provider.base_url = server.url
        resp = asyncio.run(c.speech.model(model).voice(voice).generate("hello"))
    _write_and_assert(shape, _speech_artifact_from(resp))


def _run_transcript_fixture(shape: str, provider: str, model: str) -> None:
    body = (BODY_DIR / f"{shape}.json").read_bytes()
    with _ResponseMockServer(body) as server:
        c = new_client(provider, "k")
        c.provider.base_url = server.url
        resp = asyncio.run(
            c.transcription.model(model).transcribe([audio_bytes("audio/wav", b"RIFF")])
        )
    _write_and_assert(shape, _transcript_artifact_from(resp))


def _run_stream_fixture(shape: str, provider: str) -> None:
    """B-stream: drive the real streaming path against the SSE mock, drain the
    async chunk iterator, then project the trailing handle's accumulated Response
    — the same projection as the sync chat artifact. Data-only SSE only (OpenAI /
    Google); Anthropic event-typed stream deferred (see PROVENANCE.md)."""
    body = (BODY_DIR / f"{shape}.sse").read_bytes()
    with _ResponseMockServer(body, content_type="text/event-stream") as server:
        c = new_client(provider, "k")
        c.provider.base_url = server.url
        stream = c.text.stream("ping")

        async def _drain() -> None:
            async for _ in stream:
                pass

        asyncio.run(_drain())
        resp = stream.response
    assert resp is not None
    _write_and_assert(shape, _artifact_from(resp))


def _run_models_fixture(shape, parse) -> None:
    """Catalogue parse seam is driven DIRECTLY (no HTTP path): feed the anchored
    /models body to the handwritten parser and project the ParsedModelsPage."""
    body = (BODY_DIR / f"{shape}.json").read_bytes()
    _write_and_assert(shape, _models_artifact_from(parse(body)))


class _BatchResultsMockServer:
    """Two-hop Anthropic batch mock (HANDOFF-036 A1): GET the batch status ->
    processing_status "ended", then GET .../results -> the anchored JSONL
    results file verbatim. Mirror of test_lifecycle_wire.py's mock, on the
    Anthropic single-endpoint shape."""

    def __init__(self, results: bytes) -> None:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self, payload: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path.endswith("/results"):
                    return self._send(results)
                if self.path.startswith("/v1/messages/batches/"):
                    return self._send(
                        json.dumps({"id": "batch_1", "processing_status": "ended"}).encode("utf-8")
                    )
                self.send_response(404)
                self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_BatchResultsMockServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def _batch_results_artifact(responses: list[Response]) -> dict:
    """Projection for a completed batch's RESULTS file parse (HANDOFF-036 A1):
    {kind:"batch_results", count, first{finishReason, text, usage}} — count is
    the assertion: the errored line is SKIPPED and the successful subset
    returned, never a thrown-away completed batch."""
    first: dict = {}
    if responses:
        r = responses[0]
        first = {
            "finishReason": r.finish_reason,
            "text": r.text,
            "usage": {
                "input": r.usage.input,
                "output": r.usage.output,
                "cacheRead": r.usage.cache_read,
                "cacheWrite": r.usage.cache_write,
                "reasoning": r.usage.reasoning,
                "cost": r.usage.cost,
            },
        }
    return {
        "content": {
            "count": len(responses),
            "first": first,
            "kind": "batch_results",
        },
        "error": None,
    }


def _run_batch_results_fixture(shape: str) -> None:
    """Drive the real public path: BatchHandle.poll against the two-hop mock.
    The parser INPUT is bodies/<shape>.jsonl (a JSONL results file — one
    succeeded line + one errored line; Anthropic result.type=errored carries no
    result.message). Known shared assumption (PROVENANCE.md): no SDK matches
    results by custom_id — all assume file line order."""
    results = (BODY_DIR / f"{shape}.jsonl").read_bytes()
    with _BatchResultsMockServer(results) as server:
        handle = BatchHandle(
            id="batch_1",
            provider=Provider(name="anthropic", api_key="test-key", base_url=server.url),
        )
        st = asyncio.run(handle.poll())
    assert st.result is not None, f"expected a succeeded result, got state {st.state}"
    _write_and_assert(shape, _batch_results_artifact(st.result))


def test_response_chat_openai() -> None:
    _run_fixture("chat-openai", "openai")


def test_response_chat_anthropic() -> None:
    _run_fixture("chat-anthropic", "anthropic")


def test_response_chat_google() -> None:
    _run_fixture("chat-google", "google")


# Phase 2: image response dispatch (BUG-024 surface) — one golden per
# llm:imageResponseShape (GoogleParts / DataArrayB64Json / VertexPredictions).
def test_response_image_google() -> None:
    _run_image_fixture("image-google", "google", "gemini-3.1-flash-image-preview")


def test_response_image_openai() -> None:
    _run_image_fixture("image-openai", "openai", "gpt-image-1")


def test_response_image_vertex() -> None:
    _run_image_fixture("image-vertex", "vertex", "imagen-3.0-generate-002")


# Speech (TTS) + transcription (STT) — the media/transcript accessor contract.
def test_response_speech_inworld() -> None:
    _run_speech_fixture("speech-inworld", "inworld", "inworld-tts-2", "Dennis")


def test_response_transcription_openai() -> None:
    _run_transcript_fixture("transcription-openai", "openai", "whisper-1")


# B-stream: streaming (SSE) response parity — data-only shapes.
def test_response_stream_openai() -> None:
    _run_stream_fixture("stream-openai", "openai")


def test_response_stream_google() -> None:
    _run_stream_fixture("stream-google", "google")


# Batch results parse (HANDOFF-036 A1) — errored line skipped, successful
# subset returned.
def test_response_batch_results_anthropic() -> None:
    _run_batch_results_fixture("batch-results-anthropic")


# Catalogue (/models) response parity (ADR-067 Fix B) — one golden per provider
# parse shape (anthropic cursor / openai-cohort / google cursor).
def test_response_models_anthropic() -> None:
    _run_models_fixture("models-anthropic", parse_anthropic_models_response)


def test_response_models_openai() -> None:
    _run_models_fixture("models-openai", parse_openai_cohort_models_response)


def test_response_models_google() -> None:
    _run_models_fixture("models-google", parse_google_models_response)
