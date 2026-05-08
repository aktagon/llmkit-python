"""Phase 2b smoke tests for llmkit.builders.

Exercises every public symbol — chains, immutability, terminal stubs,
type-alias re-exports — so the strict (eventual) Python coverage gate
sees full function coverage on builders/__init__.py.
"""

from __future__ import annotations

import asyncio

import pytest

from llmkit.builders import (
    Agent,
    BatchHandle,
    Client,
    File,
    Image,
    ImageData,
    ImageResponse,
    MediaRef,
    Message,
    MiddlewareFn,
    Part,
    Response,
    Text,
    Tool,
    Upload,
    ai21,
    anthropic,
    azure,
    bedrock,
    cerebras,
    cohere,
    deepseek,
    doubao,
    ernie,
    fireworks,
    google,
    grok,
    groq,
    lmstudio,
    minimax,
    mistral,
    moonshot,
    new_client,
    ollama,
    openai,
    openrouter,
    perplexity,
    qwen,
    sambanova,
    together,
    vllm,
    yi,
    zhipu,
)


def noop_middleware(_ctx: object, _event: object) -> Exception | None:
    return None


# ---------- chain methods land in fields ----------


def test_text_chain() -> None:
    c = google("k")
    assert isinstance(c, Client)
    assert isinstance(c.text, Text)
    assert isinstance(c.image, Image)
    assert isinstance(c.agent, Agent)
    assert isinstance(c.upload, Upload)

    text = (
        c.text.caching()
        .file("file-id")
        .history(Message(role="user", content="earlier"))
        .image("image/png", b"\xff")
        .max_tokens(42)
        .middleware(noop_middleware)
        .model("text-model")
        .schema('{"type":"object"}')
        .system("you are a tutor")
        .temperature(0.7)
        .text("hello")
    )

    assert text._caching is True
    assert text._files == [File(id="file-id")]
    assert text._history == [Message(role="user", content="earlier")]
    assert text._max_tokens == 42
    assert len(text._middleware) == 1
    assert text._model == "text-model"
    assert text._schema == '{"type":"object"}'
    assert text._system == "you are a tutor"
    assert text._temperature == 0.7
    assert len(text._parts) == 2
    assert text._parts[0].image == MediaRef(mime_type="image/png", bytes=b"\xff")
    assert text._parts[1].text == "hello"


def test_image_chain() -> None:
    c = google("k")
    img = (
        c.image.aspect_ratio("16:9")
        .caching()
        .image("image/png", b"\xff")
        .image_size("2K")
        .include_text()
        .middleware(noop_middleware)
        .model("img-model")
        .text("compose")
    )
    assert img._aspect_ratio == "16:9"
    assert img._caching is True
    assert img._image_size == "2K"
    assert img._include_text is True
    assert len(img._middleware) == 1
    assert img._model == "img-model"
    assert len(img._parts) == 2


def test_agent_chain() -> None:
    c = google("k")
    tool = Tool(name="calc", description="d", schema={}, run=lambda _i: "42")
    ag = (
        c.agent.caching()
        .max_tokens(1)
        .middleware(noop_middleware)
        .model("a")
        .system("sys")
        .temperature(0.5)
        .tool(tool)
    )
    assert ag._caching is True
    assert ag._max_tokens == 1
    assert len(ag._middleware) == 1
    assert ag._model == "a"
    assert ag._system == "sys"
    assert ag._temperature == 0.5
    assert ag._tools == [tool]


def test_upload_chain() -> None:
    c = google("k")
    up = (
        c.upload.bytes(b"hi")
        .filename("f")
        .middleware(noop_middleware)
        .mime_type("text/plain")
        .path("/tmp/x")
    )
    assert up._bytes == b"hi"
    assert up._filename == "f"
    assert len(up._middleware) == 1
    assert up._mime_type == "text/plain"
    assert up._path == "/tmp/x"


# ---------- chain methods clone ----------


def test_chain_returns_new_instance() -> None:
    c = google("k")
    original = c.text
    configured = original.system("hello")
    assert original is not configured
    assert original._system == ""
    assert configured._system == "hello"


# ---------- per-provider factories ----------


def test_every_provider_factory() -> None:
    factories = [
        ("custom", lambda: new_client("custom", "k")),
        ("ai21", lambda: ai21("k")),
        ("anthropic", lambda: anthropic("k")),
        ("azure", lambda: azure("k")),
        ("bedrock", lambda: bedrock("k")),
        ("cerebras", lambda: cerebras("k")),
        ("cohere", lambda: cohere("k")),
        ("deepseek", lambda: deepseek("k")),
        ("doubao", lambda: doubao("k")),
        ("ernie", lambda: ernie("k")),
        ("fireworks", lambda: fireworks("k")),
        ("google", lambda: google("k")),
        ("grok", lambda: grok("k")),
        ("groq", lambda: groq("k")),
        ("lmstudio", lambda: lmstudio("k")),
        ("minimax", lambda: minimax("k")),
        ("mistral", lambda: mistral("k")),
        ("moonshot", lambda: moonshot("k")),
        ("ollama", lambda: ollama("k")),
        ("openai", lambda: openai("k")),
        ("openrouter", lambda: openrouter("k")),
        ("perplexity", lambda: perplexity("k")),
        ("qwen", lambda: qwen("k")),
        ("sambanova", lambda: sambanova("k")),
        ("together", lambda: together("k")),
        ("vllm", lambda: vllm("k")),
        ("yi", lambda: yi("k")),
        ("zhipu", lambda: zhipu("k")),
    ]
    assert len(factories) == 28
    for expected_name, factory in factories:
        c = factory()
        assert isinstance(c, Client)
        assert c.provider.api_key == "k"
        assert c.provider.name == expected_name


# ---------- phase 3 wiring tests with a mock HTTP server ----------


import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


class _MockServer:
    """Single-shot mock; serves the same canned response for every POST.

    For test_phase3_*, we point the typed-builder Client's
    ``provider.base_url`` at the mock and assert that the wired path
    rolled chain config into the request and returned the parsed
    response.
    """

    def __init__(self, response_body: dict[str, Any]):
        self.response_body = response_body
        self.last_path = ""
        self.last_body: dict[str, Any] | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):
                pass

            def do_POST(self):
                outer.last_path = self.path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if raw:
                    outer.last_body = json.loads(raw.decode("utf-8"))
                payload = json.dumps(outer.response_body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                outer.last_path = self.path
                if self.path.endswith("/results"):
                    line = json.dumps(
                        {
                            "custom_id": "req-0",
                            "result": {
                                "message": {
                                    "content": [{"type": "text", "text": "ok"}],
                                    "usage": {"input_tokens": 1, "output_tokens": 1},
                                }
                            },
                        }
                    ).encode("utf-8") + b"\n"
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-jsonl")
                    self.send_header("Content-Length", str(len(line)))
                    self.end_headers()
                    self.wfile.write(line)
                    return
                payload = json.dumps(outer.response_body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )

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


_ANTHROPIC_RESP = {
    "content": [{"type": "text", "text": "ok"}],
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def test_phase3_text_prompt_wires_against_legacy() -> None:
    with _MockServer(_ANTHROPIC_RESP) as server:
        c = anthropic("k")
        c.provider.base_url = server.url
        resp = asyncio.run(
            c.text.system("be terse").max_tokens(50).prompt("hello")
        )
        assert resp.text == "ok"
        body = server.last_body
        assert body is not None
        assert body["system"] == "be terse"
        assert body["max_tokens"] == 50


def test_phase3_text_batch_submits_and_returns_handle() -> None:
    submit_resp = {"id": "msgbatch_123"}

    with _MockServer(submit_resp) as server:
        c = anthropic("k")
        c.provider.base_url = server.url
        handle = asyncio.run(c.text.system("s").submit_batch("p1", "p2"))
        assert isinstance(handle, BatchHandle)
        assert handle.id == "msgbatch_123"
        body = server.last_body
        assert body is not None
        # Anthropic batch shape: {requests: [{custom_id, params: {...}}]}
        assert isinstance(body["requests"], list)
        assert body["requests"][0]["custom_id"] == "req-0"
        assert body["requests"][1]["custom_id"] == "req-1"
        # System propagates per-request through buildRequest.
        assert body["requests"][0]["params"]["system"] == "s"


def test_phase3_batch_handle_wait_polls_then_fetches_results() -> None:
    poll_resp = {"id": "msgbatch_123", "processing_status": "ended"}
    with _MockServer(poll_resp) as server:
        from llmkit import Provider as ProviderType

        handle = BatchHandle(
            id="msgbatch_123",
            provider=ProviderType(
                name="anthropic", api_key="k", base_url=server.url
            ),
        )
        responses = asyncio.run(handle.wait(poll_interval=0.01))
        assert len(responses) == 1
        assert responses[0].text == "ok"


def test_phase3_upload_run_validates_xor() -> None:
    c = openai("k")
    # Empty: error
    with pytest.raises(ValueError, match="exactly one of"):
        asyncio.run(c.upload.run())
    # Both: error
    with pytest.raises(ValueError, match="mutually exclusive"):
        asyncio.run(c.upload.bytes(b"x").path("/p").run())
    # Bytes-only: deferred
    with pytest.raises(ValueError, match="not yet wired"):
        asyncio.run(c.upload.bytes(b"x").run())


def test_phase3_text_stream_yields_chunks() -> None:
    """Stream wiring proves the asyncio.Queue + to_thread bridge works.

    Uses a Bytes-style raw-SSE response. The OpenAI streaming format
    is ``data: {...}\\n\\n`` per event; legacy prompt_stream parses
    these into chunks via a callback. Our bridge piles them in an
    asyncio.Queue and yields from the async generator.
    """

    class StreamHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args, **_kwargs):
            pass

        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                'data: {"choices":[{"delta":{"content":"He"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n',
                "data: [DONE]\n\n",
            ):
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()

    httpd = HTTPServer(("127.0.0.1", 0), StreamHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        c = openai("k")
        c.provider.base_url = f"http://127.0.0.1:{httpd.server_port}"

        async def consume() -> list[str]:
            got: list[str] = []
            async for chunk in c.text.stream("hi"):
                got.append(chunk)
            return got

        chunks = asyncio.run(consume())
        assert chunks == ["He", "llo"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=2)


def test_phase3_image_generate_wires() -> None:
    google_img_resp = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"inlineData": {"mimeType": "image/png", "data": "AAAA"}}
                    ]
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
    }
    with _MockServer(google_img_resp) as server:
        c = google("k")
        c.provider.base_url = server.url
        resp = asyncio.run(
            c.image.model("gemini-3.1-flash-image-preview").generate("a banana")
        )
        assert len(resp.images) == 1
        assert resp.images[0].mime_type == "image/png"


# ---------- Agent stateful builder ----------


def test_phase3_agent_prompt_initializes_state_and_reuses() -> None:
    with _MockServer(_ANTHROPIC_RESP) as server:
        c = anthropic("k")
        c.provider.base_url = server.url
        bot = c.agent.system("be brief")
        assert bot._state is None
        r1 = asyncio.run(bot.prompt("hi"))
        assert r1.text == "ok"
        first_state = bot._state
        assert first_state is not None
        r2 = asyncio.run(bot.prompt("again"))
        assert r2.text == "ok"
        # Same state instance — history retained.
        assert bot._state is first_state


def test_phase3_agent_reset_clears_state() -> None:
    with _MockServer(_ANTHROPIC_RESP) as server:
        c = anthropic("k")
        c.provider.base_url = server.url
        bot = c.agent.system("s")
        asyncio.run(bot.prompt("hi"))
        assert bot._state is not None
        bot.reset()
        assert bot._state is None


def test_phase3_agent_state_forking_load_bearing() -> None:
    """Load-bearing contract test for PYTHON_BUILDER_POST_MUTATION["Agent"].

    Without ``out._state = None`` after every chain method, a forked
    clone via ``bot.system("new")`` would silently share its parent's
    accumulated history through the same ``AgentState`` reference.
    """
    from llmkit import Provider as ProviderType
    from llmkit.agent import Agent as LegacyAgent
    from llmkit.builders.agent import AgentState

    c = anthropic("k")
    bot = c.agent.system("orig")
    # Manually populate state to simulate post-init.
    legacy = LegacyAgent(ProviderType(name="anthropic", api_key="k"))
    bot._state = AgentState(legacy)

    forked = bot.system("new")
    assert bot._state is not None  # parent preserved
    assert forked._state is None  # fork starts fresh


# ---------- re-exported types are usable ----------


def test_type_aliases_constructible() -> None:
    msg = Message(role="user", content="hi")
    tool = Tool(name="t", description="d", schema={}, run=lambda _i: "")
    resp = Response()
    img_resp = ImageResponse()
    img_data = ImageData(mime_type="image/png", data=b"")
    f = File(id="id")
    part = Part(text="hello")
    from llmkit import Provider as ProviderType
    bh = BatchHandle(id="abc", provider=ProviderType(name="openai", api_key="k"))
    mw: MiddlewareFn = noop_middleware
    assert msg.role == "user"
    assert tool.name == "t"
    assert resp.text == ""
    assert img_resp.images == []
    assert img_data.mime_type == "image/png"
    assert f.id == "id"
    assert part.text == "hello"
    assert bh.id == "abc"
    assert mw is noop_middleware
