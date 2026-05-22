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
    SafetySetting,
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
        .add_middleware(noop_middleware)
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

    ss = SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
    text2 = c.text.safety_settings([ss])
    assert text2._safety_settings == [ss]


def test_image_chain() -> None:
    c = google("k")
    img = (
        c.image.aspect_ratio("16:9")
        .image("image/png", b"\xff")
        .image_size("2K")
        .include_text()
        .add_middleware(noop_middleware)
        .model("img-model")
        .text("compose")
    )
    assert img._aspect_ratio == "16:9"
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
        .max_tool_iterations(3)
        .add_middleware(noop_middleware)
        .model("a")
        .system("sys")
        .temperature(0.5)
        .add_tool(tool)
    )
    assert ag._caching is True
    assert ag._max_tokens == 1
    assert ag._max_tool_iterations == 3
    assert len(ag._middleware) == 1
    assert ag._model == "a"
    assert ag._system == "sys"
    assert ag._temperature == 0.5
    assert ag._tools == [tool]

    ag_ss = SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE")
    ag2 = c.agent.safety_settings([ag_ss])
    assert ag2._safety_settings == [ag_ss]


def test_upload_chain() -> None:
    c = google("k")
    up = (
        c.upload.bytes(b"hi")
        .filename("f")
        .add_middleware(noop_middleware)
        .mime_type("text/plain")
        .path("/tmp/x")
    )
    assert up._bytes == b"hi"
    assert up._filename == "f"
    assert len(up._middleware) == 1
    assert up._mime_type == "text/plain"
    assert up._path == "/tmp/x"


# ---------- appender semantics (ADR-021) ----------


def test_agent_add_tool_appends() -> None:
    c = google("k")
    t1 = Tool(name="first", description="d", schema={}, run=lambda _i: "")
    t2 = Tool(name="second", description="d", schema={}, run=lambda _i: "")
    ag = c.agent.system("S").add_tool(t1).add_tool(t2)
    assert ag._tools == [t1, t2]


def test_text_add_middleware_appends() -> None:
    c = google("k")
    bot = c.text.add_middleware(noop_middleware).add_middleware(noop_middleware)
    assert len(bot._middleware) == 2


# ---------- chain methods clone ----------


def test_chain_returns_new_instance() -> None:
    c = google("k")
    original = c.text
    configured = original.system("hello")
    assert original is not configured
    assert original._system == ""
    assert configured._system == "hello"


def test_client_with_base_url_sets_and_returns_self() -> None:
    override = "https://example.test/v1"
    c = new_client("vertex", "test-token").with_base_url(override)
    assert c.provider.base_url == override
    # Chainability: with_base_url must return the same Client so callers
    # can write `c = vertex(token).with_base_url(url)` in one line.
    assert isinstance(c, Client)


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


def test_text_prompt_surfaces_finish_reason() -> None:
    response = {
        "content": [{"type": "text", "text": "truncated"}],
        "usage": {"input_tokens": 4, "output_tokens": 10},
        "stop_reason": "max_tokens",
    }
    with _MockServer(response) as server:
        c = anthropic("k")
        c.provider.base_url = server.url
        resp = asyncio.run(c.text.max_tokens(10).prompt("ping"))
        assert resp.finish_reason == "max_tokens"
        assert resp.finish_message == ""


def test_text_prompt_omits_finish_reason_when_absent() -> None:
    with _MockServer(_ANTHROPIC_RESP) as server:
        c = anthropic("k")
        c.provider.base_url = server.url
        resp = asyncio.run(c.text.prompt("ping"))
        assert resp.finish_reason == ""
        assert resp.finish_message == ""


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
    # Bytes without filename: error
    with pytest.raises(ValueError, match="filename"):
        asyncio.run(c.upload.bytes(b"x").run())


def test_phase3_upload_run_bytes_branch_round_trips() -> None:
    """Bytes branch posts the multipart body to the configured baseUrl.

    Captures the raw multipart body and asserts the filename + payload
    + mime override made it through.
    """
    captured: dict[str, bytes] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a, **_k):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or "0")
            captured["body"] = self.rfile.read(length)
            payload = b'{"id":"file-zzz"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        c = openai("k")
        c.provider.base_url = f"http://127.0.0.1:{server.server_port}"
        result = asyncio.run(
            c.upload.bytes(b"hello").filename("note.txt").mime_type("text/plain").run()
        )
        assert result.id == "file-zzz"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    body = captured["body"]
    assert b"hello" in body
    assert b'filename="note.txt"' in body
    assert b"Content-Type: text/plain" in body


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

        async def consume() -> tuple[list[str], "TextStream"]:
            got: list[str] = []
            stream = c.text.stream("hi")
            async for chunk in stream:
                got.append(chunk)
            return got, stream

        chunks, stream = asyncio.run(consume())
        assert chunks == ["He", "llo"]
        # Trailing handle: after iteration, response is populated.
        assert stream.response is not None
        assert stream.response.text == "Hello"
        assert stream.error is None
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


def test_phase_a_agent_history_writer_replaces_chain_state() -> None:
    """ADR-020 HIST-003: bot.history(*msgs) replaces the chain history."""
    c = anthropic("k")
    msg_a = Message(role="user", content="first")
    msg_b = Message(role="assistant", content="ok")
    bot = c.agent.system("you are helpful").history(msg_a, msg_b)
    assert bot._history == [msg_a, msg_b]
    # second call replaces, doesn't append.
    msg_c = Message(role="user", content="reset")
    rebot = bot.history(msg_c)
    assert rebot._history == [msg_c]


def test_phase_a_agent_messages_reader_empty_before_prompt() -> None:
    """ADR-020 HIST-004: bot.messages is an empty tuple before .prompt()."""
    c = anthropic("k")
    bot = c.agent.system("seeded").history(Message(role="user", content="hi"))
    # Builder has chain history but no runtime state yet.
    assert bot.messages == ()


def test_phase_a_agent_messages_reader_after_init() -> None:
    """ADR-020 HIST-004: bot.messages projects internal history through the
    runtime-state adapter once the legacy agent is constructed.

    Tests the tool-turn projection: assistant-with-tools turn has
    non-empty tool_calls; tool-result turn has non-None tool_result
    and the public 'tool' role discriminator (mapped from internal
    'tool_result').
    """
    from llmkit import Provider as ProviderType
    from llmkit import ToolCall as PubToolCall
    from llmkit import ToolResult as PubToolResult
    from llmkit.agent import Agent as LegacyAgent, _InternalMessage
    from llmkit.builders.agent import AgentState

    c = anthropic("k")
    bot = c.agent
    legacy = LegacyAgent(ProviderType(name="anthropic", api_key="k"))
    legacy.history = [
        _InternalMessage(role="user", content="list py files"),
        _InternalMessage(
            role="assistant",
            tool_calls=[
                PubToolCall(id="call_1", name="list_files", input={"path": "src"})
            ],
        ),
        _InternalMessage(
            role="tool_result",
            tool_result=PubToolResult(tool_use_id="call_1", content="a.py b.py"),
        ),
    ]
    bot._state = AgentState(legacy)

    msgs = bot.messages
    assert len(msgs) == 3
    assert msgs[0].role == "user"
    assert msgs[0].content == "list py files"
    assert msgs[1].role == "assistant"
    assert len(msgs[1].tool_calls) == 1
    assert msgs[1].tool_calls[0].name == "list_files"
    # Internal 'tool_result' role flattens to public 'tool'.
    assert msgs[2].role == "tool"
    assert msgs[2].tool_result is not None
    assert msgs[2].tool_result.tool_use_id == "call_1"
    assert msgs[2].tool_result.content == "a.py b.py"


def test_phase_a_agent_history_init_seeds_runtime() -> None:
    """ADR-020 HIST-007: chain history populates the legacy agent on init."""
    from llmkit.builders.agent import _init_agent

    c = anthropic("k")
    bot = (
        c.agent.system("seed")
        .history(
            Message(role="user", content="hi"),
            Message(role="assistant", content="hi back"),
        )
    )
    state = _init_agent(bot)
    assert len(state.agent.history) == 2
    assert state.agent.history[0].role == "user"
    assert state.agent.history[0].content == "hi"
    assert state.agent.history[1].role == "assistant"
    assert state.agent.history[1].content == "hi back"


# ---------- re-exported types are usable ----------


def test_type_aliases_constructible() -> None:
    msg = Message(role="user", content="hi")
    tool = Tool(name="t", description="d", schema={}, run=lambda _i: "")
    resp = Response()
    img_resp = ImageResponse()
    img_data = ImageData(mime_type="image/png", bytes=b"")
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


# ---------- A2 bounded stream queue ----------


def test_stream_queue_applies_backpressure(monkeypatch) -> None:
    """A2: worker thread is paced by consumer when queue fills.

    Stub legacy_prompt_stream with a tight loop that pushes 200
    chunks via on_chunk. Consumer drains with asyncio.sleep(1ms)
    per chunk. Verify two invariants:

    1. All 200 chunks arrive in order (no loss, no scrambling).
    2. The producer is paced by the consumer: with maxsize=64
       and 200 chunks at 1ms drain each, the worker must spend
       at least ~135ms (200-64 chunks past the buffer × 1ms)
       inside on_chunk. An unbounded queue completes in <5ms
       because put_nowait never blocks.
    """
    import time

    from llmkit.builders import stream as stream_mod
    from llmkit.types import Response as _Resp

    total = 200
    consumer_delay = 0.001  # 1ms per chunk
    producer_wallclock = {"value": 0.0}

    def fake_legacy(provider, request, on_chunk, **kwargs):  # noqa: ARG001
        start = time.monotonic()
        for i in range(total):
            on_chunk(f"{i} ")
        producer_wallclock["value"] = time.monotonic() - start
        return _Resp()

    monkeypatch.setattr(stream_mod, "legacy_prompt_stream", fake_legacy)

    async def run() -> list[str]:
        c = anthropic("k")
        got: list[str] = []
        async for chunk in stream_mod.text_stream(c.text, "hi"):
            got.append(chunk)
            await asyncio.sleep(consumer_delay)
        return got

    got = asyncio.run(run())

    assert got == [f"{i} " for i in range(total)], "chunks lost or out of order"
    # Backpressure floor: 136 chunks must wait for consumer drain
    # at >=1ms each. Allow generous slack (50ms) for scheduler noise.
    min_expected = (total - 64) * consumer_delay * 0.5
    assert producer_wallclock["value"] >= min_expected, (
        f"producer ran in {producer_wallclock['value']*1000:.1f}ms, "
        f"expected >= {min_expected*1000:.1f}ms — backpressure may be broken"
    )


# ---------- Coverage gate Phase 2: chain methods previously untested ----------
# These are the sampling hyperparameters + Text.batch + Agent.add_tool +
# Agent.reset that the strict-mode public-symbol-untested gate flagged.


def test_text_sampling_hyperparameters_round_trip() -> None:
    c = anthropic("k")
    text = (
        c.text.frequency_penalty(0.2)
        .presence_penalty(0.4)
        .reasoning_effort("medium")
        .seed(42)
        .stop_sequences("\n\n", "END")
        .thinking_budget(2048)
        .top_k(40)
        .top_p(0.9)
    )
    assert text._frequency_penalty == 0.2
    assert text._presence_penalty == 0.4
    assert text._reasoning_effort == "medium"
    assert text._seed == 42
    assert text._stop_sequences == ["\n\n", "END"]
    assert text._thinking_budget == 2048
    assert text._top_k == 40
    assert text._top_p == 0.9


def test_agent_sampling_hyperparameters_round_trip() -> None:
    c = anthropic("k")
    ag = (
        c.agent.frequency_penalty(0.2)
        .presence_penalty(0.4)
        .reasoning_effort("high")
        .seed(7)
        .stop_sequences("STOP")
        .thinking_budget(4096)
        .top_k(50)
        .top_p(0.95)
    )
    assert ag._frequency_penalty == 0.2
    assert ag._presence_penalty == 0.4
    assert ag._reasoning_effort == "high"
    assert ag._seed == 7
    assert ag._stop_sequences == ["STOP"]
    assert ag._thinking_budget == 4096
    assert ag._top_k == 50
    assert ag._top_p == 0.95


def test_text_batch_method_exists_and_is_async() -> None:
    """Smoke-only: .batch is the multi-prompt convenience that runs
    submit_batch + wait under the hood. End-to-end exercise lives in
    plan-018 integration tests (real provider batch lifecycle); here we
    just confirm the method is wired and async."""
    import inspect

    c = anthropic("k")
    method = c.text.batch
    assert callable(method)
    assert inspect.iscoroutinefunction(method)


# === ADR-014 — Raw response escape hatch ===

_ANTHROPIC_RAW_RESP = {
    "id": "msg_01",
    "content": [{"type": "text", "text": "ok"}],
    "usage": {"input_tokens": 1, "output_tokens": 1},
    # Provider-specific field that the universal Response does not carry.
    "stop_reason": "end_turn",
}


def test_text_raw_populates_response_raw() -> None:
    with _MockServer(_ANTHROPIC_RAW_RESP) as server:
        c = anthropic("k")
        c.provider.base_url = server.url
        resp = asyncio.run(c.text.raw().prompt("hello"))
        assert resp.raw is not None
        assert resp.raw["stop_reason"] == "end_turn"


def test_text_raw_absent_leaves_response_raw_none() -> None:
    with _MockServer(_ANTHROPIC_RAW_RESP) as server:
        c = anthropic("k")
        c.provider.base_url = server.url
        resp = asyncio.run(c.text.prompt("hello"))
        assert resp.raw is None


def test_image_and_agent_raw_chain_callable() -> None:
    # Chain-method coverage. Image.generate and Agent.prompt route
    # through the same b._raw -> options.raw -> result.raw plumbing
    # exercised end-to-end in the Text.raw test; here we just confirm
    # the chain hooks are wired and flip the private field.
    c = google("k")
    img = c.image.model("gemini-2.5-flash-image-preview").raw()
    assert img._raw is True
    ag = c.agent.system("x").raw()
    assert ag._raw is True
