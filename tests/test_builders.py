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


# ---------- terminal stubs raise ----------


def _run(coro: object) -> object:
    return asyncio.get_event_loop().run_until_complete(coro)  # type: ignore[arg-type]


def test_text_prompt_raises() -> None:
    with pytest.raises(NotImplementedError, match="Text.prompt"):
        asyncio.run(google("k").text.prompt("hi"))


def test_text_stream_raises_on_iteration() -> None:
    async def consume() -> None:
        async for _ in google("k").text.stream("hi"):
            pass

    with pytest.raises(NotImplementedError, match="Text.stream"):
        asyncio.run(consume())


def test_text_batch_raises() -> None:
    with pytest.raises(NotImplementedError, match="Text.batch"):
        asyncio.run(google("k").text.batch("p1", "p2"))


def test_text_submit_batch_raises() -> None:
    with pytest.raises(NotImplementedError, match="Text.submit_batch"):
        asyncio.run(google("k").text.submit_batch("p1"))


def test_image_generate_raises() -> None:
    with pytest.raises(NotImplementedError, match="Image.generate"):
        asyncio.run(google("k").image.generate("a banana"))


def test_agent_prompt_raises() -> None:
    with pytest.raises(NotImplementedError, match="Agent.prompt"):
        asyncio.run(google("k").agent.prompt("hi"))


def test_agent_reset_raises() -> None:
    with pytest.raises(NotImplementedError, match="Agent.reset"):
        google("k").agent.reset()


def test_upload_run_raises() -> None:
    with pytest.raises(NotImplementedError, match="Upload.run"):
        asyncio.run(google("k").upload.run())


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
