"""Unit tests for llmkit.agent — the internal stateful Agent class
that the typed-builder *Agent terminal wraps. End-to-end agent tool
loops are tested via builder integration tests; here we cover the
state-management primitives directly (add_tool / reset / set_system)
so the public-symbol-untested gate has a path to STRICT."""

from __future__ import annotations

from llmkit.agent import Agent
from llmkit.types import Provider, Tool


def _calculator_tool() -> Tool:
    return Tool(
        name="add",
        description="Add two numbers",
        schema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        run=lambda args: str(args["a"] + args["b"]),
    )


def _agent() -> Agent:
    return Agent(provider=Provider(name="anthropic", api_key="test-key"))


# ---------- set_system ----------


def test_agent_set_system_assigns_field() -> None:
    a = _agent()
    assert a.system == ""
    a.set_system("You are a calculator.")
    assert a.system == "You are a calculator."


def test_agent_set_system_can_be_changed() -> None:
    # Mid-conversation system swaps aren't recommended but the primitive
    # supports it — last-writer-wins.
    a = _agent()
    a.set_system("first")
    a.set_system("second")
    assert a.system == "second"


# ---------- add_tool ----------


def test_agent_add_tool_appends_to_tools_list() -> None:
    a = _agent()
    assert a.tools == []
    t = _calculator_tool()
    a.add_tool(t)
    assert a.tools == [t]


def test_agent_add_tool_preserves_caller_order() -> None:
    a = _agent()
    t1 = _calculator_tool()
    t2 = Tool(
        name="sub",
        description="Subtract",
        schema={"type": "object", "properties": {}},
        run=lambda _i: "0",
    )
    a.add_tool(t1)
    a.add_tool(t2)
    # Caller order matters — some providers (Bedrock, Anthropic) reflect
    # the declaration order back in their tool-selection biases.
    assert a.tools == [t1, t2]


# ---------- reset ----------


def test_agent_reset_clears_history_and_tools() -> None:
    a = _agent()
    a.add_tool(_calculator_tool())
    # Simulate a previous chat by mutating history directly (chat() would
    # require a live HTTP server; we're testing reset, not chat).
    a.history.append(  # type: ignore[arg-type]
        type(
            "_M",
            (),
            {"role": "user", "content": "prior message", "tool_calls": [], "tool_result": None},
        )()
    )
    a.reset()
    assert a.history == []
    assert a.tools == []


def test_agent_reset_does_not_clear_system_or_provider() -> None:
    a = _agent()
    a.set_system("You are a helpful assistant.")
    a.add_tool(_calculator_tool())
    a.reset()
    # reset() is "clear conversation state"; the agent's identity
    # (provider, system prompt) survives so it can be reused.
    assert a.system == "You are a helpful assistant."
    assert a.provider.name == "anthropic"


def test_agent_reset_is_idempotent() -> None:
    a = _agent()
    a.reset()
    a.reset()
    assert a.history == []
    assert a.tools == []


# ---------- Options round-trip via constructor ----------


def test_agent_options_threaded_from_constructor() -> None:
    a = Agent(
        provider=Provider(name="anthropic", api_key="k"),
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        stop_sequences=["END"],
        seed=42,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        thinking_budget=1024,
        reasoning_effort="medium",
        caching=True,
        max_tool_iterations=5,
    )
    assert a.opts.max_tokens == 200
    assert a.opts.temperature == 0.7
    assert a.opts.top_p == 0.9
    assert a.opts.top_k == 40
    assert a.opts.stop_sequences == ["END"]
    assert a.opts.seed == 42
    assert a.opts.frequency_penalty == 0.1
    assert a.opts.presence_penalty == 0.2
    assert a.opts.thinking_budget == 1024
    assert a.opts.reasoning_effort == "medium"
    assert a.opts.caching is True
    assert a.opts.max_tool_iterations == 5


# ---------- ADR-026: request parity with the Text path ----------


class _CaptureServer:
    """Mock LLM endpoint capturing the request body the Agent sends."""

    def __init__(self, response: dict) -> None:
        self._response = response
        self.received_body: dict | None = None

    def __enter__(self):
        import json
        from http.server import BaseHTTPRequestHandler, HTTPServer

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                outer.received_body = json.loads(self.rfile.read(length))
                payload = json.dumps(outer._response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        import threading

        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def test_agent_request_applies_options_and_safety_like_text() -> None:
    """ADR-026 PIPE-001/004: the Agent builds its body through the shared
    _build_request, so generation options AND safety settings reach the wire
    body — matching the Text path. The old _build_agent_request applied options
    but dropped safety settings (the latent gap class). Google is used because
    it is the only shape that emits safetySettings."""
    from llmkit.types import SafetySetting

    google_response = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
    }
    with _CaptureServer(google_response) as server:
        agent = Agent(
            provider=Provider(name="google", api_key="k", base_url=server.url),
            temperature=0.1,
            safety_settings=[
                SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
            ],
        )
        resp = agent.chat("hi")

    assert resp.text == "ok"
    assert server.received_body is not None
    assert server.received_body["generationConfig"]["temperature"] == 0.1
    assert server.received_body["safetySettings"] == [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
    ]
