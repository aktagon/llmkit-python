"""



"""

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


#


def test_agent_set_system_assigns_field() -> None:
    a = _agent()
    assert a.system == ""
    a.set_system("You are a calculator.")
    assert a.system == "You are a calculator."


def test_agent_set_system_can_be_changed() -> None:
    #
    #
    a = _agent()
    a.set_system("first")
    a.set_system("second")
    assert a.system == "second"


#


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
    #
    #
    assert a.tools == [t1, t2]


#


def test_agent_reset_clears_history_and_tools() -> None:
    a = _agent()
    a.add_tool(_calculator_tool())
    #
    #
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
    #
    #
    assert a.system == "You are a helpful assistant."
    assert a.provider.name == "anthropic"


def test_agent_reset_is_idempotent() -> None:
    a = _agent()
    a.reset()
    a.reset()
    assert a.history == []
    assert a.tools == []


#


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
