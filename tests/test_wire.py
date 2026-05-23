"""










"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmkit import (
    Message,
    MissingWireVersionError,
    ToolCall,
    ToolResult,
    UnknownWireKeyError,
    UnsupportedWireVersionError,
    anthropic,
    load_history,
    save_history,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "codegen" / "testdata" / "wire" / "v1" / "messages.json"


def _canonical_fixture() -> list[Message]:
    """

"""
    return [
        Message(
            role="user",
            content="list .py files in src",
            tool_calls=[],
            tool_result=None,
        ),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="call_abc", name="list_files", input={"path": "src"})
            ],
            tool_result=None,
        ),
        Message(
            role="tool",
            content="",
            tool_calls=[],
            tool_result=ToolResult(tool_use_id="call_abc", content="a.py b.py"),
        ),
        Message(
            role="assistant",
            content="Found 2 Python files: a.py, b.py",
            tool_calls=[],
            tool_result=None,
        ),
    ]


def test_wire_golden_matches() -> None:
    """"""
    fixture = _canonical_fixture()
    actual = json.loads(save_history(fixture).decode("utf-8"))
    expected = json.loads(GOLDEN_PATH.read_text())
    assert actual == expected


def test_wire_round_trip_value_equal() -> None:
    """"""
    fixture = _canonical_fixture()
    restored = load_history(save_history(fixture))
    assert restored == fixture


def test_wire_missing_v_rejected() -> None:
    """"""
    with pytest.raises(MissingWireVersionError):
        load_history(b'{"messages": []}')


def test_wire_unsupported_version_rejected() -> None:
    """"""
    with pytest.raises(UnsupportedWireVersionError):
        load_history(b'{"_v": 99, "messages": []}')


def test_wire_unknown_top_level_key_rejected() -> None:
    """
"""
    with pytest.raises(UnknownWireKeyError):
        load_history(b'{"_v": 1, "messages": [], "stray": 42}')


def test_wire_meta_passthrough_accepted() -> None:
    """
"""
    msgs = load_history(b'{"_v": 1, "messages": [], "_meta": {"trace": "abc"}}')
    assert msgs == []


def test_chain_methods_round_trip() -> None:
    """
"""
    from llmkit.agent import Agent as LegacyAgent, _InternalMessage
    from llmkit import Provider as ProviderType
    from llmkit.builders.agent import AgentState

    c = anthropic("k")
    bot = c.agent
    legacy = LegacyAgent(ProviderType(name="anthropic", api_key="k"))
    #
    #
    legacy.history = [
        _InternalMessage(role="user", content="list .py files in src"),
        _InternalMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="call_abc", name="list_files", input={"path": "src"})
            ],
        ),
        _InternalMessage(
            role="tool_result",
            tool_result=ToolResult(tool_use_id="call_abc", content="a.py b.py"),
        ),
        _InternalMessage(
            role="assistant", content="Found 2 Python files: a.py, b.py"
        ),
    ]
    bot._state = AgentState(legacy)

    data = bot.save()
    fresh = c.agent.load(data)
    assert list(fresh.messages) == []  # runtime state not initialized yet
    assert list(fresh._history) == _canonical_fixture()


def test_drop_target_artifact_for_cross_sdk_comparator() -> None:
    """


"""
    artifact_dir = REPO_ROOT / "target" / "wire"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "python.json").write_bytes(save_history(_canonical_fixture()))
