"""ADR-023 STAB-007: per-SDK wire round-trip test against the canonical
golden at codegen/testdata/wire/v1/messages.json.

The test (1) builds the canonical fixture as in-memory Message
values, (2) calls save_history and asserts JSON-value equality with
the committed golden, (3) round-trips through load_history and
asserts value equality with the input fixture, (4) asserts
MissingWireVersionError on a `_v`-less doc, (5) asserts
UnsupportedWireVersionError on `_v: 99`, (6) round-trips via the
chain methods bot.save() / bot.load(data), and (7) drops
target/wire/python.json for the cross-SDK comparator (STAB-010).
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
    """Mirror of ontology/fixtures/wire.ttl — the canonical
    conversation covering every Message role + every tool-turn
    permutation."""
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
    """STAB-007: save_history output is JSON-value-equal to the golden."""
    fixture = _canonical_fixture()
    actual = json.loads(save_history(fixture).decode("utf-8"))
    expected = json.loads(GOLDEN_PATH.read_text())
    assert actual == expected


def test_wire_round_trip_value_equal() -> None:
    """STAB-007: load_history(save_history(msgs)) == msgs."""
    fixture = _canonical_fixture()
    restored = load_history(save_history(fixture))
    assert restored == fixture


def test_wire_missing_v_rejected() -> None:
    """STAB-011: bare-array dumps are rejected."""
    with pytest.raises(MissingWireVersionError):
        load_history(b'{"messages": []}')


def test_wire_unsupported_version_rejected() -> None:
    """STAB-003: `_v` above the SDK's compiled-in version is rejected."""
    with pytest.raises(UnsupportedWireVersionError):
        load_history(b'{"_v": 99, "messages": []}')


def test_wire_unknown_top_level_key_rejected() -> None:
    """STAB-002: unknown top-level keys (other than _v / messages /
    _meta) are rejected with a typed error."""
    with pytest.raises(UnknownWireKeyError):
        load_history(b'{"_v": 1, "messages": [], "stray": 42}')


def test_wire_meta_passthrough_accepted() -> None:
    """STAB-002: _meta is a consumer-owned namespace; load_history
    ignores it on read."""
    msgs = load_history(b'{"_v": 1, "messages": [], "_meta": {"trace": "abc"}}')
    assert msgs == []


def test_chain_methods_round_trip() -> None:
    """STAB-012: bot.save() / bot.load(data) produce bytes that load
    back into a value-equal builder."""
    from llmkit.agent import Agent as LegacyAgent, _InternalMessage
    from llmkit import Provider as ProviderType
    from llmkit.builders.agent import AgentState

    c = anthropic("k")
    bot = c.agent
    legacy = LegacyAgent(ProviderType(name="anthropic", api_key="k"))
    # Match the canonical fixture so the round-trip exercise covers
    # every turn kind.
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
    """STAB-010: each SDK's existing test phase drops a
    target/wire/<sdk>.json file consumed by the cross-SDK
    comparator. The artifact is the save_history output over the
    canonical fixture."""
    artifact_dir = REPO_ROOT / "target" / "wire"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "python.json").write_bytes(save_history(_canonical_fixture()))
