"""





"""

from __future__ import annotations

import json
from typing import Any

from .structs import Message, ToolCall, ToolResult
from .wire_version import WIRE_SCHEMA_VERSION


class UnsupportedWireVersionError(Exception):
    """
"""

    def __init__(self, got: int, want: int) -> None:
        super().__init__(
            f"llmkit: unsupported wire schema version: got {got}, want <= {want}"
        )
        self.got = got
        self.want = want


class MissingWireVersionError(Exception):
    """
"""


class UnknownWireKeyError(Exception):
    """
"""

    def __init__(self, key: str) -> None:
        super().__init__(f"llmkit: unknown top-level wire key: {key!r}")
        self.key = key


def save_history(messages: list[Message]) -> bytes:
    """



"""
    payload: dict[str, Any] = {
        "_v": WIRE_SCHEMA_VERSION,
        "messages": [_message_to_wire(m) for m in messages],
    }
    return json.dumps(payload).encode("utf-8")


def load_history(data: bytes | str) -> list[Message]:
    """







"""
    if isinstance(data, bytes):
        text = data.decode("utf-8")
    else:
        text = data
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("llmkit: wire document is not a JSON object")
    if "_v" not in parsed:
        raise MissingWireVersionError("llmkit: wire document missing _v key")
    version = parsed["_v"]
    if not isinstance(version, int):
        raise ValueError(f"llmkit: wire _v is not an integer: {version!r}")
    if version > WIRE_SCHEMA_VERSION:
        raise UnsupportedWireVersionError(got=version, want=WIRE_SCHEMA_VERSION)
    for key in parsed:
        if key not in ("_v", "messages", "_meta"):
            raise UnknownWireKeyError(key)
    raw_msgs = parsed.get("messages", [])
    if not isinstance(raw_msgs, list):
        raise ValueError("llmkit: wire messages is not an array")
    return [_message_from_wire(m) for m in raw_msgs]


def _message_to_wire(m: Message) -> dict[str, Any]:
    """"""
    return {
        "role": m.role,
        "content": m.content,
        "tool_calls": [_tool_call_to_wire(tc) for tc in m.tool_calls],
        "tool_result": _tool_result_to_wire(m.tool_result),
    }


def _tool_call_to_wire(tc: ToolCall) -> dict[str, Any]:
    out: dict[str, Any] = {"id": tc.id, "name": tc.name}
    #
    #
    #
    if tc.input is not None:
        out["input"] = tc.input
    return out


def _tool_result_to_wire(tr: ToolResult | None) -> dict[str, Any] | None:
    if tr is None:
        return None
    return {"tool_use_id": tr.tool_use_id, "content": tr.content}


def _message_from_wire(raw: Any) -> Message:
    if not isinstance(raw, dict):
        raise ValueError(f"llmkit: wire message entry is not an object: {raw!r}")
    tool_calls_raw = raw.get("tool_calls") or []
    tool_calls: list[ToolCall] = []
    for tc in tool_calls_raw:
        if not isinstance(tc, dict):
            continue
        tool_calls.append(
            ToolCall(
                id=str(tc.get("id", "")),
                name=str(tc.get("name", "")),
                input=tc.get("input"),
            )
        )
    tool_result: ToolResult | None = None
    tr_raw = raw.get("tool_result")
    if isinstance(tr_raw, dict):
        tool_result = ToolResult(
            tool_use_id=str(tr_raw.get("tool_use_id", "")),
            content=str(tr_raw.get("content", "")),
        )
    return Message(
        role=str(raw.get("role", "")),
        content=str(raw.get("content", "") or ""),
        tool_calls=tool_calls,
        tool_result=tool_result,
    )
