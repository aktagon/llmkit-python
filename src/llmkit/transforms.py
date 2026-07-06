"""Message and tool transforms. Selected by ProviderSpec fields, not provider name."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, NoReturn

from .errors import ValidationError
from .paths import parse_data_uri
from .providers.generated.providers import ProviderSpec
from .providers.generated.request import (
    SystemPlacement,
    system_placement,
    tool_call_config,
)
from .structs import Message, ToolCall, ToolResult


MessageTransform = Callable[[dict[str, Any], list["_Msg"], "Request", ProviderSpec], None]
ToolDefTransform = Callable[[dict[str, Any], list["Tool"]], None]
ToolCallTransform = Callable[[list[ToolCall], dict[str, str]], dict[str, Any]]
ToolResultTransform = Callable[[ToolResult, dict[str, str]], dict[str, Any]]
ToolCallExtractor = Callable[[dict[str, Any], Any], list[ToolCall]]


def placement_for(cfg: ProviderSpec) -> SystemPlacement:
    from .providers.generated.providers import ProviderName

    return system_placement(ProviderName(cfg.name))


def map_role(role: str, mappings: dict[str, str]) -> str:
    return mappings.get(role, role)


def select_message_transform(cfg: ProviderSpec) -> MessageTransform:
    if cfg.chat_wire_shape == "ChatBedrock":
        return transform_bedrock_converse
    if cfg.chat_wire_shape == "ChatGoogle":
        return transform_google_parts
    if cfg.chat_wire_shape == "ChatResponsesOpenAI":
        return transform_responses_input
    return transform_flat_content


def select_tool_def_transform(cfg: ProviderSpec) -> ToolDefTransform:
    if cfg.chat_wire_shape == "ChatBedrock":
        return transform_bedrock_tool_defs
    if cfg.chat_wire_shape == "ChatGoogle":
        # Google carries tool params under a per-provider wire field (ADR-025):
        # "parametersJsonSchema" accepts native JSON Schema verbatim, vs the
        # OpenAPI-3.0-subset "parameters" default.
        tc = _tool_call_def(cfg)
        field = tc.params_wire_field if tc is not None and tc.params_wire_field else "parameters"

        def _google(body: dict[str, Any], tools: list["Tool"]) -> None:
            transform_google_function_declarations(body, tools, field)

        return _google
    tc = _tool_call_def(cfg)
    if tc is not None and tc.args_format == "map":
        return transform_anthropic_tools
    return transform_openai_functions


def select_tool_call_transform(cfg: ProviderSpec) -> ToolCallTransform:
    if cfg.chat_wire_shape == "ChatBedrock":
        return transform_bedrock_tool_call_msg
    if cfg.chat_wire_shape == "ChatGoogle":
        return transform_google_tool_call_msg
    tc = _tool_call_def(cfg)
    if tc is not None and tc.args_format == "map":
        return transform_anthropic_tool_call_msg
    return transform_openai_tool_call_msg


def select_tool_result_transform(cfg: ProviderSpec) -> ToolResultTransform:
    if cfg.chat_wire_shape == "ChatBedrock":
        return transform_bedrock_tool_result_msg
    if cfg.chat_wire_shape == "ChatGoogle":
        return transform_google_tool_result_msg
    tc = _tool_call_def(cfg)
    if tc is not None and tc.result_role == "user" and tc.args_format == "map":
        return transform_anthropic_tool_result_msg
    return transform_openai_tool_result_msg


def select_tool_call_extractor(cfg: ProviderSpec) -> ToolCallExtractor:
    if cfg.chat_wire_shape == "ChatBedrock":
        return extract_bedrock_tool_calls
    if cfg.chat_wire_shape == "ChatGoogle":
        return extract_google_tool_calls
    tc = _tool_call_def(cfg)
    if tc is not None and tc.args_format == "map":
        return extract_anthropic_tool_calls
    return extract_openai_tool_calls


def _tool_call_def(cfg: ProviderSpec):
    from .providers.generated.providers import ProviderName

    return tool_call_config(ProviderName(cfg.name))


# =============================================================================
# Internal message sum (ADR-026 PIPE-007/008)
# =============================================================================

@dataclass(frozen=True)
class _MsgText:
    """A text turn: exactly a role and its text content."""
    role: str
    text: str


@dataclass(frozen=True)
class _MsgCalls:
    """An assistant turn carrying one or more tool invocations."""
    calls: list[ToolCall]


@dataclass(frozen=True)
class _MsgResult:
    """A tool turn carrying exactly one execution result."""
    result: ToolResult


# A message is *exactly one of* the three variants. The public Message
# (structs.py) is a flat product that can encode an illegal multi-carrier
# combination; this union cannot, so the transforms below dispatch with
# match/case rather than the old if/elif silent-drop order.
_Msg = _MsgText | _MsgCalls | _MsgResult


def _assert_never(value: NoReturn) -> NoReturn:
    """Exhaustiveness guard for the _Msg sum (local stand-in for the 3.11+
    typing.assert_never; the package targets 3.10 and adds no dependency).

    Reached only if a _Msg variant is added without a matching case: the type
    checker errors here statically (the argument is no longer Never), and at
    runtime this raises instead of silently dropping the message.
    """
    raise TypeError(f"unhandled message variant {type(value).__name__}")


def to_internal(messages: list[Message]) -> list[_Msg]:
    """Convert the public, untrusted Message list into the internal sum.

    This is the single carrier-validation boundary (PIPE-008): a message
    carrying more than one of {content, tool calls, tool result} is rejected
    here, not silently mis-serialized downstream. The Text/batch/stream paths
    feed user-supplied Message lists through here; the Agent builds the sum
    directly from its trusted history and so skips this check.
    """
    out: list[_Msg] = []
    for i, m in enumerate(messages):
        carriers = sum(
            (m.tool_result is not None, bool(m.tool_calls), bool(m.content))
        )
        if carriers > 1:
            raise ValidationError(
                field=f"messages[{i}]",
                message="must carry only one of content, tool calls, or tool result",
            )
        if m.tool_result is not None:
            out.append(_MsgResult(result=m.tool_result))
        elif m.tool_calls:
            out.append(_MsgCalls(calls=list(m.tool_calls)))
        else:
            out.append(_MsgText(role=m.role, text=m.content))
    return out


# =============================================================================
# Message transforms — build the messages/contents array in request body
# =============================================================================

def transform_flat_content(body: dict[str, Any], msgs: list[_Msg], req: "Request", cfg: ProviderSpec) -> None:
    body["messages"] = _build_flat_message_array(msgs, req, cfg)


def transform_responses_input(body: dict[str, Any], msgs: list[_Msg], req: "Request", cfg: ProviderSpec) -> None:
    """Build the OpenAI Responses envelope (ADR-055): the SAME flat {role,
    content} array as Chat Completions, but under the "input" key instead of
    "messages" (POSTed to /v1/responses). The array shape is shared with
    transform_flat_content via _build_flat_message_array, so the golden
    witnesses that the only wire delta is the envelope key + endpoint.
    """
    body["input"] = _build_flat_message_array(msgs, req, cfg)


def _build_flat_message_array(msgs: list[_Msg], req: "Request", cfg: ProviderSpec) -> list[dict[str, Any]]:
    """Build the shared flat message array used by both the Chat Completions
    ("messages") and Responses ("input") envelopes."""
    out: list[dict[str, Any]] = []
    placement = placement_for(cfg)

    if placement == SystemPlacement.MESSAGE_IN_ARRAY and req.system:
        out.append(
            {
                "role": map_role("system", cfg.role_mappings),
                "content": req.system,
            }
        )

    has_media = bool(req.files) or bool(req.images)

    if msgs:
        # Tool-aware dispatch (ADR-020 / ADR-026): a tool-bearing history routes
        # through the same builder as plain text — a text turn is the no-tool case.
        call_t = select_tool_call_transform(cfg)
        result_t = select_tool_result_transform(cfg)
        for m in msgs:
            match m:
                case _MsgResult():
                    out.append(result_t(m.result, cfg.role_mappings))
                case _MsgCalls():
                    out.append(call_t(m.calls, cfg.role_mappings))
                case _MsgText():
                    out.append(
                        {
                            "role": map_role(m.role, cfg.role_mappings),
                            "content": m.text,
                        }
                    )
                case _:
                    _assert_never(m)
    elif req.user:
        if has_media:
            out.append(
                {
                    "role": map_role("user", cfg.role_mappings),
                    "content": _build_flat_content_parts(req, cfg),
                }
            )
        else:
            out.append(
                {
                    "role": map_role("user", cfg.role_mappings),
                    "content": req.user,
                }
            )

    return out


def _build_flat_content_parts(req: "Request", cfg: ProviderSpec) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    is_anthropic = cfg.chat_wire_shape == "ChatAnthropic"

    for f in req.files:
        if is_anthropic:
            parts.append(
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": f.id},
                }
            )
        else:
            parts.append(
                {
                    "type": "file",
                    "file": {"file_id": f.id},
                }
            )

    for img in req.images:
        if is_anthropic:
            if img.url.startswith("data:"):
                mime_type, data = parse_data_uri(img.url)
                parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": data,
                        },
                    }
                )
            else:
                parts.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": img.url},
                    }
                )
        else:
            detail = img.detail or "auto"
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": img.url, "detail": detail},
                }
            )

    parts.append({"type": "text", "text": req.user})
    return parts


def transform_google_parts(body: dict[str, Any], msgs: list[_Msg], req: "Request", cfg: ProviderSpec) -> None:
    contents: list[dict[str, Any]] = []
    if msgs:
        call_t = select_tool_call_transform(cfg)
        result_t = select_tool_result_transform(cfg)
        # Google's wire identifies a tool result by the function NAME, but the
        # universal ToolResult carries only tool_use_id. Recover id->name from
        # the call turns, which always precede their result in a valid history,
        # and resolve the result's name from it. A new ToolResult is built (not
        # mutated) so the caller's Message/history is untouched. The agent path
        # is unaffected (its extractor sets id==name); an unmatched id passes
        # through unchanged (transform_google_tool_result_msg uses tool_use_id).
        id_to_name: dict[str, str] = {}
        for m in msgs:
            match m:
                case _MsgResult():
                    r = m.result
                    name = id_to_name.get(r.tool_use_id)
                    if name:
                        r = ToolResult(tool_use_id=name, content=r.content)
                    contents.append(result_t(r, cfg.role_mappings))
                case _MsgCalls():
                    for c in m.calls:
                        id_to_name[c.id] = c.name
                    contents.append(call_t(m.calls, cfg.role_mappings))
                case _MsgText():
                    contents.append(
                        {
                            "role": map_role(m.role, cfg.role_mappings),
                            "parts": [{"text": m.text}],
                        }
                    )
                case _:
                    _assert_never(m)
    elif req.user:
        parts = _build_google_content_parts(req)
        contents.append(
            {
                "role": map_role("user", cfg.role_mappings),
                "parts": parts,
            }
        )
    body["contents"] = contents


def _build_google_content_parts(req: "Request") -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for f in req.files:
        parts.append(
            {
                "file_data": {
                    "file_uri": f.uri,
                    "mime_type": f.mime_type,
                }
            }
        )
    for img in req.images:
        if img.url.startswith("data:"):
            mime_type, data = parse_data_uri(img.url)
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": data,
                    }
                }
            )
        else:
            mime_type = img.mime_type or "image/jpeg"
            _, data = parse_data_uri(img.url)
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": data,
                    }
                }
            )
    parts.append({"text": req.user})
    return parts


def transform_bedrock_converse(body: dict[str, Any], msgs: list[_Msg], req: "Request", cfg: ProviderSpec) -> None:
    if req.system:
        body["system"] = [{"text": req.system}]
    out: list[dict[str, Any]] = []
    if msgs:
        call_t = select_tool_call_transform(cfg)
        result_t = select_tool_result_transform(cfg)
        for m in msgs:
            match m:
                case _MsgResult():
                    out.append(result_t(m.result, cfg.role_mappings))
                case _MsgCalls():
                    out.append(call_t(m.calls, cfg.role_mappings))
                case _MsgText():
                    out.append(
                        {
                            "role": map_role(m.role, cfg.role_mappings),
                            "content": [{"text": m.text}],
                        }
                    )
                case _:
                    _assert_never(m)
    elif req.user:
        if req.images:
            content = _build_bedrock_content_parts(req)
        else:
            content = [{"text": req.user}]
        out.append(
            {
                "role": map_role("user", cfg.role_mappings),
                "content": content,
            }
        )
    body["messages"] = out


def _build_bedrock_content_parts(req: "Request") -> list[dict[str, Any]]:
    """Converse content array with image blocks (ADR-060). Each image emits
    {image:{format,source:{bytes}}}; the prompt text follows as a trailing
    {text} block, preserving caller order among images. Mirrors go/transforms.go
    buildBedrockContentParts."""
    parts: list[dict[str, Any]] = []
    for img in req.images:
        mime_type, data = parse_data_uri(img.url)
        if not mime_type:
            mime_type = img.mime_type
        parts.append(
            {
                "image": {
                    "format": _bedrock_image_format(mime_type),
                    "source": {"bytes": data},
                }
            }
        )
    parts.append({"text": req.user})
    return parts


def _bedrock_image_format(mime_type: str) -> str:
    """Derive the Converse `format` token from a MIME type (image/png -> "png")."""
    i = mime_type.rfind("/")
    if i >= 0:
        return mime_type[i + 1:]
    return mime_type


# =============================================================================
# Tool definition transforms
# =============================================================================

def transform_openai_functions(body: dict[str, Any], tools: list["Tool"]) -> None:
    body["tools"] = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema,
            },
        }
        for t in tools
    ]


def transform_anthropic_tools(body: dict[str, Any], tools: list["Tool"]) -> None:
    body["tools"] = [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.schema,
        }
        for t in tools
    ]


def transform_google_function_declarations(
    body: dict[str, Any], tools: list["Tool"], params_wire_field: str = "parameters"
) -> None:
    decls = [
        {
            "name": t.name,
            "description": t.description,
            params_wire_field: t.schema,
        }
        for t in tools
    ]
    body["tools"] = [{"functionDeclarations": decls}]


def transform_bedrock_tool_defs(body: dict[str, Any], tools: list["Tool"]) -> None:
    defs = [
        {
            "toolSpec": {
                "name": t.name,
                "description": t.description,
                "inputSchema": {"json": t.schema},
            }
        }
        for t in tools
    ]
    body["toolConfig"] = {"tools": defs}


# =============================================================================
# Tool call message transforms
# =============================================================================

def transform_openai_tool_call_msg(calls: list[ToolCall], role_mappings: dict[str, str]) -> dict[str, Any]:
    return {
        "role": map_role("assistant", role_mappings),
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.input if tc.input is not None else {}),
                },
            }
            for tc in calls
        ],
    }


def transform_anthropic_tool_call_msg(calls: list[ToolCall], role_mappings: dict[str, str]) -> dict[str, Any]:
    return {
        "role": map_role("assistant", role_mappings),
        "content": [
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.input if tc.input is not None else {},
            }
            for tc in calls
        ],
    }


def transform_google_tool_call_msg(calls: list[ToolCall], role_mappings: dict[str, str]) -> dict[str, Any]:
    return {
        "role": map_role("assistant", role_mappings),
        "parts": [
            {
                "functionCall": {
                    "name": tc.name,
                    "args": tc.input if tc.input is not None else {},
                }
            }
            for tc in calls
        ],
    }


def transform_bedrock_tool_call_msg(calls: list[ToolCall], role_mappings: dict[str, str]) -> dict[str, Any]:
    return {
        "role": map_role("assistant", role_mappings),
        "content": [
            {
                "toolUse": {
                    "toolUseId": tc.id,
                    "name": tc.name,
                    "input": tc.input if tc.input is not None else {},
                }
            }
            for tc in calls
        ],
    }


# =============================================================================
# Tool result message transforms
# =============================================================================

def transform_openai_tool_result_msg(result: ToolResult, _: dict[str, str]) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": result.content,
        "tool_call_id": result.tool_use_id,
    }


def transform_anthropic_tool_result_msg(result: ToolResult, _: dict[str, str]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": result.tool_use_id,
                "content": result.content,
            }
        ],
    }


def transform_google_tool_result_msg(result: ToolResult, _: dict[str, str]) -> dict[str, Any]:
    return {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": result.tool_use_id,
                    "response": {"result": result.content},
                }
            }
        ],
    }


def transform_bedrock_tool_result_msg(result: ToolResult, _: dict[str, str]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": result.tool_use_id,
                    "content": [{"text": result.content}],
                }
            }
        ],
    }


# =============================================================================
# Tool call extraction
# =============================================================================

def extract_openai_tool_calls(raw: dict[str, Any], tc_cfg: Any) -> list[ToolCall]:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    choice = choices[0]
    if not isinstance(choice, dict):
        return []
    message = choice.get("message")
    if not isinstance(message, dict):
        return []
    tcs = message.get("tool_calls")
    if not isinstance(tcs, list):
        return []
    calls: list[ToolCall] = []
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        if tc_cfg is not None and tc_cfg.args_format == "json_string":
            args_str = fn.get("arguments") or ""
            try:
                inp = json.loads(args_str) if args_str else {}
            except ValueError:
                inp = {}
        else:
            raw_args = fn.get("arguments")
            inp = raw_args if isinstance(raw_args, dict) else {}
        calls.append(
            ToolCall(
                id=str(tc.get("id", "")),
                name=str(fn.get("name", "")),
                input=inp,
            )
        )
    return calls


def extract_anthropic_tool_calls(raw: dict[str, Any], _: Any) -> list[ToolCall]:
    content = raw.get("content")
    if not isinstance(content, list):
        return []
    calls: list[ToolCall] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        inp = block.get("input")
        if not isinstance(inp, dict):
            inp = {}
        calls.append(
            ToolCall(
                id=str(block.get("id", "")),
                name=str(block.get("name", "")),
                input=inp,
            )
        )
    return calls


def extract_google_tool_calls(raw: dict[str, Any], _: Any) -> list[ToolCall]:
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return []
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        return []
    content = candidate.get("content")
    if not isinstance(content, dict):
        return []
    parts = content.get("parts")
    if not isinstance(parts, list):
        return []
    calls: list[ToolCall] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        fc = part.get("functionCall")
        if not isinstance(fc, dict):
            continue
        args = fc.get("args")
        if not isinstance(args, dict):
            args = {}
        name = str(fc.get("name", ""))
        calls.append(ToolCall(id=name, name=name, input=args))
    return calls


def extract_bedrock_tool_calls(raw: dict[str, Any], _: Any) -> list[ToolCall]:
    output = raw.get("output")
    if not isinstance(output, dict):
        return []
    message = output.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    calls: list[ToolCall] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        tu = block.get("toolUse")
        if not isinstance(tu, dict):
            continue
        inp = tu.get("input")
        if not isinstance(inp, dict):
            inp = {}
        calls.append(
            ToolCall(
                id=str(tu.get("toolUseId", "")),
                name=str(tu.get("name", "")),
                input=inp,
            )
        )
    return calls


# Forward-declare `Request` and `Tool` via TYPE_CHECKING import at top
from .types import Request, Tool  # noqa: E402  (imported late to avoid circular types)
