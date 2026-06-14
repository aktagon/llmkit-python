""""""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, NoReturn

from .errors import ValidationError
from .paths import parse_data_uri
from .providers.generated.providers import ProviderSpec
from .providers.generated.request import (
    AuthScheme,
    SystemPlacement,
    auth_scheme,
    system_placement,
    tool_call_config,
)
from .structs import Message, ToolCall, ToolResult


MessageTransform = Callable[[dict[str, Any], list["_Msg"], "Request", ProviderSpec], None]
ToolDefTransform = Callable[[dict[str, Any], list["Tool"]], None]
ToolCallTransform = Callable[[list[ToolCall], dict[str, str]], dict[str, Any]]
ToolResultTransform = Callable[[ToolResult, dict[str, str]], dict[str, Any]]
ToolCallExtractor = Callable[[dict[str, Any], Any], list[ToolCall]]


def is_bedrock(cfg: ProviderSpec) -> bool:
    return (
        cfg.wraps_options_in == "inferenceConfig"
        and auth_scheme_for(cfg) == AuthScheme.SIG_V4
    )


def auth_scheme_for(cfg: ProviderSpec) -> AuthScheme:
    """"""
    from .providers.generated.providers import ProviderName

    return auth_scheme(ProviderName(cfg.name))


def placement_for(cfg: ProviderSpec) -> SystemPlacement:
    from .providers.generated.providers import ProviderName

    return system_placement(ProviderName(cfg.name))


def map_role(role: str, mappings: dict[str, str]) -> str:
    return mappings.get(role, role)


def select_message_transform(cfg: ProviderSpec) -> MessageTransform:
    if is_bedrock(cfg):
        return transform_bedrock_converse
    if placement_for(cfg) == SystemPlacement.SIBLING_OBJECT:
        return transform_google_parts
    return transform_flat_content


def select_tool_def_transform(cfg: ProviderSpec) -> ToolDefTransform:
    if is_bedrock(cfg):
        return transform_bedrock_tool_defs
    if placement_for(cfg) == SystemPlacement.SIBLING_OBJECT:
        #
        #
        #
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
    if is_bedrock(cfg):
        return transform_bedrock_tool_call_msg
    if placement_for(cfg) == SystemPlacement.SIBLING_OBJECT:
        return transform_google_tool_call_msg
    tc = _tool_call_def(cfg)
    if tc is not None and tc.args_format == "map":
        return transform_anthropic_tool_call_msg
    return transform_openai_tool_call_msg


def select_tool_result_transform(cfg: ProviderSpec) -> ToolResultTransform:
    if is_bedrock(cfg):
        return transform_bedrock_tool_result_msg
    if placement_for(cfg) == SystemPlacement.SIBLING_OBJECT:
        return transform_google_tool_result_msg
    tc = _tool_call_def(cfg)
    if tc is not None and tc.result_role == "user" and tc.args_format == "map":
        return transform_anthropic_tool_result_msg
    return transform_openai_tool_result_msg


def select_tool_call_extractor(cfg: ProviderSpec) -> ToolCallExtractor:
    if is_bedrock(cfg):
        return extract_bedrock_tool_calls
    if placement_for(cfg) == SystemPlacement.SIBLING_OBJECT:
        return extract_google_tool_calls
    tc = _tool_call_def(cfg)
    if tc is not None and tc.args_format == "map":
        return extract_anthropic_tool_calls
    return extract_openai_tool_calls


def _tool_call_def(cfg: ProviderSpec):
    from .providers.generated.providers import ProviderName

    return tool_call_config(ProviderName(cfg.name))


#
#
#

@dataclass(frozen=True)
class _MsgText:
    """"""
    role: str
    text: str


@dataclass(frozen=True)
class _MsgCalls:
    """"""
    calls: list[ToolCall]


@dataclass(frozen=True)
class _MsgResult:
    """"""
    result: ToolResult


#
#
#
#
_Msg = _MsgText | _MsgCalls | _MsgResult


def _assert_never(value: NoReturn) -> NoReturn:
    """





"""
    raise TypeError(f"unhandled message variant {type(value).__name__}")


def to_internal(messages: list[Message]) -> list[_Msg]:
    """






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


#
#
#

def transform_flat_content(body: dict[str, Any], msgs: list[_Msg], req: "Request", cfg: ProviderSpec) -> None:
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
        #
        #
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

    body["messages"] = out


def _build_flat_content_parts(req: "Request", cfg: ProviderSpec) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    is_anthropic = placement_for(cfg) == SystemPlacement.TOP_LEVEL_FIELD

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
        #
        #
        #
        #
        #
        #
        #
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
        out.append(
            {
                "role": map_role("user", cfg.role_mappings),
                "content": [{"text": req.user}],
            }
        )
    body["messages"] = out


#
#
#

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


#
#
#

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


#
#
#

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


#
#
#

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


#
from .types import Request, Tool  # noqa: E402  (imported late to avoid circular types)
