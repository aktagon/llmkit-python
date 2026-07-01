"""



"""

from __future__ import annotations

from typing import Any

from llmkit.providers.generated.providers import PROVIDERS
from llmkit.transforms import (
    ToolCall,
    ToolResult,
    extract_anthropic_tool_calls,
    extract_bedrock_tool_calls,
    extract_google_tool_calls,
    extract_openai_tool_calls,
    select_tool_def_transform,
    transform_anthropic_tool_call_msg,
    transform_anthropic_tool_result_msg,
    transform_anthropic_tools,
    transform_bedrock_tool_call_msg,
    transform_bedrock_tool_defs,
    transform_bedrock_tool_result_msg,
    transform_google_function_declarations,
    transform_google_tool_call_msg,
    transform_google_tool_result_msg,
    transform_openai_functions,
    transform_openai_tool_call_msg,
    transform_openai_tool_result_msg,
)
from llmkit.types import Tool


#


def _adder_tool() -> Tool:
    return Tool(
        name="add",
        description="Add two numbers",
        schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
        run=lambda args: str(args["a"] + args["b"]),
    )


def _identity_roles() -> dict[str, str]:
    return {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }


#
#
#


def test_transform_openai_functions_wraps_each_tool() -> None:
    body: dict[str, Any] = {}
    transform_openai_functions(body, [_adder_tool()])
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": _adder_tool().schema,
            },
        }
    ]


def test_transform_anthropic_tools_uses_input_schema_key() -> None:
    body: dict[str, Any] = {}
    transform_anthropic_tools(body, [_adder_tool()])
    #
    assert body["tools"][0]["input_schema"] == _adder_tool().schema
    assert body["tools"][0]["name"] == "add"


def test_transform_google_function_declarations_nests_decls() -> None:
    body: dict[str, Any] = {}
    transform_google_function_declarations(body, [_adder_tool()], "parametersJsonSchema")
    #
    decl = body["tools"][0]["functionDeclarations"][0]
    assert decl["name"] == "add"
    #
    assert decl["parametersJsonSchema"] == _adder_tool().schema
    assert "parameters" not in decl


def test_transform_bedrock_tool_defs_uses_toolconfig_envelope() -> None:
    body: dict[str, Any] = {}
    transform_bedrock_tool_defs(body, [_adder_tool()])
    #
    spec = body["toolConfig"]["tools"][0]["toolSpec"]
    assert spec["name"] == "add"
    assert spec["inputSchema"]["json"] == _adder_tool().schema


#
#
#


def _adder_call() -> ToolCall:
    return ToolCall(id="call_abc", name="add", input={"a": 2, "b": 3})


def test_transform_openai_tool_call_msg_uses_role_assistant_with_tool_calls_array() -> None:
    msg = transform_openai_tool_call_msg([_adder_call()], _identity_roles())
    assert msg["role"] == "assistant"
    #
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_abc"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "add"
    assert isinstance(tc["function"]["arguments"], str)
    assert tc["function"]["arguments"] == '{"a": 2, "b": 3}'


def test_transform_anthropic_tool_call_msg_uses_content_array() -> None:
    msg = transform_anthropic_tool_call_msg([_adder_call()], _identity_roles())
    block = msg["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "call_abc"
    assert block["name"] == "add"
    #
    assert block["input"] == {"a": 2, "b": 3}


def test_transform_google_tool_call_msg_remaps_role_to_model() -> None:
    role_map = {"assistant": "model", "user": "user"}
    msg = transform_google_tool_call_msg([_adder_call()], role_map)
    #
    assert msg["role"] == "model"
    assert msg["parts"][0]["functionCall"]["name"] == "add"
    assert msg["parts"][0]["functionCall"]["args"] == {"a": 2, "b": 3}


def test_transform_bedrock_tool_call_msg_uses_tooluse_envelope() -> None:
    msg = transform_bedrock_tool_call_msg([_adder_call()], _identity_roles())
    tu = msg["content"][0]["toolUse"]
    assert tu["toolUseId"] == "call_abc"
    assert tu["name"] == "add"
    assert tu["input"] == {"a": 2, "b": 3}


#
#
#


def _adder_result() -> ToolResult:
    return ToolResult(tool_use_id="call_abc", content="5")


def test_transform_openai_tool_result_msg_uses_role_tool() -> None:
    msg = transform_openai_tool_result_msg(_adder_result(), _identity_roles())
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_abc"
    assert msg["content"] == "5"


def test_transform_anthropic_tool_result_msg_routes_to_user_role() -> None:
    #
    msg = transform_anthropic_tool_result_msg(_adder_result(), _identity_roles())
    assert msg["role"] == "user"
    block = msg["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_abc"
    assert block["content"] == "5"


def test_transform_google_tool_result_msg_uses_function_response_part() -> None:
    msg = transform_google_tool_result_msg(_adder_result(), _identity_roles())
    fr = msg["parts"][0]["functionResponse"]
    #
    assert fr["name"] == "call_abc"
    assert fr["response"] == {"result": "5"}


def test_transform_bedrock_tool_result_msg_wraps_content_as_text_block() -> None:
    msg = transform_bedrock_tool_result_msg(_adder_result(), _identity_roles())
    tr = msg["content"][0]["toolResult"]
    assert tr["toolUseId"] == "call_abc"
    #
    assert tr["content"] == [{"text": "5"}]


#
#
#


def test_extract_openai_tool_calls_parses_arguments_as_json_string() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 2, "b": 3}',
                            },
                        }
                    ]
                }
            }
        ]
    }
    #
    cfg = type("Cfg", (), {"args_format": "json_string"})()
    calls = extract_openai_tool_calls(raw, cfg)
    assert len(calls) == 1
    assert calls[0].name == "add"
    assert calls[0].input == {"a": 2, "b": 3}


def test_extract_openai_tool_calls_returns_empty_on_no_tool_calls() -> None:
    raw = {"choices": [{"message": {"content": "hello"}}]}
    cfg = type("Cfg", (), {"args_format": "json_string"})()
    assert extract_openai_tool_calls(raw, cfg) == []


def test_extract_anthropic_tool_calls_walks_content_blocks() -> None:
    raw = {
        "content": [
            {"type": "text", "text": "I'll use a tool"},
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "add",
                "input": {"a": 2, "b": 3},
            },
        ]
    }
    calls = extract_anthropic_tool_calls(raw, None)
    assert len(calls) == 1
    assert calls[0].id == "toolu_abc"
    assert calls[0].input == {"a": 2, "b": 3}


def test_extract_google_tool_calls_uses_name_as_id() -> None:
    raw = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "add",
                                "args": {"a": 2, "b": 3},
                            }
                        }
                    ]
                }
            }
        ]
    }
    calls = extract_google_tool_calls(raw, None)
    assert len(calls) == 1
    #
    assert calls[0].id == "add"
    assert calls[0].name == "add"
    assert calls[0].input == {"a": 2, "b": 3}


def test_extract_bedrock_tool_calls_walks_output_message_content() -> None:
    raw = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_abc",
                            "name": "add",
                            "input": {"a": 2, "b": 3},
                        }
                    }
                ]
            }
        }
    }
    calls = extract_bedrock_tool_calls(raw, None)
    assert len(calls) == 1
    assert calls[0].id == "tooluse_abc"
    assert calls[0].name == "add"


def test_extract_handles_malformed_envelopes_gracefully() -> None:
    #
    cfg = type("Cfg", (), {"args_format": "json_string"})()
    assert extract_openai_tool_calls({}, cfg) == []
    assert extract_anthropic_tool_calls({}, None) == []
    assert extract_google_tool_calls({}, None) == []
    assert extract_bedrock_tool_calls({}, None) == []


#
#
#


def test_select_tool_def_transform_openai_returns_openai_functions() -> None:
    cfg = PROVIDERS["openai"]
    fn = select_tool_def_transform(cfg)
    assert fn is transform_openai_functions


def test_select_tool_def_transform_anthropic_returns_anthropic_tools() -> None:
    cfg = PROVIDERS["anthropic"]
    fn = select_tool_def_transform(cfg)
    #
    assert fn is transform_anthropic_tools


def test_select_tool_def_transform_google_emits_parameters_json_schema() -> None:
    cfg = PROVIDERS["google"]
    fn = select_tool_def_transform(cfg)
    #
    #
    #
    body: dict[str, Any] = {}
    fn(body, [_adder_tool()])
    decl = body["tools"][0]["functionDeclarations"][0]
    assert decl["parametersJsonSchema"] == _adder_tool().schema
    assert "parameters" not in decl


def test_select_tool_def_transform_bedrock_returns_bedrock_defs() -> None:
    cfg = PROVIDERS["bedrock"]
    fn = select_tool_def_transform(cfg)
    assert fn is transform_bedrock_tool_defs
