"""Unit tests for request building. No network, no API keys."""

from __future__ import annotations

import json

import pytest

import llmkit
from llmkit.client import _build_request, _build_url
from llmkit.providers.generated.providers import PROVIDERS


def test_anthropic_builds_top_level_system() -> None:
    cfg = PROVIDERS["anthropic"]
    body, headers = _build_request(
        llmkit.Provider(
            name="anthropic", api_key="sk-ant-test", model="claude-sonnet-4-6"
        ),
        llmkit.Request(system="Be terse.", user="Hi"),
        llmkit.Options(temperature=0.3, max_tokens=100),
        cfg,
    )
    assert body["model"] == "claude-sonnet-4-6"
    assert body["system"] == "Be terse."
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.3
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"


# The thinking-budget dotted-path nesting test and the per-model
# max-tokens key table (BUG-001 / ADR-024) migrated to the
# wire-conformance suite (ADR-028 M2): the options-anthropic and
# options-openai-* fixtures in test_request_wire.py witness those bodies
# byte-for-byte across all four SDKs.


def test_openai_puts_system_in_messages_array() -> None:
    cfg = PROVIDERS["openai"]
    body, headers = _build_request(
        llmkit.Provider(name="openai", api_key="sk-openai-test"),
        llmkit.Request(system="Be terse.", user="Hi"),
        llmkit.Options(temperature=0.5),
        cfg,
    )
    assert body["messages"][0] == {"role": "system", "content": "Be terse."}
    assert body["messages"][1] == {"role": "user", "content": "Hi"}
    assert "system" not in body
    assert headers["Authorization"] == "Bearer sk-openai-test"


def test_google_places_system_in_sibling_object() -> None:
    # The generationConfig wrapped-options asserts migrated to the
    # options-google wire fixture (ADR-028 M2); this test's remaining
    # subject is system placement (PlacementSiblingObject) and the
    # contents shape — M4 surfaces not yet fixture-covered.
    cfg = PROVIDERS["google"]
    body, _ = _build_request(
        llmkit.Provider(name="google", api_key="AIza-test"),
        llmkit.Request(system="Be terse.", user="Hi"),
        llmkit.Options(temperature=0.7, max_tokens=200),
        cfg,
    )
    assert body["system_instruction"] == {"parts": [{"text": "Be terse."}]}
    assert body["contents"][0]["parts"][0]["text"] == "Hi"


def test_bedrock_uses_text_block_arrays_and_inference_config() -> None:
    cfg = PROVIDERS["bedrock"]
    body, _ = _build_request(
        llmkit.Provider(name="bedrock", api_key="AKIA-test"),
        llmkit.Request(system="You are a chatbot.", user="Hi"),
        llmkit.Options(temperature=0.4, max_tokens=50),
        cfg,
    )
    assert body["system"] == [{"text": "You are a chatbot."}]
    assert body["messages"][0]["content"] == [{"text": "Hi"}]
    assert body["inferenceConfig"]["temperature"] == 0.4
    assert body["inferenceConfig"]["maxTokens"] == 50


def test_google_tool_result_resolves_function_name() -> None:
    """ADR-026 #2: Google's wire identifies a tool result by function NAME, but
    the universal ToolResult carries only tool_use_id. On the Text/batch path a
    user supplies a history where the id differs from the name (unlike the
    agent, whose extractor sets id==name), so the result's functionResponse.name
    must resolve back to the function name via the preceding tool-call turn — not
    echo the raw id."""
    cfg = PROVIDERS["google"]
    req = llmkit.Request(
        messages=[
            llmkit.Message(role="user", content="weather in Paris?"),
            llmkit.Message(
                role="assistant",
                tool_calls=[
                    llmkit.ToolCall(
                        id="call_abc123", name="get_weather", input={"city": "Paris"}
                    )
                ],
            ),
            llmkit.Message(
                role="tool",
                tool_result=llmkit.ToolResult(
                    tool_use_id="call_abc123", content="sunny, 21C"
                ),
            ),
        ]
    )
    body, _ = _build_request(
        llmkit.Provider(name="google", api_key="AIza-test"),
        req,
        llmkit.Options(),
        cfg,
    )
    contents = body["contents"]
    assert len(contents) == 3
    fr = contents[2]["parts"][0]["functionResponse"]
    assert fr["name"] == "get_weather"


def test_google_url_appends_api_key_as_query_param() -> None:
    cfg = PROVIDERS["google"]
    url = _build_url(
        llmkit.Provider(name="google", api_key="AIza-key", model="gemini-2.5-flash"),
        cfg,
    )
    assert "gemini-2.5-flash" in url
    assert "key=AIza-key" in url


def test_validation_rejects_empty_api_key() -> None:
    import asyncio

    from llmkit.builders import new_client

    with pytest.raises(llmkit.ValidationError) as ei:
        c = new_client("anthropic", "")
        asyncio.run(c.text.prompt("hi"))
    assert ei.value.field == "api_key"


def test_validation_rejects_unsupported_option() -> None:
    # Anthropic does not support frequency_penalty per the ontology.
    import asyncio

    from llmkit.builders import new_client

    with pytest.raises(llmkit.ValidationError) as ei:
        c = new_client("anthropic", "sk-test")
        asyncio.run(c.text.frequency_penalty(0.5).prompt("hi"))
    assert ei.value.field == "frequency_penalty"


def test_sigv4_headers_have_required_fields() -> None:
    from llmkit.sigv4 import sign_sigv4

    url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/test/converse"
    body = b'{"messages":[]}'
    headers = sign_sigv4(
        url=url,
        body=body,
        access_key="AKIAIOSFODNN7EXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        session_token="",
        region="us-east-1",
        service="bedrock",
    )
    assert headers["Host"] == "bedrock-runtime.us-east-1.amazonaws.com"
    assert headers["X-Amz-Content-Sha256"]
    assert headers["X-Amz-Date"]
    assert headers["Authorization"].startswith(
        "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE"
    )
    assert "SignedHeaders=" in headers["Authorization"]
    assert "Signature=" in headers["Authorization"]


def test_providers_registry_has_all_expected_keys() -> None:
    expected = {
        "openai",
        "anthropic",
        "google",
        "grok",
        "bedrock",
        "openrouter",
        "ollama",
    }
    assert expected.issubset(llmkit.PROVIDERS.keys())


def test_usage_cost_extracted_for_openrouter() -> None:
    """BUG-005 / ADR-027: OpenRouter reports usage.cost (USD) -> Usage.cost."""
    from llmkit.client import _parse_response

    body = json.dumps(
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.00042},
        }
    )
    resp = _parse_response("openrouter", body.encode())
    assert resp.usage.cost == 0.00042


def test_usage_cost_grok_ticks_to_usd() -> None:
    """ADR-027 usageCostScale: xAI reports cost_in_usd_ticks (1 USD = 1e10
    ticks), so scale 1e-10 converts to USD. 2856000 ticks = $0.0002856."""
    from llmkit.client import _parse_response

    body = json.dumps(
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 136,
                "completion_tokens": 100,
                "cost_in_usd_ticks": 2856000,
            },
        }
    )
    resp = _parse_response("grok", body.encode())
    assert resp.usage.cost == 0.0002856


def test_usage_cost_zero_for_no_cost_provider() -> None:
    """OpenAI declares no usage_cost_path, so a stray cost field is ignored."""
    from llmkit.client import _parse_response

    body = json.dumps(
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.99},
        }
    )
    resp = _parse_response("openai", body.encode())
    assert resp.usage.cost == 0.0


def test_reasoning_tokens_extracted_for_openai() -> None:
    """OpenAI o1/o3/o4 expose reasoning_tokens via completion_tokens_details."""
    from llmkit.client import _parse_response

    body = json.dumps(
        {
            "choices": [{"message": {"content": "reasoned answer"}}],
            "usage": {
                "prompt_tokens": 40,
                "completion_tokens": 25,
                "completion_tokens_details": {"reasoning_tokens": 17},
            },
        }
    )
    resp = _parse_response("openai", body.encode())
    assert resp.usage.input == 40
    assert resp.usage.output == 25
    assert resp.usage.reasoning == 17


def test_reasoning_tokens_zero_for_unreported_provider() -> None:
    """Anthropic does not report reasoning tokens separately; Usage.reasoning stays 0."""
    from llmkit.client import _parse_response

    body = json.dumps(
        {
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
    )
    resp = _parse_response("anthropic", body.encode())
    assert resp.usage.reasoning == 0


def test_google_safety_settings_written_as_top_level_field() -> None:
    from llmkit.types import SafetySetting

    cfg = PROVIDERS["google"]
    opts = llmkit.Options(
        safety_settings=[
            SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        ]
    )
    body, _ = _build_request(
        llmkit.Provider(name="google", api_key="k"),
        llmkit.Request(user="hi"),
        opts,
        cfg,
    )
    assert "safetySettings" in body
    assert body["safetySettings"] == [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
    ]


def test_openai_safety_settings_silently_dropped() -> None:
    from llmkit.types import SafetySetting

    cfg = PROVIDERS["openai"]
    opts = llmkit.Options(
        safety_settings=[
            SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        ]
    )
    body, _ = _build_request(
        llmkit.Provider(name="openai", api_key="k"),
        llmkit.Request(user="hi"),
        opts,
        cfg,
    )
    assert "safetySettings" not in body
