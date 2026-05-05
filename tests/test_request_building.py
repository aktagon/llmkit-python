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
        llmkit.Provider(name="anthropic", api_key="sk-ant-test", model="claude-sonnet-4-6"),
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


def test_anthropic_thinking_budget_nests_dotted_path_with_extras() -> None:
    """Regression: thinking.budget_tokens must nest into {thinking: {...}},
    not be a literal top-level "thinking.budget_tokens" key. Anthropic
    silently ignores unknown top-level keys, so this only shows up as a
    body-shape check. extra_fields_json adds {"type":"enabled"} as a sibling.
    """
    cfg = PROVIDERS["anthropic"]
    body, _headers = _build_request(
        llmkit.Provider(name="anthropic", api_key="sk", model="claude-sonnet-4-6"),
        llmkit.Request(user="hi"),
        llmkit.Options(thinking_budget=1024),
        cfg,
    )
    assert "thinking.budget_tokens" not in body
    assert body["thinking"] == {"budget_tokens": 1024, "type": "enabled"}


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


def test_google_wraps_options_in_generation_config() -> None:
    cfg = PROVIDERS["google"]
    body, _ = _build_request(
        llmkit.Provider(name="google", api_key="AIza-test"),
        llmkit.Request(system="Be terse.", user="Hi"),
        llmkit.Options(temperature=0.7, max_tokens=200),
        cfg,
    )
    assert body["system_instruction"] == {"parts": [{"text": "Be terse."}]}
    assert body["contents"][0]["parts"][0]["text"] == "Hi"
    assert body["generationConfig"]["temperature"] == 0.7
    assert body["generationConfig"]["max_output_tokens"] == 200
    assert "temperature" not in body
    assert "max_output_tokens" not in body


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


def test_google_url_appends_api_key_as_query_param() -> None:
    cfg = PROVIDERS["google"]
    url = _build_url(
        llmkit.Provider(name="google", api_key="AIza-key", model="gemini-2.5-flash"),
        cfg,
    )
    assert "gemini-2.5-flash" in url
    assert "key=AIza-key" in url


def test_validation_rejects_empty_api_key() -> None:
    with pytest.raises(llmkit.ValidationError) as ei:
        llmkit.prompt(
            provider=llmkit.Provider(name="anthropic", api_key=""),
            request=llmkit.Request(user="hi"),
        )
    assert ei.value.field == "api_key"


def test_validation_rejects_unsupported_option() -> None:
    # Anthropic does not support frequency_penalty per the ontology.
    with pytest.raises(llmkit.ValidationError) as ei:
        llmkit.prompt(
            provider=llmkit.Provider(name="anthropic", api_key="sk-test"),
            request=llmkit.Request(user="hi"),
            frequency_penalty=0.5,
        )
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
    assert headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE")
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


def test_reasoning_tokens_extracted_for_openai() -> None:
    """OpenAI o1/o3/o4 expose reasoning_tokens via completion_tokens_details."""
    from llmkit.client import _parse_response

    body = json.dumps({
        "choices": [{"message": {"content": "reasoned answer"}}],
        "usage": {
            "prompt_tokens": 40,
            "completion_tokens": 25,
            "completion_tokens_details": {"reasoning_tokens": 17},
        },
    })
    resp = _parse_response("openai", body.encode())
    assert resp.tokens.input == 40
    assert resp.tokens.output == 25
    assert resp.tokens.reasoning == 17


def test_reasoning_tokens_zero_for_unreported_provider() -> None:
    """Anthropic does not report reasoning tokens separately; Usage.reasoning stays 0."""
    from llmkit.client import _parse_response

    body = json.dumps({
        "content": [{"type": "text", "text": "hello"}],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    })
    resp = _parse_response("anthropic", body.encode())
    assert resp.tokens.reasoning == 0
