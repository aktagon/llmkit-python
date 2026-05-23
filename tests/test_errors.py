"""

"""

from __future__ import annotations

import json

from llmkit.errors import (
    APIError,
    MiddlewareVetoError,
    ValidationError,
    extract_retry_after,
    parse_error,
)


#


def test_parse_error_openai_envelope() -> None:
    body = json.dumps(
        {
            "error": {
                "message": "Incorrect API key provided",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        }
    ).encode("utf-8")
    err = parse_error("openai", 401, body, headers=None)
    assert err.provider == "openai"
    assert err.status_code == 401
    assert err.message == "Incorrect API key provided"
    assert err.type == "invalid_request_error"
    assert err.retryable is False  # 401 is not retryable


def test_parse_error_anthropic_envelope() -> None:
    body = json.dumps(
        {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limited",
            },
        }
    ).encode("utf-8")
    err = parse_error("anthropic", 429, body, headers=None)
    assert err.message == "Rate limited"
    assert err.type == "rate_limit_error"
    assert err.retryable is True  # 429 IS retryable


def test_parse_error_google_envelope_with_status() -> None:
    body = json.dumps(
        {
            "error": {
                "code": 400,
                "message": "Invalid argument",
                "status": "INVALID_ARGUMENT",
            }
        }
    ).encode("utf-8")
    err = parse_error("google", 400, body, headers=None)
    assert err.message == "Invalid argument"
    assert err.type == "INVALID_ARGUMENT"


def test_parse_error_5xx_marks_retryable() -> None:
    body = json.dumps({"error": {"message": "internal"}}).encode("utf-8")
    err = parse_error("openai", 503, body, headers=None)
    assert err.retryable is True


def test_parse_error_unknown_provider_falls_back_to_raw_body() -> None:
    body = b"plain text error body"
    err = parse_error("nonexistent-provider", 500, body, headers=None)
    assert err.message == "plain text error body"
    assert err.retryable is True


def test_parse_error_malformed_json_falls_back_to_raw_body() -> None:
    body = b"<html>500 server error</html>"
    err = parse_error("openai", 500, body, headers=None)
    assert "html" in err.message  # raw HTML body landed in message
    assert err.retryable is True


def test_parse_error_picks_up_retry_after_from_headers() -> None:
    body = json.dumps({"error": {"message": "Rate limited"}}).encode("utf-8")
    err = parse_error(
        "openai", 429, body, headers={"Retry-After": "30"}
    )
    assert err.retry_after == 30.0


#


def test_extract_retry_after_canonical_case() -> None:
    assert extract_retry_after({"Retry-After": "60"}) == 60.0


def test_extract_retry_after_lowercase_fallback() -> None:
    #
    assert extract_retry_after({"retry-after": "45"}) == 45.0


def test_extract_retry_after_returns_zero_when_missing() -> None:
    assert extract_retry_after({"x-other": "x"}) == 0.0


def test_extract_retry_after_returns_zero_when_none() -> None:
    assert extract_retry_after(None) == 0.0


def test_extract_retry_after_returns_zero_on_non_numeric() -> None:
    #
    #
    assert extract_retry_after({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}) == 0.0


#


def test_api_error_str() -> None:
    err = APIError(provider="openai", status_code=401, message="bad key")
    assert str(err) == "openai: bad key (401)"


def test_validation_error_str() -> None:
    err = ValidationError(field="aspect_ratio", message="4:5 not supported")
    assert str(err) == "validation: aspect_ratio - 4:5 not supported"


def test_middleware_veto_error_str_carries_cause() -> None:
    cause = ValueError("hit budget cap")
    err = MiddlewareVetoError(cause=cause)
    assert "hit budget cap" in str(err)
