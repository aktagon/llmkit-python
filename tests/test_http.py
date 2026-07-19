"""VULN-001 regression, main request path (http.py). A malformed URL makes
urllib.request.Request(url, ...) raise a bare ValueError whose message
embeds the full URL — including any spliced API key query param. Every
http.py entry point routes request construction through `_new_request`,
which redacts the URL and drops the chain (`from None`) so neither
str(exc) nor a traceback print (logging.exception / exc_info=True /
traceback.format_exc / an uncaught exception) can resurface the key."""

from __future__ import annotations

import traceback

import pytest

from llmkit.http import do_get, do_post


def _full_chain_text(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def test_do_get_malformed_url_does_not_leak_api_key() -> None:
    secret = "sk-super-secret-provider-key"
    malformed_url = f"not-a-valid-url?key={secret}"

    with pytest.raises(ValueError) as exc_info:
        do_get(malformed_url, {})

    exc = exc_info.value
    message = str(exc)
    assert secret not in message
    assert "key=" not in message

    chain = _full_chain_text(exc)
    assert secret not in chain
    assert "key=" not in chain
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True


def test_do_post_malformed_url_does_not_leak_api_key() -> None:
    secret = "sk-super-secret-provider-key"
    malformed_url = f"not-a-valid-url?key={secret}"

    with pytest.raises(ValueError) as exc_info:
        do_post(malformed_url, b"{}", {})

    exc = exc_info.value
    message = str(exc)
    assert secret not in message
    assert "key=" not in message

    chain = _full_chain_text(exc)
    assert secret not in chain
    assert "key=" not in chain
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True
