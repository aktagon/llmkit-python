"""VULN-001 regression: a malformed catalogue base_url must not leak the
spliced API key. urllib.request.urlopen raises a bare ValueError for a
malformed URL whose message embeds the full URL (including any `?key=...`
query param); _http_get must re-raise without interpolating that exception
NOR chaining it as __cause__/__context__ (which would resurface via
logging.exception, exc_info=True, traceback.format_exc, or an uncaught
exception printout)."""

from __future__ import annotations

import traceback

import pytest

from llmkit.models import ErrModelsUnavailable, _http_get


def _full_chain_text(exc: BaseException) -> str:
    """Everything a traceback print would show: the exception's own str(),
    PLUS any chained __cause__/__context__ frames."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def test_http_get_malformed_url_does_not_leak_api_key() -> None:
    secret = "sk-super-secret-catalogue-key"
    malformed_url = f"not-a-valid-url?key={secret}"

    with pytest.raises(ErrModelsUnavailable) as exc_info:
        _http_get(malformed_url, {})

    exc = exc_info.value
    message = str(exc)
    assert secret not in message
    assert "key=" not in message

    # The traceback-chain vector: __cause__/__context__ must not carry the
    # key-bearing ValueError either.
    chain = _full_chain_text(exc)
    assert secret not in chain
    assert "key=" not in chain
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True
