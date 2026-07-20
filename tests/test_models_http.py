"""





"""

from __future__ import annotations

import traceback

import pytest

from llmkit.models import ErrModelsUnavailable, _http_get


def _full_chain_text(exc: BaseException) -> str:
    """
"""
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

    #
    #
    chain = _full_chain_text(exc)
    assert secret not in chain
    assert "key=" not in chain
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True
