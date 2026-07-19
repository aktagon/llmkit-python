"""VULN-001 regression: a malformed catalogue base_url must not leak the
spliced API key. urllib.request.urlopen raises a bare ValueError for a
malformed URL whose message embeds the full URL (including any `?key=...`
query param); _http_get must re-raise without interpolating that exception."""

from __future__ import annotations

import pytest

from llmkit.models import ErrModelsUnavailable, _http_get


def test_http_get_malformed_url_does_not_leak_api_key() -> None:
    secret = "sk-super-secret-catalogue-key"
    malformed_url = f"not-a-valid-url?key={secret}"

    with pytest.raises(ErrModelsUnavailable) as exc_info:
        _http_get(malformed_url, {})

    message = str(exc_info.value)
    assert secret not in message
    assert "key=" not in message
