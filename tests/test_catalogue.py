"""Phase 2.5 catalogue tests (ADR-019). Mirror of Go go/catalogue_test.go
and TS ts/tests/catalogue.test.ts."""

from __future__ import annotations

import asyncio

import pytest

from llmkit.builders import anthropic, cerebras, openai
from llmkit.models import (
    ErrModelsNotSupported,
    ErrModelsScope,
    ErrModelsUnavailable,
)
from llmkit.types import Capability, Provider


def test_models_list_returns_compiled_in_catalogue() -> None:
    c = anthropic("test-key")
    models = c.models.list()
    assert len(models) > 0
    # sorted by (provider, id) -> first entry is anthropic
    assert models[0].provider.name == "anthropic"


def test_models_with_capability_narrows_to_image_generation() -> None:
    c = openai("test-key")
    all_models = c.models.list()
    image_only = c.models.with_capability(Capability.IMAGE_GENERATION).list()
    assert len(image_only) > 0
    assert len(image_only) < len(all_models)
    for m in image_only:
        assert Capability.IMAGE_GENERATION in m.capabilities


def test_models_with_capability_does_not_mutate_parent() -> None:
    c = openai("test-key")
    parent = c.models
    parent.with_capability(Capability.IMAGE_GENERATION)
    all_models = parent.list()
    filtered = parent.with_capability(Capability.IMAGE_GENERATION).list()
    assert len(all_models) > len(filtered)


def test_models_get_returns_known_model() -> None:
    c = anthropic("test-key")
    got = c.models.get("claude-opus-4-7")
    assert got is not None
    assert got.id == "claude-opus-4-7"


def test_models_get_returns_none_for_unknown_id() -> None:
    c = anthropic("test-key")
    assert c.models.get("nonexistent-model-xyz") is None


def test_providers_list_returns_configured_provider_with_endpoint() -> None:
    c = anthropic("test-key")
    got = c.providers.list()
    assert len(got) == 1
    assert got[0].name == "anthropic"


def test_providers_list_empty_for_endpointless_provider() -> None:
    c = cerebras("test-key")
    assert c.providers.list() == []


def test_providers_supported_returns_full_sdk_roster() -> None:
    c = anthropic("test-key")
    supported = c.providers.supported()
    assert len(supported) >= 10
    names = [p.name for p in supported]
    # Wire-format names — guards against str(Enum) leaking "ProviderName.ANTHROPIC".
    assert "anthropic" in names
    assert "openai" in names
    assert "google" in names
    assert not any(n.startswith("ProviderName.") for n in names)


def test_scoped_list_raises_not_supported_for_endpointless_provider() -> None:
    c = cerebras("test-key")
    with pytest.raises(ErrModelsNotSupported):
        asyncio.run(c.models.provider(Provider(name="cerebras", api_key="k")).list())


def test_scoped_list_raises_unavailable_for_phase3_stub() -> None:
    c = anthropic("test-key")
    with pytest.raises(ErrModelsUnavailable):
        asyncio.run(c.models.provider(Provider(name="anthropic", api_key="k")).list())


def test_scoped_get_raises_unavailable_for_phase3_stub() -> None:
    c = anthropic("test-key")
    with pytest.raises(ErrModelsUnavailable):
        asyncio.run(
            c.models.provider(Provider(name="anthropic", api_key="k")).get("claude-opus-4-7")
        )


def test_scoped_raw_chain_is_immutable() -> None:
    c = anthropic("test-key")
    scoped = c.models.provider(Provider(name="anthropic", api_key="k"))
    forked = scoped.raw()
    assert scoped.raw_flag is False
    assert forked.raw_flag is True


def test_error_sentinels_default_messages() -> None:
    # Exercises each sentinel's default constructor so coverage sees __init__.
    assert "models endpoint" in str(ErrModelsNotSupported())
    assert "unavailable" in str(ErrModelsUnavailable())
    assert "scope" in str(ErrModelsScope())


def test_models_live_captures_unavailable_into_errors_map() -> None:
    c = anthropic("test-key")
    res = asyncio.run(c.models.live())
    assert res.models == []
    assert "anthropic" in res.errors
    assert "unavailable" in res.errors["anthropic"]
