"""ADR-038: the `providers` namespace (providers.info / providers.list) is the
narrow public per-provider metadata access — the public replacement for reaching
into the internal spec (BUG-012). The import is consumer-style (`from llmkit
import providers`, no `generated` segment); a missing re-export fails it at
collection time. Values are a projection of provider A-Box facts; the field-set
assertion guards against the projection silently widening back toward the
37-field spec.
"""

from __future__ import annotations

import dataclasses

from llmkit import ProviderName, providers


def test_info_projects_anthropic_metadata() -> None:
    info = providers.info(ProviderName.ANTHROPIC)
    assert info.id == ProviderName.ANTHROPIC
    assert info.slug == "anthropic"
    assert info.env_var == "ANTHROPIC_API_KEY"
    assert info.default_model == "claude-sonnet-4-6"
    assert info.base_url == "https://api.anthropic.com"
    assert info.browser_callable is False


def test_info_projects_exactly_the_contract_fields() -> None:
    field_names = [f.name for f in dataclasses.fields(providers.ProviderInfo)]
    assert field_names == [
        "id",
        "slug",
        "env_var",
        "default_model",
        "base_url",
        "browser_callable",
    ]


def test_browser_callable_is_the_cors_fact() -> None:
    # ADR-035: true only for google (CORS-verified), false otherwise.
    assert providers.info(ProviderName.GOOGLE).browser_callable is True
    assert providers.info(ProviderName.GROK).browser_callable is False


def test_list_enumerates_every_provider_sorted_by_slug() -> None:
    all_info = providers.list()
    assert len(all_info) == len(list(ProviderName))
    slugs = [i.slug for i in all_info]
    assert slugs == sorted(slugs)
