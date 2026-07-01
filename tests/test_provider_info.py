"""






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


def test_info_projects_exactly_the_contract_fields() -> None:
    field_names = [f.name for f in dataclasses.fields(providers.ProviderInfo)]
    assert field_names == ["id", "slug", "env_var", "default_model", "base_url"]


def test_list_enumerates_every_provider_sorted_by_slug() -> None:
    all_info = providers.list()
    assert len(all_info) == len(list(ProviderName))
    slugs = [i.slug for i in all_info]
    assert slugs == sorted(slugs)
