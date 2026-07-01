#

from __future__ import annotations

from .providers import PROVIDERS, ProviderName


def response_text_path(provider: ProviderName) -> str:
    """"""
    return PROVIDERS[provider.value].response_text_path


def usage_paths(provider: ProviderName) -> tuple[str, str]:
    """"""
    config = PROVIDERS[provider.value]
    return config.usage_input_path, config.usage_output_path


def usage_cost_path(provider: ProviderName) -> str:
    """"""
    return PROVIDERS[provider.value].usage_cost_path


def usage_cost_scale(provider: ProviderName) -> float:
    """"""
    return PROVIDERS[provider.value].usage_cost_scale
