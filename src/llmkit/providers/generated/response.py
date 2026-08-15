# Code generated — DO NOT EDIT.

from __future__ import annotations

from dataclasses import dataclass

from .providers import PROVIDERS, ProviderName


def response_text_path(provider: ProviderName) -> str:
    """"""
    return PROVIDERS[provider.value].response_text_path


@dataclass(frozen=True)
class ResponseTextConfig:
    """












"""

    blocks_path: str
    marker_path: str
    marker_value: str
    value_path: str


RESPONSE_TEXT_CONFIGS: dict[str, ResponseTextConfig] = {
    "ChatAnthropic": ResponseTextConfig(blocks_path="content", marker_path="type", marker_value="text", value_path="text"),
    "ChatBedrock": ResponseTextConfig(blocks_path="output.message.content", marker_path="text", marker_value="", value_path="text"),
    "ChatGoogle": ResponseTextConfig(blocks_path="candidates[0].content.parts", marker_path="text", marker_value="", value_path="text"),
}


def response_text_config(chat_wire_shape: str) -> ResponseTextConfig | None:
    """


"""
    return RESPONSE_TEXT_CONFIGS.get(chat_wire_shape)


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
