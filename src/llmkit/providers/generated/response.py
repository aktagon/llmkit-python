# Code generated — DO NOT EDIT.

from __future__ import annotations

from dataclasses import dataclass

from .providers import PROVIDERS, ProviderName


def response_text_path(provider: ProviderName) -> str:
    """JSONPath into the provider response that yields the assistant text."""
    return PROVIDERS[provider.value].response_text_path


@dataclass(frozen=True)
class ResponseTextConfig:
    """Where the assistant's text sits in a block-ARRAY response.

    Located by discriminator, not by array position: a leading thinking
    block or non-text part shifts text out from under a fixed path
    (BUG-053).

    marker_path == ''                    every element is a text block
    marker_path set, marker_value == ''  element is text if the key is PRESENT
    both set                             element is text if the key EQUALS the value

    marker_value is also a WRITE instruction: encode_response stamps it
    onto the block it writes, so an emitted body reads back through this
    same table.
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
    """Text-block selector for a wire shape, or None if text is a plain scalar.

    None SELECTS the response_text_path reader above; it does not mean the
    shape has no text."""
    return RESPONSE_TEXT_CONFIGS.get(chat_wire_shape)


def usage_paths(provider: ProviderName) -> tuple[str, str]:
    """(input_tokens_path, output_tokens_path) in the provider response."""
    config = PROVIDERS[provider.value]
    return config.usage_input_path, config.usage_output_path


def usage_cost_path(provider: ProviderName) -> str:
    """Dotted path to provider-reported cost, or "" when unreported (ADR-027)."""
    return PROVIDERS[provider.value].usage_cost_path


def usage_cost_scale(provider: ProviderName) -> float:
    """Multiplier converting the reported cost to USD (xAI ticks -> 1e-10); default 1.0 (ADR-027)."""
    return PROVIDERS[provider.value].usage_cost_scale
