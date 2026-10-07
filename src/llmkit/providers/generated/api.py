# Code generated — DO NOT EDIT.

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class APISubOptionDef:
    py_func: str
    py_param_type: str


@dataclass(frozen=True)
class APIOptionDef:
    py_func: str
    sub_options: tuple[APISubOptionDef, ...] = ()


@dataclass(frozen=True)
class APIEntryPointDef:
    py_func: str
    py_param_type: str
    comment: str = ""


@dataclass(frozen=True)
class APIResponseFieldDef:
    py_field_name: str
    py_field_type: str
    source_path: str


API_OPTIONS: tuple[APIOptionDef, ...] = (
    APIOptionDef(
        py_func="with_add_tool",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_aspect_ratio",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_background",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_bytes",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_caching",
        sub_options=(
            APISubOptionDef(
                py_func="cache_t_t_l",
                py_param_type="float",
            ),
        ),
    ),
    APIOptionDef(
        py_func="with_count",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_file",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_filename",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_frequency_penalty",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_history",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_image",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_image_size",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_include_text",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_lyrics",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_mask",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_max_tokens",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_max_tool_iterations",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_mime_type",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_model",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_output_format",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_output_u_r_i",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_path",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_presence_penalty",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_protocol",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_provider",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_quality",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_raw",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_reasoning_effort",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_safety_filter",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_safety_settings",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_schema",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_seed",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_stop_sequences",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_system",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_temperature",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_text",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_thinking_budget",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_top_k",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_top_p",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_voice",
        sub_options=(
        ),
    ),
    APIOptionDef(
        py_func="with_with_capability",
        sub_options=(
        ),
    ),
)


API_ENTRY_POINTS: tuple[APIEntryPointDef, ...] = (
    APIEntryPointDef(
        py_func="agent",
        py_param_type="Provider",
    ),
    APIEntryPointDef(
        py_func="batch",
        py_param_type="list[Request]",
    ),
    APIEntryPointDef(
        py_func="decode_response",
        py_param_type="ChatWireShape",
    ),
    APIEntryPointDef(
        py_func="encode_response",
        py_param_type="ChatWireShape",
    ),
    APIEntryPointDef(
        py_func="generate_image",
        py_param_type="ImageRequest",
    ),
    APIEntryPointDef(
        py_func="generate_music",
        py_param_type="MusicRequest",
    ),
    APIEntryPointDef(
        py_func="generate_speech",
        py_param_type="SpeechRequest",
    ),
    APIEntryPointDef(
        py_func="poll",
        py_param_type="BatchHandle",
    ),
    APIEntryPointDef(
        py_func="prompt",
        py_param_type="Request",
    ),
    APIEntryPointDef(
        py_func="prompt_stream",
        py_param_type="Request",
    ),
    APIEntryPointDef(
        py_func="submit",
        py_param_type="TranscriptionRequest",
    ),
    APIEntryPointDef(
        py_func="submit",
        py_param_type="VideoRequest",
    ),
    APIEntryPointDef(
        py_func="supports",
        py_param_type="Capability",
    ),
    APIEntryPointDef(
        py_func="transcribe",
        py_param_type="TranscriptionRequest",
    ),
    APIEntryPointDef(
        py_func="upload_file",
        py_param_type="Bytes",
    ),
    APIEntryPointDef(
        py_func="wait",
        py_param_type="BatchHandle",
    ),
    APIEntryPointDef(
        py_func="wait",
        py_param_type="TranscriptionHandle",
    ),
    APIEntryPointDef(
        py_func="wait",
        py_param_type="VideoHandle",
    ),
)


CACHE_RESPONSE_FIELDS: tuple[APIResponseFieldDef, ...] = (
    APIResponseFieldDef(
        py_field_name="audio",
        py_field_type="list[AudioData]",
        source_path="audioPath",
    ),
    APIResponseFieldDef(
        py_field_name="audio",
        py_field_type="AudioData",
        source_path="audioPath",
    ),
    APIResponseFieldDef(
        py_field_name="cache_read",
        py_field_type="int",
        source_path="cacheReadTokensPath",
    ),
    APIResponseFieldDef(
        py_field_name="cache_write",
        py_field_type="int",
        source_path="cacheWriteTokensPath",
    ),
    APIResponseFieldDef(
        py_field_name="finish_message",
        py_field_type="str",
        source_path="finishMessagePath",
    ),
    APIResponseFieldDef(
        py_field_name="finish_reason",
        py_field_type="str",
        source_path="finishReasonPath",
    ),
    APIResponseFieldDef(
        py_field_name="images",
        py_field_type="list[ImageData]",
        source_path="candidates[0].content.parts[*].inlineData",
    ),
    APIResponseFieldDef(
        py_field_name="input",
        py_field_type="int",
        source_path="usageInputPath",
    ),
    APIResponseFieldDef(
        py_field_name="output",
        py_field_type="int",
        source_path="usageOutputPath",
    ),
    APIResponseFieldDef(
        py_field_name="reasoning",
        py_field_type="int",
        source_path="reasoningTokensPath",
    ),
    APIResponseFieldDef(
        py_field_name="text",
        py_field_type="str",
        source_path="candidates[0].content.parts[*].text",
    ),
    APIResponseFieldDef(
        py_field_name="videos",
        py_field_type="list[VideoData]",
        source_path="video",
    ),
)


def cache_usage_paths(provider: str) -> tuple[str, str]:
    """Return (write_tokens_path, read_tokens_path) for a provider."""
    from .caching import caching_config
    from .providers import ProviderName
    config = caching_config(ProviderName(provider))
    if config is None:
        return "", ""
    return config.write_tokens_path, config.read_tokens_path
