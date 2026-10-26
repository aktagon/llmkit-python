# Code generated — DO NOT EDIT.

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Final

from .providers import ProviderName

# Carried inputs: the inputs a request variant carries (ADR-096).
TEXT_ONLY: Final = "TextOnly"
WITH_REFERENCE_IMAGES: Final = "WithReferenceImages"
WITH_SEED_FRAME: Final = "WithSeedFrame"
WITH_LAST_FRAME: Final = "WithLastFrame"

# Image request shapes: each names one request body (ADR-096).
INLINE_PARTS: Final = "InlineParts"
JSON_GENERATIONS_OPENAI: Final = "JSONGenerationsOpenAI"
MULTIPART_FORM: Final = "MultipartForm"
JSON_INLINE_REFS: Final = "JSONInlineRefs"
# Deprecated: no provider accepts this since 2026-10-15 (Vertex Imagen shut down, BUG-101); removed at the next major.
JSON_PREDICT: Final = "JSONPredict"
JSON_GENERATIONS: Final = "JSONGenerations"
JSON_GENERATIONS_INPUT_REFS: Final = "JSONGenerationsInputRefs"


@dataclass(frozen=True)
class RequestVariant:
    """One wire request of an image or video config, keyed by the inputs
    it carries (ADR-096). shape is "" on a video variant; endpoint ""
    means the provider's main endpoint template. carries lists the carried
    inputs; a first-and-last-frame variant carries two."""

    carries: tuple[str, ...]
    shape: str
    endpoint: str


@dataclass(frozen=True)
class ImageModelDef:
    model_id: str
    label: str
    aspect_ratios: tuple[str, ...] = field(default_factory=tuple)
    image_sizes: tuple[str, ...] = field(default_factory=tuple)
    # Images llmkit serializes when the wire shape fixes the count
    # (e.g. Grok's single-seed slot); 0 = no llmkit limit, the provider
    # decides volume (BUG-011).
    max_input_images: int = 0
    # Whether llmkit sends Image Parts for this model; False means the
    # runtime rejects them (BUG-096).
    supports_image_input: bool = False


@dataclass(frozen=True)
class ImageGenDef:
    output_mode: str
    # Response wire family selecting the response parser (BUG-024).
    response_shape: str
    # Dotted-from-root usage-token paths; empty when unreported.
    usage_input_path: str
    usage_output_path: str
    usage_reasoning_path: str
    # One request per carried input, TextOnly first (ADR-096).
    variants: tuple[RequestVariant, ...] = field(default_factory=tuple)
    models: tuple[ImageModelDef, ...] = field(default_factory=tuple)


_IMAGE_GEN: dict[ProviderName, ImageGenDef] = {
    ProviderName.GOOGLE: ImageGenDef(
        output_mode="Base64Inline",
        response_shape="GoogleParts",
        usage_input_path="usageMetadata.promptTokenCount",
        usage_output_path="usageMetadata.candidatesTokenCount",
        usage_reasoning_path="usageMetadata.thoughtsTokenCount",
        variants=(
            RequestVariant(carries=("TextOnly",), shape="InlineParts", endpoint=""),
            RequestVariant(carries=("WithReferenceImages",), shape="InlineParts", endpoint=""),
        ),
        models=(
            ImageModelDef(
                model_id="gemini-nano-banana-2.1",
                label="Nano Banana 2.1",
                aspect_ratios=("16:9", "1:1", "1:4", "1:8", "21:9", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16"),
                image_sizes=("1K", "2K", "4K"),
                max_input_images=0,
                supports_image_input=True,
            ),
        ),
    ),
    ProviderName.GROK: ImageGenDef(
        output_mode="Base64Inline",
        response_shape="DataArrayB64Json",
        usage_input_path="",
        usage_output_path="",
        usage_reasoning_path="",
        variants=(
            RequestVariant(carries=("TextOnly",), shape="JSONInlineRefs", endpoint="/v1/images/generations"),
            RequestVariant(carries=("WithReferenceImages",), shape="JSONInlineRefs", endpoint="/v1/images/edits"),
        ),
        models=(
            ImageModelDef(
                model_id="grok-imagine-image-2.0",
                label="Grok Imagine Image 2.0",
                aspect_ratios=("16:9", "19.5:9", "1:1", "1:2", "20:9", "21:9", "2:1", "2:3", "3:2", "3:4", "4:3", "9:16", "9:19.5", "9:20", "auto"),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="grok-imagine-image-quality",
                label="Grok Imagine Quality",
                aspect_ratios=("16:9", "19.5:9", "1:1", "1:2", "20:9", "2:1", "2:3", "3:2", "3:4", "4:3", "9:16", "9:19.5", "9:20", "auto"),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
        ),
    ),
    ProviderName.OPENAI: ImageGenDef(
        output_mode="Base64Inline",
        response_shape="DataArrayB64Json",
        usage_input_path="usage.input_tokens",
        usage_output_path="usage.output_tokens",
        usage_reasoning_path="",
        variants=(
            RequestVariant(carries=("TextOnly",), shape="JSONGenerationsOpenAI", endpoint="/v1/images/generations"),
            RequestVariant(carries=("WithReferenceImages",), shape="MultipartForm", endpoint="/v1/images/edits"),
        ),
        models=(
            ImageModelDef(
                model_id="gpt-image-1",
                label="GPT Image 1",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="gpt-image-1-mini",
                label="GPT Image 1 Mini",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="gpt-image-1.5",
                label="GPT Image 1.5",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="gpt-image-2",
                label="GPT Image 2",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="gpt-image-2.5-flare",
                label="GPT Image 2.5 Flare",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="gpt-image-2.5-sunburst",
                label="GPT Image 2.5 Sunburst",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
        ),
    ),
    ProviderName.OPENROUTER: ImageGenDef(
        output_mode="Base64Inline",
        response_shape="DataArrayB64Json",
        usage_input_path="usage.prompt_tokens",
        usage_output_path="usage.completion_tokens",
        usage_reasoning_path="",
        variants=(
            RequestVariant(carries=("TextOnly",), shape="JSONGenerations", endpoint="/v1/images"),
            RequestVariant(carries=("WithReferenceImages",), shape="JSONGenerationsInputRefs", endpoint="/v1/images"),
        ),
        models=(
            ImageModelDef(
                model_id="bytedance-seed/seedream-4.5",
                label="Seedream 4.5",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="bytedance-seed/seedream-5-0-flash",
                label="Seedream 5.0 Flash",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="google/gemini-2.5-flash-image",
                label="Nano Banana (Gemini 2.5 Flash Image)",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="google/gemini-3-pro-image",
                label="Nano Banana Pro (Gemini 3 Pro Image)",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="google/gemini-3.1-flash-image",
                label="Nano Banana 2 (Gemini 3.1 Flash Image)",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="google/gemini-3.1-flash-lite-image",
                label="Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="meta/muse-image",
                label="Muse Image",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=False,
            ),
            ImageModelDef(
                model_id="openai/gpt-5-image",
                label="GPT-5 Image",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="openai/gpt-5-image-mini",
                label="GPT-5 Image Mini",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="openai/gpt-5.4-image-2",
                label="GPT-5.4 Image 2",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="openai/gpt-image-2",
                label="GPT Image 2",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="openai/gpt-image-2.5-sunburst",
                label="GPT Image 2.5 Sunburst",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
            ImageModelDef(
                model_id="tencent/hy-image-v3.5-preview",
                label="Hy Image 3.5 Preview",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=True,
            ),
        ),
    ),
    ProviderName.RECRAFT: ImageGenDef(
        output_mode="Base64Inline",
        response_shape="DataArrayB64Json",
        usage_input_path="",
        usage_output_path="",
        usage_reasoning_path="",
        variants=(
            RequestVariant(carries=("TextOnly",), shape="JSONGenerations", endpoint="/v1/images/generations"),
        ),
        models=(
            ImageModelDef(
                model_id="recraftv3",
                label="Recraft V3",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=False,
            ),
            ImageModelDef(
                model_id="recraftv3_vector",
                label="Recraft V3 (vector / SVG)",
                aspect_ratios=(),
                image_sizes=(),
                max_input_images=0,
                supports_image_input=False,
            ),
        ),
    ),
}


def image_gen_config(provider: ProviderName) -> ImageGenDef | None:
    return _IMAGE_GEN.get(provider)
