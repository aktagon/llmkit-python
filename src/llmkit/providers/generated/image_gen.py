#

from __future__ import annotations

from dataclasses import dataclass, field

from .providers import ProviderName


@dataclass(frozen=True)
class ImageModelDef:
    model_id: str
    label: str
    aspect_ratios: tuple[str, ...] = field(default_factory=tuple)
    image_sizes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ImageGenDef:
    input_mode: str
    output_mode: str
    max_input_count: int
    gen_endpoint: str
    edit_endpoint: str
    models: tuple[ImageModelDef, ...] = field(default_factory=tuple)


_IMAGE_GEN: dict[ProviderName, ImageGenDef] = {
    ProviderName.GOOGLE: ImageGenDef(
        input_mode="InlineParts",
        output_mode="Base64Inline",
        max_input_count=14,
        gen_endpoint="",
        edit_endpoint="",
        models=(
            ImageModelDef(
                model_id="gemini-3-pro-image-preview",
                label="Nano Banana Pro",
                aspect_ratios=("16:9", "1:1", "21:9", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16"),
                image_sizes=("1K", "2K", "4K"),
            ),
            ImageModelDef(
                model_id="gemini-3.1-flash-image-preview",
                label="Nano Banana 2",
                aspect_ratios=("16:9", "1:1", "1:4", "1:8", "21:9", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16"),
                image_sizes=("1K", "2K", "4K", "512"),
            ),
        ),
    ),
    ProviderName.GROK: ImageGenDef(
        input_mode="JSONInlineRefs",
        output_mode="Base64Inline",
        max_input_count=16,
        gen_endpoint="/v1/images/generations",
        edit_endpoint="/v1/images/edits",
        models=(
            ImageModelDef(
                model_id="grok-imagine-image-quality",
                label="Grok Imagine Quality",
                aspect_ratios=("16:9", "19.5:9", "1:1", "1:2", "20:9", "2:1", "2:3", "3:2", "3:4", "4:3", "9:16", "9:19.5", "9:20", "auto"),
                image_sizes=(),
            ),
        ),
    ),
    ProviderName.OPENAI: ImageGenDef(
        input_mode="MultipartForm",
        output_mode="Base64Inline",
        max_input_count=16,
        gen_endpoint="/v1/images/generations",
        edit_endpoint="/v1/images/edits",
        models=(
            ImageModelDef(
                model_id="gpt-image-1",
                label="GPT Image 1",
                aspect_ratios=(),
                image_sizes=(),
            ),
            ImageModelDef(
                model_id="gpt-image-1-mini",
                label="GPT Image 1 Mini",
                aspect_ratios=(),
                image_sizes=(),
            ),
            ImageModelDef(
                model_id="gpt-image-1.5",
                label="GPT Image 1.5",
                aspect_ratios=(),
                image_sizes=(),
            ),
            ImageModelDef(
                model_id="gpt-image-2",
                label="GPT Image 2",
                aspect_ratios=(),
                image_sizes=(),
            ),
        ),
    ),
    ProviderName.VERTEX: ImageGenDef(
        input_mode="JSONPredict",
        output_mode="Base64Inline",
        max_input_count=1,
        gen_endpoint="",
        edit_endpoint="",
        models=(
            ImageModelDef(
                model_id="imagen-3.0-fast-generate-001",
                label="Imagen 3 Fast",
                aspect_ratios=("16:9", "1:1", "3:4", "4:3", "9:16"),
                image_sizes=(),
            ),
            ImageModelDef(
                model_id="imagen-3.0-generate-002",
                label="Imagen 3",
                aspect_ratios=("16:9", "1:1", "3:4", "4:3", "9:16"),
                image_sizes=(),
            ),
            ImageModelDef(
                model_id="imagen-4.0-generate-preview-06-06",
                label="Imagen 4 Preview",
                aspect_ratios=("16:9", "1:1", "3:4", "4:3", "9:16"),
                image_sizes=(),
            ),
        ),
    ),
}


def image_gen_config(provider: ProviderName) -> ImageGenDef | None:
    return _IMAGE_GEN.get(provider)
