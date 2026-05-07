"""Image generation runtime — mirror of go/image.go.

Pre-flight validation rejects unsupported aspect ratios, sizes, and
reference-image counts before any HTTP call. The body shape matches
Google's generateContent endpoint; OpenAI/Vertex variants will dispatch
on cfg.input_mode when those land.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .errors import APIError, ValidationError, parse_error
from .http import do_post
from .middleware import fire_post, fire_pre
from .paths import extract_int_path
from .providers.generated.image_gen import (
    ImageGenDef,
    ImageModelDef,
    image_gen_config,
)
from .providers.generated.middleware import Event, MiddlewareFn, MiddlewareOp, Usage
from .providers.generated.providers import PROVIDERS, ProviderName
from .providers.generated.request import AuthScheme, auth_scheme
from .types import Provider


@dataclass
class MediaRef:
    """Inline media payload (mime type + raw bytes). Reused by every Part
    variant that carries non-text content (image today; audio/video/document
    as those land)."""

    mime_type: str = ""
    bytes: bytes = b""


@dataclass
class Part:
    """Universal multimodal input atom. Exactly one of text or image is
    set; both empty or both set is invalid (rejected by pre-flight).
    Construct via the package-level Text() and Image() helpers."""

    text: str = ""
    image: MediaRef | None = None


def Text(s: str) -> Part:  # noqa: N802 — public constructor; PascalCase for parity with Go/TS.
    """Construct a text-bearing Part."""
    return Part(text=s)


def Image(mime: str, data: bytes) -> Part:  # noqa: N802 — public constructor.
    """Construct an image-bearing Part. mime is the IANA media type
    (e.g., 'image/png'); data is the raw bytes (not base64-encoded)."""
    return Part(image=MediaRef(mime_type=mime, bytes=data))


@dataclass
class ImageData:
    """One decoded image returned by the provider."""

    mime_type: str = ""
    data: bytes = b""


@dataclass
class ImageRequest:
    """Image-generation request. Model is required.

    Input is provided in one of two mutually-exclusive forms:
      - prompt: terse sugar for the text-only hot path. Internally
        desugars to parts=[Text(prompt)] before serialisation.
      - parts: canonical multimodal sequence; required for editing and
        compositional generation where caller-controlled ordering matters.

    Pre-flight validation requires exactly one of prompt or parts to be
    non-empty (XOR). Image-typed parts respect img_cfg.max_input_count.
    """

    model: str = ""
    prompt: str = ""
    parts: list[Part] = field(default_factory=list)


@dataclass
class ImageResponse:
    images: list[ImageData] = field(default_factory=list)
    text: str = ""
    tokens: Usage = field(default_factory=Usage)


def generate_image(
    provider: Provider,
    request: ImageRequest,
    *,
    aspect_ratio: str = "",
    image_size: str = "",
    include_text: bool = False,
    middleware: list[MiddlewareFn] | None = None,
    request_timeout: float = 600.0,
) -> ImageResponse:
    """Produce one or more images from a text prompt, optionally conditioned
    on reference images for editing/composition.
    """
    if not provider.api_key:
        raise ValidationError(field="api_key", message="required")
    if not request.model:
        raise ValidationError(field="model", message="required for image generation")

    parts = _normalize_image_parts(request)

    cfg = PROVIDERS.get(provider.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {provider.name}")

    pname = ProviderName(provider.name)
    img_cfg = image_gen_config(pname)
    if img_cfg is None:
        raise ValidationError(
            field="provider",
            message=f"{provider.name} does not support image generation",
        )
    model = _find_image_model(img_cfg, request.model)
    if model is None:
        raise ValidationError(
            field="model",
            message=f"{request.model} is not a known image-generation model for {provider.name}",
        )
    if aspect_ratio and aspect_ratio not in model.aspect_ratios:
        raise ValidationError(
            field="aspect_ratio",
            message=f"{aspect_ratio} not supported by {request.model}",
        )
    if image_size and image_size not in model.image_sizes:
        raise ValidationError(
            field="image_size",
            message=f"{image_size} not supported by {request.model}",
        )
    image_count = sum(1 for p in parts if p.image is not None)
    if image_count > img_cfg.max_input_count:
        raise ValidationError(
            field="parts",
            message=(
                f"{image_count} image parts exceeds maximum "
                f"{img_cfg.max_input_count} for {provider.name}"
            ),
        )

    mws = list(middleware or [])
    base_event = Event(
        op=MiddlewareOp.IMAGE_GENERATION,
        provider=provider.name,
        model=request.model,
    )
    start = time.monotonic()
    fire_pre(mws, base_event)

    try:
        body = _build_image_body(parts, aspect_ratio, image_size, include_text)
        json_body = json.dumps(body).encode("utf-8")
        url = _build_image_url(provider, cfg, request.model)
        headers = _image_auth_headers(provider, cfg, pname)

        try:
            resp_body = do_post(url, json_body, headers, timeout=request_timeout)
        except APIError as raw_err:
            err = parse_error(
                provider.name,
                raw_err.status_code,
                raw_err.message.encode("utf-8"),
                None,
            )
            raise err from raw_err

        result = _parse_image_response(provider.name, resp_body, cfg)
    except Exception as exc:
        post_event = dataclasses.replace(
            base_event,
            err=str(exc),
            duration=time.monotonic() - start,
        )
        fire_post(mws, post_event)
        raise

    post_event = dataclasses.replace(
        base_event,
        usage=result.tokens,
        duration=time.monotonic() - start,
    )
    fire_post(mws, post_event)
    return result


def _find_image_model(cfg: ImageGenDef, model_id: str) -> ImageModelDef | None:
    for m in cfg.models:
        if m.model_id == model_id:
            return m
    return None


def _normalize_image_parts(request: ImageRequest) -> list[Part]:
    """Enforce the XOR rule and produce the canonical list[Part] the rest
    of the pipeline operates on. When only prompt is set (the text-only
    sugar path), synthesise [Text(prompt)]. Both empty or both set raises
    ValidationError."""
    has_prompt = bool(request.prompt)
    has_parts = bool(request.parts)
    if has_prompt and has_parts:
        raise ValidationError(field="parts", message="set prompt or parts, not both")
    if not has_prompt and not has_parts:
        raise ValidationError(field="prompt", message="set either prompt or parts")
    return [Text(request.prompt)] if has_prompt else list(request.parts)


def _build_image_body(
    parts: list[Part],
    aspect_ratio: str,
    image_size: str,
    include_text: bool,
) -> dict[str, Any]:
    wire: list[dict[str, Any]] = []
    for p in parts:
        if p.image is not None:
            wire.append(
                {
                    "inlineData": {
                        "mimeType": p.image.mime_type,
                        "data": base64.b64encode(p.image.bytes).decode("ascii"),
                    }
                }
            )
        else:
            wire.append({"text": p.text})

    modalities = ["TEXT", "IMAGE"] if include_text else ["IMAGE"]
    generation_config: dict[str, Any] = {"responseModalities": modalities}
    img_config: dict[str, Any] = {}
    if aspect_ratio:
        img_config["aspectRatio"] = aspect_ratio
    if image_size:
        img_config["imageSize"] = image_size
    if img_config:
        generation_config["imageConfig"] = img_config

    return {
        "contents": [{"parts": wire}],
        "generationConfig": generation_config,
    }


def _build_image_url(p: Provider, cfg: Any, model: str) -> str:
    base = p.base_url or cfg.base_url
    endpoint = cfg.endpoint
    if auth_scheme(ProviderName(p.name)) == AuthScheme.QUERY_PARAM_KEY:
        endpoint = endpoint + "?" + cfg.auth_query_param + "=" + p.api_key
    endpoint = endpoint.replace("{model}", model)
    endpoint = endpoint.replace("{apiKey}", p.api_key)
    return base + endpoint


def _image_auth_headers(p: Provider, cfg: Any, pname: ProviderName) -> dict[str, str]:
    headers: dict[str, str] = {}
    scheme = auth_scheme(pname)
    if scheme == AuthScheme.BEARER_TOKEN:
        headers[cfg.auth_header] = cfg.auth_prefix + " " + p.api_key
    elif scheme == AuthScheme.HEADER_API_KEY:
        headers[cfg.auth_header] = p.api_key
    if cfg.required_header:
        headers[cfg.required_header] = cfg.required_header_value
    return headers


def _parse_image_response(provider_name: str, body: bytes, cfg: Any) -> ImageResponse:
    try:
        raw = json.loads(body)
    except ValueError as exc:
        raise APIError(
            provider=provider_name,
            message=f"unmarshal image response: {exc}",
            status_code=0,
        ) from exc

    images, text = _extract_google_image_parts(raw)
    tokens = Usage(
        input=extract_int_path(raw, cfg.usage_input_path),
        output=extract_int_path(raw, cfg.usage_output_path),
    )
    return ImageResponse(images=images, text=text, tokens=tokens)


def _extract_google_image_parts(raw: dict[str, Any]) -> tuple[list[ImageData], str]:
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return [], ""
    first = candidates[0] if isinstance(candidates[0], dict) else {}
    content = first.get("content") if isinstance(first, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return [], ""

    images: list[ImageData] = []
    text_parts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        inline = part.get("inlineData")
        if isinstance(inline, dict):
            data = inline.get("data")
            mime = inline.get("mimeType", "")
            if isinstance(data, str):
                try:
                    decoded = base64.b64decode(data)
                except (ValueError, TypeError):
                    continue
                images.append(ImageData(mime_type=mime, data=decoded))
        text = part.get("text")
        if isinstance(text, str) and text:
            text_parts.append(text)
    return images, "".join(text_parts)
