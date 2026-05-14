"""Image generation runtime — mirror of go/image.go.

Pre-flight validation rejects unsupported aspect ratios, sizes, and
reference-image counts before any HTTP call. Dispatch branches on
img_cfg.input_mode (InlineParts → Google generateContent;
MultipartForm → OpenAI Image API with /generations vs /edits picked
dynamically per call based on whether image parts are present).
"""

from __future__ import annotations

import base64
import dataclasses
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .errors import APIError, ValidationError, parse_error
from .http import do_multipart_post_multi, do_post
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
    Typed-builder accumulators (``c.text.text(s)``, ``c.image.image(m, b)``,
    ...) are the canonical user-facing path; users assembling Part lists
    manually can construct Part(text=..., image=MediaRef(...)) directly."""

    text: str = ""
    image: MediaRef | None = None


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
        desugars to parts=[Part(text=prompt)] before serialisation.
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
    # Provider stop signal. Examples per provider:
    #   Google:    "STOP" (ok), "IMAGE_OTHER", "SAFETY", "MAX_TOKENS"
    #   OpenAI Images API: no equivalent field (always empty)
    #   xAI Grok:          no equivalent field (always empty)
    #   Vertex Imagen:     RAI filter reason when content is blocked
    finish_reason: str = ""
    # Free-text provider explanation. Gemini populates this for non-success
    # finish_reason values. Use as user-facing message when len(images) == 0.
    finish_message: str = ""


def generate_image(
    provider: Provider,
    request: ImageRequest,
    *,
    aspect_ratio: str = "",
    image_size: str = "",
    include_text: bool = False,
    quality: str = "",
    output_format: str = "",
    background: str = "",
    count: int | None = None,
    mask: MediaRef | None = None,
    safety_filter: str = "",
    extra_fields: dict[str, Any] | None = None,
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
    # Empty whitelist means "no client-side check; pass through" — used by
    # providers (e.g., OpenAI) that accept arbitrary sizes within documented
    # bounds (plan 020 q1).
    if aspect_ratio and model.aspect_ratios and aspect_ratio not in model.aspect_ratios:
        raise ValidationError(
            field="aspect_ratio",
            message=f"{aspect_ratio} not supported by {request.model}",
        )
    if image_size and model.image_sizes and image_size not in model.image_sizes:
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

    # Per-provider knob validation. Quality/output_format/background are
    # OpenAI-only on the wire; count (n) is OpenAI + xAI; mask is OpenAI
    # edits-only. Mirrors go/image.go.
    if img_cfg.input_mode == "InlineParts":
        if quality:
            raise ValidationError(field="quality", message=f"not supported by {provider.name}")
        if output_format:
            raise ValidationError(field="output_format", message=f"not supported by {provider.name}")
        if background:
            raise ValidationError(field="background", message=f"not supported by {provider.name}")
        if count is not None:
            raise ValidationError(field="count", message=f"not supported by {provider.name}")
        if mask is not None:
            raise ValidationError(field="mask", message=f"not supported by {provider.name}")
        if safety_filter:
            raise ValidationError(field="safety_filter", message=f"not supported by {provider.name}")
    elif img_cfg.input_mode == "JSONInlineRefs":
        if quality:
            raise ValidationError(field="quality", message=f"not supported by {provider.name}")
        if output_format:
            raise ValidationError(field="output_format", message=f"not supported by {provider.name}")
        if background:
            raise ValidationError(field="background", message=f"not supported by {provider.name}")
        if mask is not None:
            raise ValidationError(field="mask", message=f"not supported by {provider.name}")
        if safety_filter:
            raise ValidationError(field="safety_filter", message=f"not supported by {provider.name}")
    elif img_cfg.input_mode == "MultipartForm":
        if mask is not None and image_count == 0:
            raise ValidationError(
                field="mask",
                message="requires at least one image part (edits branch only)",
            )
        if safety_filter:
            raise ValidationError(field="safety_filter", message=f"not supported by {provider.name}")
    elif img_cfg.input_mode == "JSONPredict":
        if quality:
            raise ValidationError(field="quality", message=f"not supported by {provider.name}")
        if output_format:
            raise ValidationError(field="output_format", message=f"not supported by {provider.name}")
        if background:
            raise ValidationError(field="background", message=f"not supported by {provider.name}")

    mws = list(middleware or [])
    base_event = Event(
        op=MiddlewareOp.IMAGE_GENERATION,
        provider=provider.name,
        model=request.model,
    )
    start = time.monotonic()
    fire_pre(mws, base_event)

    try:
        headers = _image_auth_headers(provider, cfg, pname)
        base_url = provider.base_url or cfg.base_url

        try:
            has_images = any(p.image is not None for p in parts)
            if img_cfg.input_mode == "JSONInlineRefs":
                if has_images:
                    body = _build_xai_edit_body(
                        parts, request.model, aspect_ratio, image_size, count, extra_fields
                    )
                    url = base_url + img_cfg.edit_endpoint
                else:
                    body = _build_xai_gen_body(
                        parts, request.model, aspect_ratio, image_size, count, extra_fields
                    )
                    url = base_url + img_cfg.gen_endpoint
                json_body = json.dumps(body).encode("utf-8")
                resp_body = do_post(
                    url,
                    json_body,
                    {**headers, "content-type": "application/json"},
                    timeout=request_timeout,
                )
            elif img_cfg.input_mode == "MultipartForm":
                if has_images:
                    files, fields = _build_openai_edit_multipart(
                        parts, request.model, image_size, quality, output_format, background, count, mask, extra_fields
                    )
                    resp_body, status = do_multipart_post_multi(
                        base_url + img_cfg.edit_endpoint,
                        files,
                        fields,
                        headers,
                        timeout=request_timeout,
                    )
                    if status >= 400:
                        raise parse_error(provider.name, status, resp_body, None)
                else:
                    body = _build_openai_gen_body(
                        parts, request.model, image_size, quality, output_format, background, count, extra_fields
                    )
                    json_body = json.dumps(body).encode("utf-8")
                    resp_body = do_post(
                        base_url + img_cfg.gen_endpoint,
                        json_body,
                        {**headers, "content-type": "application/json"},
                        timeout=request_timeout,
                    )
            elif img_cfg.input_mode == "JSONPredict":
                body = _build_vertex_body(parts, aspect_ratio, count, mask, safety_filter, extra_fields)
                json_body = json.dumps(body).encode("utf-8")
                endpoint = (cfg.endpoint or "").replace("{model}", request.model)
                resp_body = do_post(
                    base_url + endpoint,
                    json_body,
                    {**headers, "content-type": "application/json"},
                    timeout=request_timeout,
                )
            else:
                body = _build_image_body(parts, aspect_ratio, image_size, include_text)
                json_body = json.dumps(body).encode("utf-8")
                url = _build_image_url(provider, cfg, request.model)
                resp_body = do_post(
                    url,
                    json_body,
                    {**headers, "content-type": "application/json"},
                    timeout=request_timeout,
                )
        except APIError as raw_err:
            # parse_error returned above is already an APIError; only wrap raw HTTP errors.
            if raw_err.status_code == 0 or raw_err.message:
                err = parse_error(
                    provider.name,
                    raw_err.status_code,
                    raw_err.message.encode("utf-8") if raw_err.message else b"",
                    None,
                )
                raise err from raw_err
            raise

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
    sugar path), synthesise [Part(text=prompt)]. Both empty or both set raises
    ValidationError."""
    has_prompt = bool(request.prompt)
    has_parts = bool(request.parts)
    if has_prompt and has_parts:
        raise ValidationError(field="parts", message="set prompt or parts, not both")
    if not has_prompt and not has_parts:
        raise ValidationError(field="prompt", message="set either prompt or parts")
    return [Part(text=request.prompt)] if has_prompt else list(request.parts)


def _join_text_parts(parts: list[Part]) -> str:
    return "\n".join(p.text for p in parts if p.image is None and p.text)


def _ext_from_mime(mime: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
    }.get(mime, ".bin")


def _build_openai_gen_body(
    parts: list[Part],
    model: str,
    image_size: str,
    quality: str,
    output_format: str,
    background: str,
    count: int | None,
    extra_fields: dict[str, Any] | None,
) -> dict[str, Any]:
    """JSON body for /v1/images/generations.

    Note: gpt-image-* models always return base64-encoded images via
    ``data[i].b64_json`` and reject the ``response_format`` parameter
    (it belonged to the legacy dall-e-* surface). Don't set it.
    """
    body: dict[str, Any] = {
        "model": model,
        "prompt": _join_text_parts(parts),
    }
    if image_size:
        body["size"] = image_size
    if quality:
        body["quality"] = quality
    if output_format:
        body["output_format"] = output_format
    if background:
        body["background"] = background
    if count is not None:
        body["n"] = count
    for k, v in (extra_fields or {}).items():
        body[k] = v
    return body


def _build_openai_edit_multipart(
    parts: list[Part],
    model: str,
    image_size: str,
    quality: str,
    output_format: str,
    background: str,
    count: int | None,
    mask: MediaRef | None,
    extra_fields: dict[str, Any] | None,
) -> tuple[list[tuple[str, str, str, bytes]], dict[str, str]]:
    """Multipart payload for /v1/images/edits. Each image Part becomes one
    image[] file in caller order; text Parts join into the ``prompt`` field.
    """
    files: list[tuple[str, str, str, bytes]] = []
    idx = 0
    for part in parts:
        if part.image is None:
            continue
        mime = part.image.mime_type or "image/png"
        files.append(
            (
                "image[]",
                f"image-{idx}{_ext_from_mime(mime)}",
                mime,
                part.image.bytes,
            )
        )
        idx += 1
    if mask is not None:
        mask_mime = mask.mime_type or "image/png"
        files.append(
            ("mask", f"mask{_ext_from_mime(mask_mime)}", mask_mime, mask.bytes)
        )
    fields: dict[str, str] = {
        "model": model,
        "prompt": _join_text_parts(parts),
    }
    if image_size:
        fields["size"] = image_size
    if quality:
        fields["quality"] = quality
    if output_format:
        fields["output_format"] = output_format
    if background:
        fields["background"] = background
    if count is not None:
        fields["n"] = str(count)
    for k, v in (extra_fields or {}).items():
        fields[k] = v if isinstance(v, str) else json.dumps(v)
    return files, fields


def _build_xai_gen_body(
    parts: list[Part],
    model: str,
    aspect_ratio: str,
    image_size: str,
    count: int | None,
    extra_fields: dict[str, Any] | None,
) -> dict[str, Any]:
    """JSON body for xAI Grok /v1/images/generations.

    image_size maps to ``resolution`` (xAI's name for the same concept).
    aspect_ratio maps as-is. response_format=b64_json is forced because
    xAI defaults to URL.
    """
    body: dict[str, Any] = {
        "model": model,
        "prompt": _join_text_parts(parts),
        "response_format": "b64_json",
    }
    if aspect_ratio:
        body["aspect_ratio"] = aspect_ratio
    if image_size:
        body["resolution"] = image_size
    if count is not None:
        body["n"] = count
    for k, v in (extra_fields or {}).items():
        body[k] = v
    return body


def _build_xai_edit_body(
    parts: list[Part],
    model: str,
    aspect_ratio: str,
    image_size: str,
    count: int | None,
    extra_fields: dict[str, Any] | None,
) -> dict[str, Any]:
    """JSON body for xAI Grok /v1/images/edits. Single image part →
    ``image: {url: "data:..."}``; multiple → ``images: [...]`` in caller order."""
    body = _build_xai_gen_body(parts, model, aspect_ratio, image_size, count, extra_fields)
    refs: list[dict[str, str]] = []
    for part in parts:
        if part.image is None:
            continue
        mime = part.image.mime_type or "image/png"
        encoded = base64.b64encode(part.image.bytes).decode("ascii")
        refs.append({"url": f"data:{mime};base64,{encoded}"})
    if len(refs) == 1:
        body["image"] = refs[0]
    elif len(refs) > 1:
        body["images"] = refs
    return body


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


def _build_vertex_body(
    parts: list[Part],
    aspect_ratio: str,
    count: int | None,
    mask: Any,
    safety_filter: str,
    extra_fields: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the Vertex AI Imagen :predict request body.

    Vertex uses an instances/parameters envelope: instance carries the
    per-call inputs (prompt, image ref for editing, mask for inpainting);
    parameters carries config (sampleCount, aspectRatio). Extra fields like
    negativePrompt and safetySetting spread into parameters so callers can
    reach Imagen-specific knobs without typed chain methods.
    """
    instance: dict[str, Any] = {"prompt": _join_text_parts(parts)}
    for p in parts:
        if p.image is not None:
            instance["image"] = {
                "bytesBase64Encoded": base64.b64encode(p.image.bytes).decode(
                    "ascii"
                )
            }
            break  # Vertex Imagen takes a single edit-target image
    if mask is not None:
        instance["mask"] = {
            "image": {
                "bytesBase64Encoded": base64.b64encode(mask.bytes).decode("ascii")
            }
        }

    parameters: dict[str, Any] = {"sampleCount": count if count is not None else 1}
    if aspect_ratio:
        parameters["aspectRatio"] = aspect_ratio
    if safety_filter:
        parameters["safetySetting"] = safety_filter
    if extra_fields:
        for k, v in extra_fields.items():
            parameters[k] = v

    return {"instances": [instance], "parameters": parameters}


def _parse_vertex_image_response(raw: dict[str, Any]) -> ImageResponse:
    """Decode Vertex AI Imagen :predict responses. Shape:
    {predictions: [{bytesBase64Encoded, mimeType}]}. Vertex does not return
    token counts in the predict response so Usage stays zero.
    """
    preds = raw.get("predictions") if isinstance(raw, dict) else None
    images: list[ImageData] = []
    finish_reason = ""
    if isinstance(preds, list):
        for entry in preds:
            if not isinstance(entry, dict):
                continue
            if not finish_reason:
                rai = entry.get("raiFilteredReason")
                if isinstance(rai, str) and rai:
                    finish_reason = rai
            b64 = entry.get("bytesBase64Encoded")
            if not isinstance(b64, str) or not b64:
                continue
            mime_val = entry.get("mimeType")
            mime = (
                mime_val if isinstance(mime_val, str) and mime_val else "image/png"
            )
            try:
                decoded = base64.b64decode(b64)
            except (ValueError, TypeError):
                continue
            images.append(ImageData(mime_type=mime, data=decoded))
    return ImageResponse(
        images=images,
        text="",
        tokens=Usage(),
        finish_reason=finish_reason,
    )


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

    if provider_name == "openai":
        return _parse_image_response_data_array(raw, "input_tokens", "output_tokens")
    if provider_name == "grok":
        # xAI reports usage.cost_in_usd_ticks instead of token counts;
        # passing empty field names yields zero tokens (correct, no
        # fabricated values).
        return _parse_image_response_data_array(raw, "", "")
    if provider_name == "vertex":
        return _parse_vertex_image_response(raw)

    images, text, finish_reason, finish_message = _extract_google_image_parts(raw)
    tokens = Usage(
        input=extract_int_path(raw, cfg.usage_input_path),
        output=extract_int_path(raw, cfg.usage_output_path),
    )
    return ImageResponse(
        images=images,
        text=text,
        tokens=tokens,
        finish_reason=finish_reason,
        finish_message=finish_message,
    )


def _parse_image_response_data_array(
    raw: dict[str, Any],
    input_token_field: str,
    output_token_field: str,
) -> ImageResponse:
    """Walk the data[] array shape used by both OpenAI's and xAI's image
    APIs. Decodes data[i].b64_json into ImageData; honors data[i].mime_type
    when echoed back (xAI does, OpenAI does not), defaulting to image/png.
    Concatenates any data[i].revised_prompt into text. Pass empty token-
    field names for providers that don't report counts (xAI reports
    usage.cost_in_usd_ticks instead).
    """
    data = raw.get("data") if isinstance(raw, dict) else None
    images: list[ImageData] = []
    revised: list[str] = []
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            b64 = entry.get("b64_json")
            if isinstance(b64, str) and b64:
                try:
                    decoded = base64.b64decode(b64)
                except (ValueError, TypeError):
                    decoded = b""
                if decoded:
                    echoed = entry.get("mime_type")
                    mime = echoed if isinstance(echoed, str) and echoed else "image/png"
                    images.append(ImageData(mime_type=mime, data=decoded))
            rp = entry.get("revised_prompt")
            if isinstance(rp, str) and rp:
                revised.append(rp)
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if isinstance(usage, dict):
        in_tokens = int(usage.get(input_token_field, 0)) if input_token_field else 0
        out_tokens = int(usage.get(output_token_field, 0)) if output_token_field else 0
    else:
        in_tokens = out_tokens = 0
    tokens = Usage(input=in_tokens, output=out_tokens)
    return ImageResponse(images=images, text="\n".join(revised), tokens=tokens)


def _extract_google_image_parts(
    raw: dict[str, Any],
) -> tuple[list[ImageData], str, str, str]:
    """Return (images, text, finish_reason, finish_message). finish_reason/
    finish_message are pulled from candidates[0]; both stay empty when the
    response carries no such fields."""
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return [], "", "", ""
    first = candidates[0] if isinstance(candidates[0], dict) else {}
    finish_reason = first.get("finishReason") if isinstance(first, dict) else None
    finish_message = first.get("finishMessage") if isinstance(first, dict) else None
    fr_str = finish_reason if isinstance(finish_reason, str) else ""
    fm_str = finish_message if isinstance(finish_message, str) else ""
    content = first.get("content") if isinstance(first, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return [], "", fr_str, fm_str

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
    return images, "".join(text_parts), fr_str, fm_str
