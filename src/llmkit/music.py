"""







"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .errors import APIError, ValidationError, parse_error
from .http import do_post
from .image import Part, _image_auth_headers
from .middleware import fire_post, fire_pre
from .providers.generated.middleware import Event, MiddlewareFn, MiddlewareOp, Usage
from .providers.generated.music_gen import (
    MusicGenDef,
    MusicModelDef,
    music_gen_config,
)
from .providers.generated.providers import PROVIDERS, ProviderName
from .providers.generated.request import AuthScheme, auth_scheme
from .types import Provider

from .structs import AudioData, MusicResponse  # noqa: E402,F401


@dataclass
class MusicRequest:
    """












"""

    model: str = ""
    prompt: str = ""
    parts: list[Part] = field(default_factory=list)


#
#
#
#
#
#


def generate_music(
    provider: Provider,
    request: MusicRequest,
    *,
    middleware: list[MiddlewareFn] | None = None,
    request_timeout: float = 600.0,
    raw: bool = False,
) -> MusicResponse:
    """







"""
    if not provider.api_key:
        raise ValidationError(field="api_key", message="required")
    if not request.model:
        raise ValidationError(field="model", message="required for music generation")

    parts = _normalize_music_parts(request)
    for i, part in enumerate(parts):
        set_count = 0
        if part.text:
            set_count += 1
        if part.lyrics:
            set_count += 1
        if part.image is not None:
            raise ValidationError(
                field=f"parts[{i}]",
                message="music generation does not accept image parts",
            )
        if set_count != 1:
            raise ValidationError(
                field=f"parts[{i}]",
                message="must have exactly one of text or lyrics set",
            )

    cfg = PROVIDERS.get(provider.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {provider.name}")

    pname = ProviderName(provider.name)
    mg_cfg = music_gen_config(pname)
    if mg_cfg is None:
        raise ValidationError(
            field="provider",
            message=f"{provider.name} does not support music generation",
        )
    model = _find_music_model(mg_cfg, request.model)
    if model is None:
        raise ValidationError(
            field="model",
            message=f"{request.model} is not a known music-generation model for {provider.name}",
        )
    #
    #
    #

    mws = list(middleware or [])
    base_event = Event(
        op=MiddlewareOp.MUSIC_GENERATION,
        provider=provider.name,
        model=request.model,
    )
    start = time.monotonic()
    fire_pre(mws, base_event)

    try:
        headers = _image_auth_headers(provider, cfg, pname)
        base_url = provider.base_url or cfg.base_url
        url, body = _dispatch_music_http(
            provider, cfg, mg_cfg, request.model, parts, base_url
        )
        json_body = json.dumps(body).encode("utf-8")
        try:
            resp_body = do_post(
                url,
                json_body,
                {**headers, "content-type": "application/json"},
                timeout=request_timeout,
            )
        except APIError as raw_err:
            err = parse_error(
                provider.name,
                raw_err.status_code,
                raw_err.message.encode("utf-8") if raw_err.message else b"",
                None,
            )
            raise err from raw_err

        result = _parse_music_response(mg_cfg.wire_shape, model.output_mime, resp_body)
        if raw:
            try:
                result.raw = json.loads(resp_body)
            except ValueError:
                result.raw = None
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
        usage=result.usage,
        duration=time.monotonic() - start,
    )
    fire_post(mws, post_event)
    return result


def _find_music_model(cfg: MusicGenDef, model_id: str) -> MusicModelDef | None:
    for m in cfg.models:
        if m.model_id == model_id:
            return m
    return None


def _normalize_music_parts(request: MusicRequest) -> list[Part]:
    """

"""
    has_prompt = bool(request.prompt)
    has_parts = bool(request.parts)
    if has_prompt and has_parts:
        raise ValidationError(field="parts", message="set prompt or parts, not both")
    if not has_prompt and not has_parts:
        raise ValidationError(field="prompt", message="set either prompt or parts")
    return [Part(text=request.prompt)] if has_prompt else list(request.parts)


#
#
#
#
#
#
#
#
#
#
def _dispatch_music_http(
    provider: Provider,
    cfg: Any,
    mg_cfg: MusicGenDef,
    model: str,
    parts: list[Part],
    base_url: str,
) -> tuple[str, dict[str, Any]]:
    if mg_cfg.wire_shape == "MusicPredict":
        body = _build_vertex_music_body(parts)
        endpoint = mg_cfg.gen_endpoint or cfg.endpoint or ""
        endpoint = endpoint.replace("{model}", model)
        return base_url + endpoint, body
    if mg_cfg.wire_shape == "MusicMinimax":
        body = _build_minimax_music_body(parts, model)
        url = (
            mg_cfg.gen_endpoint
            if mg_cfg.gen_endpoint.startswith("http")
            else base_url + mg_cfg.gen_endpoint
        )
        return url, body
    #
    body = _build_gemini_music_body(parts)
    return _build_music_url(provider, cfg, mg_cfg, model), body


def _build_vertex_music_body(parts: list[Part]) -> dict[str, Any]:
    """


"""
    prompt = _join_prompt_text(parts)
    lyrics = _join_lyrics_text(parts)
    if lyrics:
        prompt = prompt + "\n" + lyrics if prompt else lyrics
    return {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1},
    }


def _build_gemini_music_body(parts: list[Part]) -> dict[str, Any]:
    """

"""
    wire: list[dict[str, Any]] = []
    for p in parts:
        if p.lyrics:
            wire.append({"text": p.lyrics})
        else:
            wire.append({"text": p.text})
    return {
        "contents": [{"parts": wire}],
        "generationConfig": {"responseModalities": ["AUDIO"]},
    }


def _build_minimax_music_body(parts: list[Part], model: str) -> dict[str, Any]:
    """

"""
    body: dict[str, Any] = {
        "model": model,
        "prompt": _join_prompt_text(parts),
        "output_format": "hex",
        "audio_setting": {
            "sample_rate": 44100,
            "bitrate": 128000,
            "format": "mp3",
        },
    }
    lyrics = _join_lyrics_text(parts)
    if lyrics:
        body["lyrics"] = lyrics
    return body


def _join_prompt_text(parts: list[Part]) -> str:
    return "\n".join(p.text for p in parts if p.text)


def _join_lyrics_text(parts: list[Part]) -> str:
    return "\n".join(p.lyrics for p in parts if p.lyrics)


def _build_music_url(p: Provider, cfg: Any, mg_cfg: MusicGenDef, model: str) -> str:
    """

"""
    base = p.base_url or cfg.base_url
    endpoint = mg_cfg.gen_endpoint or cfg.endpoint or ""
    if auth_scheme(ProviderName(p.name)) == AuthScheme.QUERY_PARAM_KEY:
        sep = "&" if "?" in endpoint else "?"
        endpoint = endpoint + sep + cfg.auth_query_param + "=" + p.api_key
    endpoint = endpoint.replace("{model}", model)
    endpoint = endpoint.replace("{apiKey}", p.api_key)
    return base + endpoint


def _parse_music_response(
    wire_shape: str, fallback_mime: str, body: bytes
) -> MusicResponse:
    """

"""
    from .structs import MusicResponse  # deferred to mirror image.py pattern

    try:
        raw = json.loads(body)
    except ValueError as exc:
        raise APIError(
            message=f"unmarshal music response: {exc}",
            status_code=0,
        ) from exc

    if wire_shape == "MusicPredict":
        return _parse_vertex_music_response(raw, fallback_mime)
    if wire_shape == "MusicMinimax":
        return _parse_minimax_music_response(raw, fallback_mime)
    return _parse_gemini_music_response(raw, fallback_mime)


def _parse_vertex_music_response(
    raw: dict[str, Any], fallback_mime: str
) -> MusicResponse:
    """
"""
    from .structs import MusicResponse

    preds = raw.get("predictions") if isinstance(raw, dict) else None
    audio: list[AudioData] = []
    finish_reason = ""
    if isinstance(preds, list):
        for entry in preds:
            if not isinstance(entry, dict):
                continue
            if not finish_reason:
                rai = entry.get("raiFilteredReason")
                if isinstance(rai, str) and rai:
                    finish_reason = rai
            b64 = entry.get("audioContent")
            if not isinstance(b64, str) or not b64:
                b64 = entry.get("bytesBase64Encoded")
            if not isinstance(b64, str) or not b64:
                continue
            mime_val = entry.get("mimeType")
            mime = mime_val if isinstance(mime_val, str) and mime_val else fallback_mime
            try:
                decoded = base64.b64decode(b64)
            except (ValueError, binascii.Error):
                continue
            audio.append(AudioData(mime_type=mime, bytes=decoded))
    return MusicResponse(
        audio=audio, text="", usage=Usage(), finish_reason=finish_reason
    )


def _parse_gemini_music_response(
    raw: dict[str, Any], fallback_mime: str
) -> MusicResponse:
    """
"""
    from .structs import MusicResponse

    candidates = raw.get("candidates") if isinstance(raw, dict) else None
    if not isinstance(candidates, list) or not candidates:
        return MusicResponse(audio=[], text="", usage=Usage())
    first = candidates[0] if isinstance(candidates[0], dict) else {}
    content = first.get("content") if isinstance(first, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None

    audio: list[AudioData] = []
    text_parts: list[str] = []
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData")
            if isinstance(inline, dict):
                data = inline.get("data")
                mime_val = inline.get("mimeType")
                mime = (
                    mime_val
                    if isinstance(mime_val, str) and mime_val
                    else fallback_mime
                )
                if isinstance(data, str):
                    try:
                        decoded = base64.b64decode(data)
                    except (ValueError, binascii.Error):
                        continue
                    audio.append(AudioData(mime_type=mime, bytes=decoded))
            text = part.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
    finish_reason = first.get("finishReason") if isinstance(first, dict) else None
    fr_str = finish_reason if isinstance(finish_reason, str) else ""
    return MusicResponse(
        audio=audio,
        text="".join(text_parts),
        usage=Usage(),
        finish_reason=fr_str,
    )


def _parse_minimax_music_response(
    raw: dict[str, Any], fallback_mime: str
) -> MusicResponse:
    """
"""
    from .structs import MusicResponse

    audio: list[AudioData] = []
    data = raw.get("data") if isinstance(raw, dict) else None
    if isinstance(data, dict):
        h = data.get("audio")
        if isinstance(h, str) and h:
            try:
                decoded = bytes.fromhex(h)
            except ValueError:
                decoded = b""
            if decoded:
                audio.append(AudioData(mime_type=fallback_mime, bytes=decoded))
    finish_message = ""
    base_resp = raw.get("base_resp") if isinstance(raw, dict) else None
    if isinstance(base_resp, dict):
        msg = base_resp.get("status_msg")
        if isinstance(msg, str) and msg and msg != "success":
            finish_message = msg
    return MusicResponse(
        audio=audio, text="", usage=Usage(), finish_message=finish_message
    )
