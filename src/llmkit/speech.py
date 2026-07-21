"""






"""

from __future__ import annotations

import base64
import dataclasses
import json
import time
from dataclasses import dataclass
from typing import Any

from .errors import APIError, ValidationError, parse_error
from .http import do_post
from .image import _image_auth_headers
from .middleware import fire_post, fire_pre, set_event_error
from .providers.generated.middleware import Event, MiddlewareFn, MiddlewareOp
from .providers.generated.providers import PROVIDERS, ProviderName
from .providers.generated.speech_gen import (
    SpeechGenDef,
    SpeechModelDef,
    speech_gen_config,
)
from .types import Provider

from .structs import AudioData, SpeechResponse, Usage  # noqa: E402,F401


@dataclass
class SpeechRequest:
    """


"""

    model: str = ""
    voice: str = ""
    text: str = ""


def generate_speech(
    provider: Provider,
    request: SpeechRequest,
    *,
    middleware: list[MiddlewareFn] | None = None,
    request_timeout: float = 600.0,
) -> SpeechResponse:
    """



"""
    cfg = PROVIDERS.get(provider.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {provider.name}")
    if not provider.api_key:
        raise ValidationError(field="api_key", message="required")
    if not request.model:
        raise ValidationError(field="model", message="required for speech generation")
    if not request.text:
        raise ValidationError(field="text", message="required for speech generation")
    if not request.voice:
        raise ValidationError(field="voice", message="required for speech generation")

    pname = ProviderName(provider.name)
    sg_cfg = speech_gen_config(pname)
    if sg_cfg is None:
        raise ValidationError(
            field="provider",
            message=f"{provider.name} does not support speech generation",
        )
    model = _find_speech_model(sg_cfg, request.model)
    if model is None:
        raise ValidationError(
            field="model",
            message=f"{request.model} is not a known speech-generation model for {provider.name}",
        )
    if request.voice not in sg_cfg.voices:
        raise ValidationError(
            field="voice",
            message=f"{request.voice} is not a known voice for {provider.name}",
        )

    mws = list(middleware or [])
    base_event = Event(
        op=MiddlewareOp.SPEECH_GENERATION,
        provider=provider.name,
        model=request.model,
    )
    start = time.monotonic()
    fire_pre(mws, base_event)

    try:
        headers = _image_auth_headers(provider, cfg, pname)
        base_url = provider.base_url or cfg.base_url
        url, body = _dispatch_speech_http(cfg, sg_cfg, request, base_url)
        json_body = json.dumps(body).encode("utf-8")
        try:
            resp_body = do_post(
                url,
                json_body,
                {**headers, "content-type": "application/json"},
                timeout=request_timeout,
            )
        except APIError as raw_err:
            raise parse_error(
                provider.name,
                raw_err.status_code,
                raw_err.message.encode("utf-8") if raw_err.message else b"",
                None,
            ) from raw_err

        result = _parse_speech_response(
            provider.name, sg_cfg.audio_response_encoding, model.output_mime, resp_body
        )
    except Exception as exc:
        post_event = dataclasses.replace(
            base_event,
            duration=time.monotonic() - start,
        )
        set_event_error(post_event, exc)
        fire_post(mws, post_event)
        raise

    post_event = dataclasses.replace(
        base_event,
        usage=result.usage,
        duration=time.monotonic() - start,
    )
    fire_post(mws, post_event)
    return result


def _find_speech_model(cfg: SpeechGenDef, model_id: str) -> SpeechModelDef | None:
    for m in cfg.models:
        if m.model_id == model_id:
            return m
    return None


#
#
#
#
def _dispatch_speech_http(
    cfg: Any,
    sg_cfg: SpeechGenDef,
    request: SpeechRequest,
    base_url: str,
) -> tuple[str, dict[str, Any]]:
    endpoint = sg_cfg.gen_endpoint or cfg.endpoint or ""
    url = endpoint if endpoint.startswith("http") else base_url + endpoint
    if sg_cfg.wire_shape == "SpeechOpenAI":
        return url, _build_openai_speech_body(request)
    return url, _build_inworld_speech_body(request)


#
#
def _build_openai_speech_body(request: SpeechRequest) -> dict[str, Any]:
    return {
        "model": request.model,
        "input": request.text,
        "voice": request.voice,
        "response_format": "mp3",
    }


#
#
#
def _build_inworld_speech_body(request: SpeechRequest) -> dict[str, Any]:
    return {
        "text": request.text,
        "voiceId": request.voice,
        "modelId": request.model,
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": 22050,
        },
        "deliveryMode": "BALANCED",
    }


#
#
#
#
def _parse_speech_response(
    provider_name: str, audio_encoding: str, fallback_mime: str, resp_body: bytes
) -> SpeechResponse:
    if audio_encoding == "rawBody":
        return SpeechResponse(
            audio=AudioData(mime_type=fallback_mime, bytes=resp_body), usage=Usage()
        )
    #
    #
    #
    try:
        raw = json.loads(resp_body)
    except json.JSONDecodeError as exc:
        raise APIError(
            provider=provider_name,
            status_code=200,
            message=f"speech response: not valid JSON: {exc}",
        ) from exc
    content = raw.get("audioContent") if isinstance(raw, dict) else None
    if not isinstance(content, str) or not content:
        raise APIError(
            provider=provider_name,
            status_code=200,
            message="speech response: missing or empty audioContent",
        )
    try:
        decoded = base64.b64decode(content, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise APIError(
            provider=provider_name,
            status_code=200,
            message=f"speech response: invalid base64 in audioContent: {exc}",
        ) from exc
    return SpeechResponse(
        audio=AudioData(mime_type=fallback_mime, bytes=decoded), usage=Usage()
    )
