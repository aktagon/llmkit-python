"""Speech generation (text-to-speech) runtime — mirror of go/speech.go and
ts/src/speech.ts (ADR-049).

Pre-flight validation (model + text or lines required; voice or speakers;
provider supports speech; model known; speakers and style the model takes)
runs before any HTTP call. Voices are free-form: there is no catalogue. Five
wire shapes: SpeechInworld, SpeechOpenAI, SpeechGemini, SpeechMiniMax and
SpeechXAI. Sync, single AudioData.
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
from .image import _image_auth_headers
from .middleware import fire_post, fire_pre, set_event_error
from .providers.generated.middleware import Event, MiddlewareFn, MiddlewareOp
from .providers.generated.providers import PROVIDERS, ProviderName
from .providers.generated.request import AuthScheme, auth_scheme
from .providers.generated.speech_gen import (
    SpeechGenDef,
    SpeechModelDef,
    speech_gen_config,
)
from .types import Provider, SpeechLine, SpeechSpeaker

from .structs import AudioData, SpeechResponse, Usage  # noqa: E402,F401


@dataclass
class SpeechRequest:
    """Text-to-speech request. Text is the single utterance to speak
    (single-turn, no Message/Role wrapper — ADR-049 SPK-003) and is empty when
    lines are set; voice is a free-form voice id; model is required. Speakers
    and lines make a multi-speaker request; style is an optional delivery
    style (BUG-091)."""

    model: str = ""
    voice: str = ""
    text: str = ""
    style: str = ""
    speakers: list[SpeechSpeaker] = field(default_factory=list)
    lines: list[SpeechLine] = field(default_factory=list)


def generate_speech(
    provider: Provider,
    request: SpeechRequest,
    *,
    middleware: list[MiddlewareFn] | None = None,
) -> SpeechResponse:
    """Synthesize speech audio from text.

    Internal helper — the public surface is Speech.generate in
    llmkit/builders/speech.py.
    """
    cfg = PROVIDERS.get(provider.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {provider.name}")
    if not provider.api_key:
        raise ValidationError(field="api_key", message="required")
    if not request.model:
        raise ValidationError(field="model", message="required for speech generation")
    _validate_speech_inputs(request)

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
    _validate_speech_model(request, model)

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
        # Speech base: explicit override > the config's distinct speech base
        # (MiniMax, HANDOFF-094) > chat base, the music base contract.
        base_url = provider.base_url or sg_cfg.base_url or cfg.base_url
        url, body = _dispatch_speech_http(cfg, sg_cfg, request, base_url, provider.api_key)
        json_body = json.dumps(body).encode("utf-8")
        try:
            resp_body = do_post(
                url,
                json_body,
                {**headers, "content-type": "application/json"},
                timeout=provider.timeout,
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


def _validate_speech_inputs(request: SpeechRequest) -> None:
    """Check the request on its own, before the provider config is read
    (BUG-091). The messages are the same in every SDK."""
    if not request.lines and not request.text:
        raise ValidationError(field="text", message="required for speech generation")
    if request.lines and request.text:
        raise ValidationError(field="text", message="must be empty when lines are set")
    if not request.speakers:
        if not request.voice:
            raise ValidationError(field="voice", message="required for speech generation")
    else:
        if request.voice:
            raise ValidationError(field="voice", message="cannot be set together with speakers")
        if request.style:
            raise ValidationError(
                field="style",
                message="cannot be set together with speakers; style each line instead",
            )
        if not request.lines:
            raise ValidationError(field="lines", message="required when speakers are set")
    names = {s.name for s in request.speakers}
    for line in request.lines:
        if line.speaker not in names:
            raise ValidationError(field="lines", message=f"{line.speaker} is not a declared speaker")


def _validate_speech_model(request: SpeechRequest, model: SpeechModelDef) -> None:
    """Check the request against the model's speaker and style facts (BUG-091)."""
    if request.speakers:
        if model.max_speakers == 0:
            raise ValidationError(
                field="speakers",
                message=f"{model.model_id} does not support multiple speakers",
            )
        if len(request.speakers) != model.max_speakers:
            raise ValidationError(
                field="speakers",
                message=f"exactly {model.max_speakers} speakers are required, got {len(request.speakers)}",
            )
    styled = bool(request.style) or any(line.style for line in request.lines)
    if styled and not model.supports_speech_style:
        raise ValidationError(
            field="style", message=f"{model.model_id} does not support a speech style"
        )


def _find_speech_model(cfg: SpeechGenDef, model_id: str) -> SpeechModelDef | None:
    for m in cfg.models:
        if m.model_id == model_id:
            return m
    return None


# _dispatch_speech_http picks a wire shape per provider config (never by
# provider name). SpeechInworld is a flat-JSON POST whose response carries
# base64 audio; SpeechOpenAI is a flat-JSON POST whose response body is the raw
# audio bytes; SpeechGemini is an Interactions API POST whose audio sits in
# steps[]; SpeechMiniMax is a flat-JSON POST whose reply carries hex audio at
# data.audio; SpeechXAI is a flat-JSON POST whose response body is the raw
# audio bytes. Query-param-key providers (Google) carry the key in the URL, as
# on every other Google call.
def _dispatch_speech_http(
    cfg: Any,
    sg_cfg: SpeechGenDef,
    request: SpeechRequest,
    base_url: str,
    api_key: str,
) -> tuple[str, dict[str, Any]]:
    endpoint = sg_cfg.gen_endpoint or cfg.endpoint or ""
    url = endpoint if endpoint.startswith("http") else base_url + endpoint
    if auth_scheme(ProviderName(cfg.name)) == AuthScheme.QUERY_PARAM_KEY:
        sep = "&" if "?" in url else "?"
        url = url + sep + cfg.auth_query_param + "=" + api_key
    if sg_cfg.wire_shape == "SpeechOpenAI":
        return url, _build_openai_speech_body(request)
    if sg_cfg.wire_shape == "SpeechGemini":
        return url, _build_gemini_speech_body(request)
    if sg_cfg.wire_shape == "SpeechMiniMax":
        return url, _build_minimax_speech_body(request)
    if sg_cfg.wire_shape == "SpeechXAI":
        return url, _build_xai_speech_body(request)
    return url, _build_inworld_speech_body(request)


# _build_minimax_speech_body assembles the MiniMax /v1/t2a_v2 request body
# (HANDOFF-094): only the fields the reference marks required. stream,
# output_format (hex) and audio_setting (mp3) take their defaults.
def _build_minimax_speech_body(request: SpeechRequest) -> dict[str, Any]:
    return {
        "model": request.model,
        "text": request.text,
        "voice_setting": {"voice_id": request.voice},
    }


# _build_xai_speech_body assembles the xAI /v1/tts request body (HANDOFF-095).
# The endpoint has no model field. language is always "auto" (llmkit has no
# language option), and with_timestamps is never sent: it switches the reply
# from raw audio bytes to JSON.
def _build_xai_speech_body(request: SpeechRequest) -> dict[str, Any]:
    return {
        "text": request.text,
        "voice_id": request.voice,
        "language": "auto",
        "output_format": {"codec": "mp3"},
    }


# _build_gemini_speech_body assembles the Gemini TTS request body on the
# Interactions API (BUG-091): one user_input turn and an audio response.
# Single voice: one text item and speech_config [{voice}]; a style annotates
# the text item. Multi-speaker: one text item per line, each annotated with its
# speaker and optional style, and speech_config {speakers}. store is always
# False: the Interactions API keeps requests by default.
def _build_gemini_speech_body(request: SpeechRequest) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    speech_config: Any
    if request.speakers:
        for line in request.lines:
            meta: dict[str, Any] = {"type": "speech_metadata", "speaker": line.speaker}
            if line.style:
                meta["style"] = line.style
            content.append({"type": "text", "text": line.text, "annotations": [meta]})
        speech_config = {
            "speakers": [{"speaker": s.name, "voice": s.voice} for s in request.speakers]
        }
    else:
        item: dict[str, Any] = {"type": "text", "text": request.text}
        if request.style:
            item["annotations"] = [{"type": "speech_metadata", "style": request.style}]
        content.append(item)
        speech_config = [{"voice": request.voice}]
    return {
        "model": request.model,
        "input": [{"type": "user_input", "content": content}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": speech_config},
        "store": False,
    }


# _build_openai_speech_body assembles the OpenAI /v1/audio/speech request body.
# Slice 1 fixes response_format=mp3 (KISS); format selection is a later slice.
# A style is sent as instructions (BUG-091), which gpt-4o-mini-tts accepts.
def _build_openai_speech_body(request: SpeechRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model,
        "input": request.text,
        "voice": request.voice,
        "response_format": "mp3",
    }
    if request.style:
        body["instructions"] = request.style
    return body


# _build_inworld_speech_body assembles the Inworld /tts/v1/voice request body.
# Slice 1 sends a fixed audioConfig (LINEAR16/22050 -> WAV) and BALANCED
# delivery; format/sample-rate selection is a later slice (ADR-049 OQ-5).
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


# _parse_speech_response decodes the synthesized audio per the wire shape's
# audio response encoding (ADR-051 OAA-002). "rawBody" (OpenAI, xAI) takes
# the response body verbatim as the audio bytes; "base64Envelope" (Inworld) parses
# a JSON envelope and base64-decodes the audio field; "hexEnvelope" (MiniMax)
# hex-decodes data.audio and raises on a nonzero base_resp.
def _parse_speech_response(
    provider_name: str, audio_encoding: str, fallback_mime: str, resp_body: bytes
) -> SpeechResponse:
    if audio_encoding == "rawBody":
        return SpeechResponse(
            audio=AudioData(mime_type=fallback_mime, bytes=resp_body), usage=Usage()
        )
    if audio_encoding == "base64InteractionSteps":
        return _parse_interaction_steps_speech(provider_name, fallback_mime, resp_body)
    if audio_encoding == "hexEnvelope":
        return _parse_hex_envelope_speech(provider_name, fallback_mime, resp_body)
    # base64Envelope: {"audioContent": "<base64>", "usage": {...}}. A 2xx body
    # that does not parse to audio is a decoding error (HANDOFF-036 A5) --
    # never a silent empty clip.
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


# _parse_interaction_steps_speech decodes a Gemini TTS Interactions API reply
# (BUG-091). Google's documented path is
# [.steps[] | select(.type=="model_output") | .content[] | select(.type=="audio")] | last | .data:
# the LAST audio item across the model_output steps wins. Its mime_type is used
# when present, else the model's outputMime. No audio item, or invalid base64,
# is a decoding error (HANDOFF-036 A5).
def _parse_interaction_steps_speech(
    provider_name: str, fallback_mime: str, resp_body: bytes
) -> SpeechResponse:
    try:
        raw = json.loads(resp_body)
    except json.JSONDecodeError as exc:
        raise APIError(
            provider=provider_name,
            status_code=200,
            message=f"speech response: not valid JSON: {exc}",
        ) from exc
    last: dict[str, Any] | None = None
    steps = raw.get("steps") if isinstance(raw, dict) else None
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        content = step.get("content")
        for item in content if isinstance(content, list) else []:
            if isinstance(item, dict) and item.get("type") == "audio":
                last = item
    data = last.get("data") if last is not None else None
    if not isinstance(data, str) or not data:
        raise APIError(
            provider=provider_name,
            status_code=200,
            message="speech response: no audio item in steps[model_output].content",
        )
    try:
        decoded = base64.b64decode(data, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise APIError(
            provider=provider_name,
            status_code=200,
            message=f"speech response: invalid base64 in audio data: {exc}",
        ) from exc
    mime = last.get("mime_type") if last is not None else None
    return SpeechResponse(
        audio=AudioData(mime_type=mime if isinstance(mime, str) and mime else fallback_mime, bytes=decoded),
        usage=Usage(),
    )


# _parse_hex_envelope_speech decodes a MiniMax t2a_v2 reply (HANDOFF-094):
# {"data": {"audio": "<hex>"}, "base_resp": {"status_code", "status_msg"}}.
# MiniMax reports failures in base_resp on an HTTP 200, so a nonzero
# status_code is an APIError carrying status_msg. A success with missing audio,
# or invalid hex, is a decoding error (HANDOFF-036 A5).
def _parse_hex_envelope_speech(
    provider_name: str, fallback_mime: str, resp_body: bytes
) -> SpeechResponse:
    try:
        raw = json.loads(resp_body)
    except json.JSONDecodeError as exc:
        raise APIError(
            provider=provider_name,
            status_code=200,
            message=f"speech response: not valid JSON: {exc}",
        ) from exc
    if not isinstance(raw, dict):
        raw = {}
    base_resp = raw.get("base_resp")
    if isinstance(base_resp, dict):
        code = base_resp.get("status_code")
        if isinstance(code, int) and code != 0:
            msg = base_resp.get("status_msg")
            raise APIError(
                provider=provider_name,
                status_code=200,
                type=str(code),
                message=msg if isinstance(msg, str) else "",
            )
    data = raw.get("data")
    audio = data.get("audio") if isinstance(data, dict) else None
    if not isinstance(audio, str) or not audio:
        raise APIError(
            provider=provider_name,
            status_code=200,
            message="speech response: missing or empty data.audio",
        )
    try:
        decoded = bytes.fromhex(audio)
    except ValueError as exc:
        raise APIError(
            provider=provider_name,
            status_code=200,
            message=f"speech response: invalid hex in data.audio: {exc}",
        ) from exc
    return SpeechResponse(
        audio=AudioData(mime_type=fallback_mime, bytes=decoded), usage=Usage()
    )
