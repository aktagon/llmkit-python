"""











"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from ..errors import APIError, ValidationError
from ..http import do_get, do_post
from ..image import Part, _image_auth_headers
from ..providers.generated.providers import PROVIDERS, ProviderName
from ..providers.generated.transcription_gen import (
    TranscriptionDef,
    transcription_config,
)
from ..structs import (
    TranscriptionHandle as _TranscriptionHandleData,
    TranscriptionResponse,
    TranscriptSegment,
)
from ..types import Provider

if TYPE_CHECKING:
    from . import Transcription


#
#
#
_DEFAULT_POLL_INTERVAL = 3.0
_DEFAULT_REQUEST_TIMEOUT = 600.0


class TranscriptionHandle(_TranscriptionHandleData):
    """



"""

    async def wait(
        self,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        request_timeout: float = _DEFAULT_REQUEST_TIMEOUT,
    ) -> TranscriptionResponse:
        return await asyncio.to_thread(
            _wait_transcription, self, poll_interval, request_timeout
        )


async def transcription_submit(
    b: "Transcription", audio_parts: list[Part]
) -> TranscriptionHandle:
    provider = Provider(
        name=b.client.provider.name,
        api_key=b.client.provider.api_key,
        headers=b.client.provider.headers,
    )
    if b.client.provider.base_url:
        provider.base_url = b.client.provider.base_url

    return await asyncio.to_thread(
        _submit_transcription, provider, list(audio_parts)
    )


def _submit_transcription(
    provider: Provider, parts: list[Part]
) -> TranscriptionHandle:
    """




"""
    cfg = PROVIDERS.get(provider.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {provider.name}")

    pname = ProviderName(provider.name)
    tc_cfg = transcription_config(pname)
    if tc_cfg is None:
        raise ValidationError(
            field="provider",
            message=f"{provider.name} does not support transcription",
        )
    #
    #
    if tc_cfg.interaction == "sync":
        raise ValidationError(
            field="interaction",
            message=f"{provider.name} transcribes synchronously; use Transcribe, not Submit/Wait",
        )

    audio_url, audio_bytes = _normalize_audio_part(parts)

    base = _transcription_base_url(provider, cfg)
    headers = _image_auth_headers(provider, cfg, pname)

    #
    #
    if audio_bytes is not None:
        if not tc_cfg.upload_endpoint:
            raise ValidationError(
                field="parts",
                message=f"{provider.name} does not accept audio bytes; pass a public audio URL",
            )
        upload_headers = {**headers, "content-type": "application/octet-stream"}
        upload_body = do_post(base + tc_cfg.upload_endpoint, audio_bytes, upload_headers)
        try:
            up = json.loads(upload_body)
        except ValueError as exc:
            raise APIError(
                message=f"unmarshal transcription upload response: {exc}",
                status_code=0,
            ) from exc
        audio_url = _lookup_handle_field(up, "upload_url")
        if not audio_url:
            raise APIError(
                message="transcription upload: response carried no upload_url",
                status_code=0,
            )

    submit_body = json.dumps({"audio_url": audio_url}).encode("utf-8")
    resp_body = do_post(
        base + tc_cfg.submit_endpoint,
        submit_body,
        {**headers, "content-type": "application/json"},
    )
    try:
        raw = json.loads(resp_body)
    except ValueError as exc:
        raise APIError(
            message=f"unmarshal transcription submit response: {exc}",
            status_code=0,
        ) from exc
    handle_id = _lookup_handle_field(raw, tc_cfg.submit_handle_field)
    if not handle_id:
        raise APIError(
            message=f"transcription submit: empty handle field {tc_cfg.submit_handle_field!r}",
            status_code=0,
        )
    return TranscriptionHandle(id=handle_id, provider=provider)


def _wait_transcription(
    handle: TranscriptionHandle,
    poll_interval: float,
    request_timeout: float,
) -> TranscriptionResponse:
    """




"""
    p = handle.provider
    cfg = PROVIDERS.get(p.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {p.name}")
    pname = ProviderName(p.name)
    tc_cfg = transcription_config(pname)
    if tc_cfg is None:
        raise ValidationError(
            field="provider",
            message=f"{p.name} does not support transcription",
        )

    base = _transcription_base_url(p, cfg)
    headers = _image_auth_headers(p, cfg, pname)
    poll_url = base + tc_cfg.poll_endpoint.replace("{id}", handle.id)

    deadline = time.monotonic() + request_timeout
    while True:
        if time.monotonic() > deadline:
            raise APIError(
                message=f"transcription poll: timed out after {request_timeout}s waiting for {handle.id}",
                status_code=0,
            )
        resp_body = do_get(poll_url, headers)
        try:
            raw = json.loads(resp_body)
        except ValueError as exc:
            raise APIError(
                message=f"unmarshal transcription poll response: {exc}",
                status_code=0,
            ) from exc
        status = _lookup_handle_field(raw, tc_cfg.status_path)
        if status == tc_cfg.done_status:
            return _transcription_result(tc_cfg, raw)
        if status == tc_cfg.error_status:
            msg = _lookup_handle_field(raw, cfg.error_message_path)
            if not msg:
                msg = "transcription failed"
            raise APIError(message=f"transcription failed: {msg}", status_code=0)
        #
        time.sleep(poll_interval)


async def transcription_transcribe(
    b: "Transcription", audio_parts: list[Part]
) -> TranscriptionResponse:
    provider = Provider(
        name=b.client.provider.name,
        api_key=b.client.provider.api_key,
        headers=b.client.provider.headers,
    )
    if b.client.provider.base_url:
        provider.base_url = b.client.provider.base_url
    return await asyncio.to_thread(
        _transcribe_sync, provider, b._model, list(audio_parts)
    )


def _transcribe_sync(
    provider: Provider, model: str, parts: list[Part]
) -> TranscriptionResponse:
    """



"""
    cfg = PROVIDERS.get(provider.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {provider.name}")
    pname = ProviderName(provider.name)
    tc_cfg = transcription_config(pname)
    if tc_cfg is None:
        raise ValidationError(
            field="provider",
            message=f"{provider.name} does not support transcription",
        )
    if tc_cfg.interaction != "sync":
        raise ValidationError(
            field="interaction",
            message=f"{provider.name} transcribes asynchronously; use Submit/Wait, not Transcribe",
        )
    if not model:
        raise ValidationError(
            field="model", message="required for synchronous transcription"
        )
    ref = _normalize_audio_bytes_part(parts)

    base = _transcription_base_url(provider, cfg)
    headers = _image_auth_headers(provider, cfg, pname)
    body, content_type = _build_openai_transcription_multipart(
        model, "verbose_json", ref
    )
    resp_body = do_post(
        base + tc_cfg.submit_endpoint,
        body,
        {**headers, "content-type": content_type},
    )
    try:
        raw = json.loads(resp_body)
    except ValueError as exc:
        raise APIError(
            message=f"unmarshal transcription response: {exc}", status_code=0
        ) from exc
    return _transcription_result_from_openai(raw)


def _build_openai_transcription_multipart(
    model: str, response_format: str, ref: Any
) -> tuple[bytes, str]:
    """



"""
    boundary = "----llmkitFormBoundaryADR051"
    crlf = "\r\n"
    mime = ref.mime_type or "application/octet-stream"
    ext = _audio_ext_for_mime(ref.mime_type)
    chunks: list[bytes] = []

    def field(name: str, value: str) -> None:
        chunks.append(
            (
                f"--{boundary}{crlf}"
                f'Content-Disposition: form-data; name="{name}"{crlf}{crlf}'
                f"{value}{crlf}"
            ).encode("utf-8")
        )

    field("model", model)
    field("response_format", response_format)
    chunks.append(
        (
            f"--{boundary}{crlf}"
            f'Content-Disposition: form-data; name="file"; filename="audio.{ext}"{crlf}'
            f"Content-Type: {mime}{crlf}{crlf}"
        ).encode("utf-8")
    )
    chunks.append(ref.bytes)
    chunks.append(crlf.encode("utf-8"))
    chunks.append(f"--{boundary}--{crlf}".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _audio_ext_for_mime(mime: str) -> str:
    """
"""
    return {
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mp4": "m4a",
        "audio/m4a": "m4a",
        "audio/x-m4a": "m4a",
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/opus": "ogg",
        "audio/flac": "flac",
    }.get(mime, "bin")


def _transcription_result_from_openai(
    raw: dict[str, Any],
) -> TranscriptionResponse:
    """



"""
    text = raw.get("text") if isinstance(raw, dict) else None
    text = text if isinstance(text, str) else ""
    segs = raw.get("segments") if isinstance(raw, dict) else None
    segments: list[TranscriptSegment] = []
    if isinstance(segs, list):
        for sd in segs:
            if not isinstance(sd, dict):
                continue
            seg = TranscriptSegment()
            st = sd.get("text")
            seg.text = st if isinstance(st, str) else ""
            start = sd.get("start")
            if isinstance(start, (int, float)) and not isinstance(start, bool):
                seg.start = int(round(start * 1000))
            end = sd.get("end")
            if isinstance(end, (int, float)) and not isinstance(end, bool):
                seg.end = int(round(end * 1000))
            segments.append(seg)
    return TranscriptionResponse(text=text, segments=segments)


def _normalize_audio_bytes_part(parts: list[Part]) -> Any:
    """

"""
    ref = None
    audio_count = 0
    for i, part in enumerate(parts):
        if part.audio is not None:
            audio_count += 1
            ref = part.audio
        elif part.audio_url:
            raise ValidationError(
                field=f"parts[{i}]",
                message="synchronous transcription accepts inline audio bytes only (audio_bytes); a remote audio URL is not supported",
            )
        elif part.text or part.image is not None or part.lyrics:
            raise ValidationError(
                field=f"parts[{i}]",
                message="transcription accepts only audio parts (audio_bytes)",
            )
        else:
            raise ValidationError(field=f"parts[{i}]", message="empty part")
    if audio_count != 1 or ref is None:
        raise ValidationError(
            field="parts",
            message="transcription requires exactly one audio part",
        )
    return ref


def _transcription_result(
    tc_cfg: TranscriptionDef, raw: dict[str, Any]
) -> TranscriptionResponse:
    """

"""
    if tc_cfg.wire_shape == "TranscriptionAssemblyAI":
        return _transcription_result_from_assemblyai(raw)
    raise APIError(
        message=f"transcription: unsupported wire shape {tc_cfg.wire_shape!r}",
        status_code=0,
    )


def _transcription_result_from_assemblyai(
    raw: dict[str, Any],
) -> TranscriptionResponse:
    """



"""
    text = raw.get("text") if isinstance(raw, dict) else None
    text = text if isinstance(text, str) else ""
    words = raw.get("words") if isinstance(raw, dict) else None
    segments: list[TranscriptSegment] = []
    if isinstance(words, list):
        for w in words:
            if not isinstance(w, dict):
                continue
            seg = TranscriptSegment()
            wt = w.get("text")
            seg.text = wt if isinstance(wt, str) else ""
            start = w.get("start")
            if isinstance(start, (int, float)) and not isinstance(start, bool):
                seg.start = int(start)
            end = w.get("end")
            if isinstance(end, (int, float)) and not isinstance(end, bool):
                seg.end = int(end)
            speaker = w.get("speaker")
            seg.speaker = speaker if isinstance(speaker, str) else ""
            segments.append(seg)
    return TranscriptionResponse(text=text, segments=segments)


def _normalize_audio_part(parts: list[Part]) -> tuple[str, bytes | None]:
    """


"""
    url = ""
    raw: bytes | None = None
    audio_count = 0
    for i, part in enumerate(parts):
        if part.audio_url:
            audio_count += 1
            url = part.audio_url
        elif part.audio is not None:
            audio_count += 1
            raw = part.audio.bytes
        elif part.text or part.image is not None or part.lyrics:
            raise ValidationError(
                field=f"parts[{i}]",
                message="transcription accepts only audio parts (audio / audio_bytes)",
            )
        else:
            raise ValidationError(field=f"parts[{i}]", message="empty part")
    if audio_count != 1:
        raise ValidationError(
            field="parts",
            message="transcription requires exactly one audio part",
        )
    return url, raw


def _transcription_base_url(provider: Provider, cfg: Any) -> str:
    """


"""
    if provider.base_url:
        return provider.base_url
    return cfg.base_url


def _lookup_handle_field(raw: Any, path: str) -> str:
    """
"""
    if not path:
        return ""
    cur: Any = raw
    for seg in path.split("."):
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(seg)
    if isinstance(cur, str):
        return cur
    if isinstance(cur, int) and not isinstance(cur, bool):
        return str(cur)
    if isinstance(cur, float):
        return str(int(cur))
    return ""
