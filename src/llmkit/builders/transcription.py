"""Transcription (speech-to-text) runtime (ADR-048) — mirror of
go/transcription.go and go/transcription_builder.go.

Transcription is asynchronous: ``transcription_submit`` POSTs the job (with an
upload hop first for local-bytes audio) and returns a ``TranscriptionHandle``
immediately; the caller polls the handle with ``await handle.wait()`` (modeled
on the video handle, ADR-034 / ADR-014).

Pre-flight validation (exactly one audio part; non-audio parts rejected) runs
before any HTTP call. The submit/poll/status facts are config; only the result
decode is wire-shape-keyed (STT-005). Slice 1 wires TranscriptionAssemblyAI:
upload -> submit -> poll -> {text, words[]}.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from ..errors import APIError, ValidationError
from ..http import do_get, do_post
from ..image import Part, _image_auth_headers
from ..job import (
    JobStatus,
    LifecycleConfig,
    PollBody,
    classify_by_config,
    poll_engine_once,
    poll_job_async,
    _Classification,
)
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


# Default poll cadence for TranscriptionHandle.wait. AssemblyAI jobs run from
# seconds to minutes; the SDK polls every poll_interval until request_timeout
# elapses. Mirror of go/transcription.go transcriptionPollInterval / Timeout.
_DEFAULT_POLL_INTERVAL = 3.0
_DEFAULT_REQUEST_TIMEOUT = 600.0
# The OVERALL poll-loop wall-clock backstop (seconds) — distinct from the
# per-HTTP-request _DEFAULT_REQUEST_TIMEOUT (S05). Mirror of go
# transcriptionPollTimeout (10 min).
_DEFAULT_POLL_DEADLINE = 600.0


class TranscriptionHandle(_TranscriptionHandleData):
    """Typed-builder TranscriptionHandle. Inherits the ontology-generated data
    shape (id, provider) and adds a ``wait()`` method so callers can write
    ``handle = await transcription.submit(...); resp = await handle.wait()`` —
    mirroring Go's ``TranscriptionHandle.Wait`` value-receiver shape (ADR-048).
    """

    async def wait(
        self,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        request_timeout: float = _DEFAULT_REQUEST_TIMEOUT,
        poll_deadline: float = _DEFAULT_POLL_DEADLINE,
    ) -> TranscriptionResponse:
        """Poll until the transcription job reaches a terminal state, then return
        the finished response. A thin loop over ``poll`` (ADR-063 POLL-003) via the
        shared engine; the between-poll wait is a cancellable ``asyncio.sleep`` so
        ``asyncio.CancelledError`` propagates (S06). ``poll_deadline`` is the NEW
        overall wall-clock backstop, distinct from the per-request
        ``request_timeout`` (S05)."""
        adapter = _new_transcription_adapter(
            self, poll_interval, request_timeout, poll_deadline
        )
        return await poll_job_async(adapter)

    async def poll(
        self,
        *,
        request_timeout: float = _DEFAULT_REQUEST_TIMEOUT,
        poll_deadline: float = _DEFAULT_POLL_DEADLINE,
    ) -> JobStatus[TranscriptionResponse]:
        """Perform exactly ONE provider round-trip and return the normalized
        JobStatus (ADR-063 POLL-001). On a completed job JobStatus.result carries
        the finished TranscriptionResponse; a failed job populates JobStatus.cause
        (the provider error rides on cause.message, preserving the wait surface)."""
        adapter = _new_transcription_adapter(
            self, _DEFAULT_POLL_INTERVAL, request_timeout, poll_deadline
        )
        return await poll_engine_once(adapter)


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
    """Submit an asynchronous speech-to-text job and return a
    TranscriptionHandle immediately. Pre-flight validation rejects an input that
    is not exactly one audio Part before any HTTP call (STT-003). For an
    audio-bytes part the runtime performs the upload hop (POST the raw bytes,
    read upload_url) before submitting (STT-005). Mirror of go
    submitTranscription."""
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
    # A synchronous provider has no job handle; Submit/Wait is the wrong terminal
    # for it (ADR-051 OAA-003). Name the supported one.
    if tc_cfg.interaction == "sync":
        raise ValidationError(
            field="interaction",
            message=f"{provider.name} transcribes synchronously; use Transcribe, not Submit/Wait",
        )

    audio_url, audio_bytes = _normalize_audio_part(parts)

    base = _transcription_base_url(provider, cfg)
    headers = _image_auth_headers(provider, cfg, pname)

    # Upload hop (STT-005): a bytes part is uploaded first to obtain a URL the
    # submit body can reference. URL parts skip this entirely.
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


class _TranscriptionAdapter:
    """Binds async transcription to the job engine's four seams. classify uses the
    config-backed default (status vs done/error values); result decodes the
    finished transcript per wire shape (no second hop). Mirror of go
    transcriptionAdapter."""

    def __init__(
        self,
        lc: LifecycleConfig,
        headers: dict[str, str],
        poll_url: str,
        tc_cfg: TranscriptionDef,
        request_timeout: float,
    ) -> None:
        self._lc = lc
        self._headers = headers
        self._poll_url = poll_url
        self._tc_cfg = tc_cfg
        self._request_timeout = request_timeout

    def config(self) -> LifecycleConfig:
        return self._lc

    def poll(self) -> PollBody:
        resp_body = do_get(self._poll_url, self._headers, timeout=self._request_timeout)
        try:
            raw = json.loads(resp_body)
        except ValueError as exc:
            raise APIError(
                message=f"unmarshal transcription poll response: {exc}",
                status_code=0,
            ) from exc
        return PollBody(raw=raw)

    def classify(self, body: PollBody) -> _Classification:
        return classify_by_config(self._lc, body)

    def result(self, body: PollBody) -> TranscriptionResponse:
        return _transcription_result(self._tc_cfg, body.raw)


def _new_transcription_adapter(
    handle: TranscriptionHandle,
    poll_interval: float,
    request_timeout: float,
    poll_deadline: float,
) -> _TranscriptionAdapter:
    """Assemble the transcription adapter + its LifecycleConfig. The
    status-to-terminal mapping stays config (status_path / done_status /
    error_status, STT-005); the provider error message rides on
    cfg.error_message_path so wait still surfaces it (S02). The handle carries the
    transcript id and provider config, so wait works across process boundaries.
    Mirror of go newTranscriptionAdapter."""
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

    lc = LifecycleConfig(
        noun="transcription",
        status_path=tc_cfg.status_path,
        done_values=tuple(v for v in (tc_cfg.done_status,) if v),
        error_values=tuple(v for v in (tc_cfg.error_status,) if v),
        error_message_path=cfg.error_message_path,
        poll_interval=poll_interval,
        poll_timeout=poll_deadline,
    )
    return _TranscriptionAdapter(lc, headers, poll_url, tc_cfg, request_timeout)


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
    """Run a SYNCHRONOUS speech-to-text request (ADR-051): one multipart/form-data
    POST returns the transcript directly, no job handle. Pre-flight rejects a
    non-sync provider (naming Submit/Wait), a missing model, a remote audio URL
    (OpenAI ingests inline bytes only — the inverse of AssemblyAI, OAA-005), and
    a non-single-audio-bytes input. Mirror of go transcribeSync."""
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
    """Encode the OpenAI /v1/audio/transcriptions body as multipart/form-data in
    FIXED field order (model, response_format, file) so all four SDKs emit the
    same canonical descriptor. The file part carries its IANA Content-Type and a
    filename whose extension reflects the format. Mirror of go
    buildOpenAITranscriptionMultipart."""
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
    """Map an audio IANA media type to the file extension OpenAI uses to detect
    the format. Mirror of go audioExtForMime."""
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
    """Extract the transcript text and (when present) segment timings from a
    synchronous OpenAI response. verbose_json offsets are SECONDS (float) ->
    integer milliseconds (x1000, rounded, OAA-006). Models without segments[]
    -> empty segments, not an error. Usage stays zero (OAA-007). Mirror of go
    transcriptionResultFromOpenAI."""
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
    """Enforce the single-audio-part rule for the sync path (OAA-005): exactly
    one inline-bytes audio Part. A remote URL is rejected (OpenAI ingests no URL
    — the inverse of AssemblyAI). Mirror of go normalizeAudioBytesPart."""
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
    """Extract the finished transcript per wire shape. Only the result decode is
    wire-shape-keyed (STT-005); the submit/poll/status facts are config. Mirror
    of go transcriptionResult."""
    if tc_cfg.wire_shape == "TranscriptionAssemblyAI":
        return _transcription_result_from_assemblyai(raw)
    raise APIError(
        message=f"transcription: unsupported wire shape {tc_cfg.wire_shape!r}",
        status_code=0,
    )


def _transcription_result_from_assemblyai(
    raw: dict[str, Any],
) -> TranscriptionResponse:
    """Extract the transcript text and word-level timing segments from a
    completed AssemblyAI transcript object. start/end are integer milliseconds;
    speaker is present only on diarized transcripts. Usage stays zero —
    AssemblyAI bills by audio duration, not tokens (ADR-048 OQ-2). Mirror of go
    transcriptionResultFromAssemblyAI."""
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
    """Enforce the single-audio-part rule (STT-003) and return the audio source:
    a URL XOR raw bytes. A request with a non-audio part, or with anything other
    than exactly one audio part, is rejected pre-flight. Mirror of go
    normalizeAudioPart."""
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
    """Resolve the base for the transcription API: an explicit per-client
    override wins (tests point it at a mock; users at a proxy), else the
    provider's chat base. Submit/poll/upload endpoints are always relative paths
    joined to this base. Mirror of go transcriptionBaseURL."""
    if provider.base_url:
        return provider.base_url
    return cfg.base_url


def _lookup_handle_field(raw: Any, path: str) -> str:
    """Descend a dotted path (e.g. "id", "status", "error") through the decoded
    response, returning the leaf string or "" if any segment is missing."""
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
