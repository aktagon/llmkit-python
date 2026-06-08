"""Video generation runtime (ADR-034) — mirror of go/video.go and
go/video_builder.go.

Video generation is asynchronous: ``video_submit`` POSTs the job and
returns a ``VideoHandle`` immediately; the caller polls the handle with
``await handle.wait()`` (modeled on the batch handle, ADR-014).

Pre-flight validation (model required; XOR prompt/parts; lyrics and image
parts rejected) runs before any HTTP call. Dispatch branches on
vg_cfg.wire_shape — the single discriminator, never the provider name.
Slice 1 wires VideoGrok (xAI) only: {model, prompt} submit, url delivery.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from typing import TYPE_CHECKING, Any

from ..errors import APIError, ValidationError
from ..http import do_get, do_post
from ..image import Part, _image_auth_headers
from ..middleware import fire_post, fire_pre
from ..providers.generated.middleware import Event, MiddlewareFn, MiddlewareOp
from ..providers.generated.providers import PROVIDERS, ProviderName
from ..providers.generated.video_gen import (
    VideoGenDef,
    VideoModelDef,
    video_gen_config,
)
from ..structs import VideoData, VideoHandle as _VideoHandleData, VideoResponse
from ..types import Provider

if TYPE_CHECKING:
    from . import Video


# Default poll cadence for VideoHandle.wait. xAI documents up-to-several-minute
# generations; the SDK polls every poll_interval until request_timeout elapses
# (ADR-034 D2 — mirrors BatchHandle.wait's per-call kwargs, ADR-014).
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_REQUEST_TIMEOUT = 600.0


@dataclasses.dataclass
class VideoRequest:
    """Canonical video-generation request (ADR-034).

    Model is required: video-generation models are explicit choices and the
    text-generation default does not generate video.

    Input is provided in one of two mutually-exclusive forms:
      - prompt: terse sugar for the prompt-only hot path. Internally
        desugars to parts=[Part(text=prompt)] before serialisation.
      - parts: canonical sequence of text parts (slice 1 is text-to-video).

    Pre-flight validation requires exactly one of prompt or parts to be
    non-empty (XOR).
    """

    model: str = ""
    prompt: str = ""
    parts: list[Part] = dataclasses.field(default_factory=list)


class VideoHandle(_VideoHandleData):
    """Typed-builder VideoHandle. Inherits the ontology-generated data shape
    (id, provider, raw) and adds a ``wait()`` method so callers can write
    ``handle = await video.submit(...); resp = await handle.wait()`` —
    mirroring Go's ``VideoHandle.Wait`` value-receiver shape (ADR-014).

    Poll cadence is per-call (poll_interval / request_timeout), mirroring
    ``BatchHandle.wait`` so the video handle matches its batch twin."""

    async def wait(
        self,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        request_timeout: float = _DEFAULT_REQUEST_TIMEOUT,
        raw: bool = False,
    ) -> VideoResponse:
        return await asyncio.to_thread(
            _wait_video, self, poll_interval, request_timeout, raw
        )


async def video_submit(b: "Video", msg: str) -> VideoHandle:
    provider = Provider(
        name=b.client.provider.name,
        api_key=b.client.provider.api_key,
    )
    if b.client.provider.base_url:
        provider.base_url = b.client.provider.base_url

    # Mirror go/video_builder.go: chain-accumulated parts plus an optional
    # trailing text part from submit(msg).
    request = VideoRequest(model=b._model)
    if b._parts:
        if msg:
            request.parts = [*b._parts, Part(text=msg)]
        else:
            request.parts = list(b._parts)
    elif msg:
        request.prompt = msg

    return await asyncio.to_thread(
        _submit_video,
        provider,
        request,
        list(b._middleware),
        b._raw,
    )


def _submit_video(
    provider: Provider,
    request: VideoRequest,
    middleware: list[MiddlewareFn],
    raw: bool,
) -> VideoHandle:
    """Submit an asynchronous text-to-video job and return a VideoHandle
    immediately. Pre-flight validation rejects unknown models and
    unsupported part kinds before any HTTP call. Mirror of go submitVideo."""
    if not provider.api_key:
        raise ValidationError(field="api_key", message="required")
    if not request.model:
        raise ValidationError(field="model", message="required for video generation")

    parts = _normalize_video_parts(request)
    for i, part in enumerate(parts):
        if part.lyrics:
            raise ValidationError(
                field=f"parts[{i}]",
                message="video generation does not accept lyrics parts",
            )
        if part.image is not None:
            raise ValidationError(
                field=f"parts[{i}]",
                message="image-to-video is not yet wired (slice 1 is text-to-video)",
            )
        if not part.text:
            raise ValidationError(field=f"parts[{i}]", message="must have Text set")

    cfg = PROVIDERS.get(provider.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {provider.name}")

    pname = ProviderName(provider.name)
    vg_cfg = video_gen_config(pname)
    if vg_cfg is None:
        raise ValidationError(
            field="provider",
            message=f"{provider.name} does not support video generation",
        )
    if _find_video_model(vg_cfg, request.model) is None:
        raise ValidationError(
            field="model",
            message=f"{request.model} is not a known video-generation model for {provider.name}",
        )

    mws = list(middleware)
    base_event = Event(
        op=MiddlewareOp.VIDEO_GENERATION,
        provider=provider.name,
        model=request.model,
    )
    start = time.monotonic()
    fire_pre(mws, base_event)

    try:
        headers = _image_auth_headers(provider, cfg, pname)
        base_url = provider.base_url or cfg.base_url
        request_id = _dispatch_video_submit(
            cfg, vg_cfg, request.model, parts, base_url, headers
        )
    except Exception as exc:
        fire_post(
            mws,
            dataclasses.replace(
                base_event, err=str(exc), duration=time.monotonic() - start
            ),
        )
        raise

    fire_post(
        mws,
        dataclasses.replace(base_event, duration=time.monotonic() - start),
    )
    return VideoHandle(id=request_id, provider=provider, raw=raw)


def _dispatch_video_submit(
    cfg: Any,
    vg_cfg: VideoGenDef,
    model: str,
    parts: list[Part],
    base_url: str,
    headers: dict[str, str],
) -> str:
    """POST the submit body per wire shape (never by provider name) and
    return the provider-assigned request id.

      - VideoGrok (xAI): POST {model, prompt} to gen_endpoint; the response
        is {"request_id": "..."}.
    """
    # Only VideoGrok is wired (slice 1). The default arm is the Grok shape.
    body = {"model": model, "prompt": _join_prompt_text(parts)}
    json_body = json.dumps(body).encode("utf-8")
    resp_body = do_post(
        base_url + vg_cfg.gen_endpoint,
        json_body,
        {**headers, "content-type": "application/json"},
    )
    try:
        raw = json.loads(resp_body)
    except ValueError as exc:
        raise APIError(
            message=f"unmarshal video submit response: {exc}", status_code=0
        ) from exc
    request_id = raw.get("request_id") if isinstance(raw, dict) else None
    if not isinstance(request_id, str) or not request_id:
        raise APIError(message="video submit: empty request_id", status_code=0)
    return request_id


def _wait_video(
    handle: VideoHandle,
    poll_interval: float,
    request_timeout: float,
    raw: bool,
) -> VideoResponse:
    """Poll the provider until the video job reaches a terminal state, then
    return the finished VideoResponse. A failed or expired job raises. Poll
    cadence uses poll_interval until request_timeout elapses. The handle
    carries the request id and provider config, so wait works across process
    boundaries. Mirror of go VideoHandle.Wait."""
    p = handle.provider
    cfg = PROVIDERS.get(p.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {p.name}")
    pname = ProviderName(p.name)
    vg_cfg = video_gen_config(pname)
    if vg_cfg is None:
        raise ValidationError(
            field="provider",
            message=f"{p.name} does not support video generation",
        )

    base = p.base_url or cfg.base_url
    headers = _image_auth_headers(p, cfg, pname)
    poll_url = _video_poll_url(base, handle.id)

    # ADR-014 cross-process resume: a handle that remembers raw takes effect
    # at wait time even if the raw kwarg was not passed.
    raw = raw or handle.raw

    deadline = time.monotonic() + request_timeout
    while True:
        if time.monotonic() > deadline:
            raise APIError(
                message=f"video poll: timed out after {request_timeout}s waiting for {handle.id}",
                status_code=0,
            )
        resp_body = do_get(poll_url, headers)
        resp, done = _parse_video_poll(vg_cfg, resp_body)
        if done:
            if raw:
                try:
                    resp.raw = json.loads(resp_body)
                except ValueError:
                    resp.raw = None
            return resp
        time.sleep(poll_interval)


def _video_poll_url(base: str, id: str) -> str:
    """Build the per-wire-shape poll URL. VideoGrok: GET {base}/v1/videos/{id}."""
    return base + "/v1/videos/" + id


def _parse_video_poll(vg_cfg: VideoGenDef, body: bytes) -> tuple[VideoResponse, bool]:
    """Decode one poll response. Returns (resp, done):

      - done=False when the job is still pending (caller keeps polling).
      - done=True with the finished VideoResponse when status is
        terminal-success.
      - raises when the job failed or expired.

    VideoGrok: {"status": "...", "video": {"url", "duration"}} or
    {"status": "failed", "error": {"code", "message"}}.
    """
    try:
        raw = json.loads(body)
    except ValueError as exc:
        raise APIError(
            message=f"unmarshal video poll response: {exc}", status_code=0
        ) from exc

    status = raw.get("status") if isinstance(raw, dict) else None
    if status == "done":
        return _video_result_from_grok(vg_cfg, raw), True
    if status in ("failed", "expired"):
        msg = status
        err_obj = raw.get("error") if isinstance(raw, dict) else None
        if isinstance(err_obj, dict):
            m = err_obj.get("message")
            if isinstance(m, str) and m:
                msg = m
        raise APIError(message=f"video generation {status}: {msg}", status_code=0)
    # pending (or any non-terminal status)
    return VideoResponse(), False


def _video_result_from_grok(vg_cfg: VideoGenDef, raw: dict[str, Any]) -> VideoResponse:
    """Extract the finished video from a Grok poll response. Grok uses url
    delivery: VideoData.url carries a temporary xAI-hosted URL and bytes
    stays empty (the SDK does not download on the caller's behalf)."""
    mime = _video_fallback_mime(vg_cfg)
    video = raw.get("video") if isinstance(raw, dict) else None
    if not isinstance(video, dict):
        return VideoResponse()
    url = video.get("url")
    data = VideoData(mime_type=mime, url=url if isinstance(url, str) else "")
    duration = video.get("duration")
    if isinstance(duration, (int, float)):
        data.duration_seconds = int(duration)
    return VideoResponse(videos=[data])


def _video_fallback_mime(vg_cfg: VideoGenDef) -> str:
    """Return the first model's output MIME, used when the provider does not
    echo a MIME on the result."""
    if vg_cfg.models:
        return vg_cfg.models[0].output_mime
    return "video/mp4"


def _normalize_video_parts(request: VideoRequest) -> list[Part]:
    """Enforce the XOR rule and produce the canonical list[Part]. When only
    prompt is set, synthesise [Part(text=prompt)]. Both empty or both set
    raises ValidationError."""
    has_prompt = bool(request.prompt)
    has_parts = bool(request.parts)
    if has_prompt and has_parts:
        raise ValidationError(field="parts", message="set Prompt or Parts, not both")
    if not has_prompt and not has_parts:
        raise ValidationError(field="prompt", message="set either Prompt or Parts")
    return [Part(text=request.prompt)] if has_prompt else list(request.parts)


def _find_video_model(cfg: VideoGenDef, model_id: str) -> VideoModelDef | None:
    for m in cfg.models:
        if m.model_id == model_id:
            return m
    return None


def _join_prompt_text(parts: list[Part]) -> str:
    return "\n".join(p.text for p in parts if p.text)
