"""










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


#
#
#
_DEFAULT_POLL_INTERVAL = 5.0
_DEFAULT_REQUEST_TIMEOUT = 600.0


@dataclasses.dataclass
class VideoRequest:
    """











"""

    model: str = ""
    prompt: str = ""
    parts: list[Part] = dataclasses.field(default_factory=list)


class VideoHandle(_VideoHandleData):
    """





"""

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

    #
    #
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
    """

"""
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
    """








"""
    #
    #
    #
    body = {"model": model, "prompt": _join_prompt_text(parts)}
    json_body = json.dumps(body).encode("utf-8")
    resp_body = do_post(
        _resolve_video_endpoint(base_url, vg_cfg.gen_endpoint),
        json_body,
        {**headers, "content-type": "application/json"},
    )
    try:
        raw = json.loads(resp_body)
    except ValueError as exc:
        raise APIError(
            message=f"unmarshal video submit response: {exc}", status_code=0
        ) from exc
    handle_id = _lookup_handle_field(raw, vg_cfg.submit_handle_field)
    if not handle_id:
        raise APIError(
            message=f"video submit: empty handle field {vg_cfg.submit_handle_field!r}",
            status_code=0,
        )
    return handle_id


def _wait_video(
    handle: VideoHandle,
    poll_interval: float,
    request_timeout: float,
    raw: bool,
) -> VideoResponse:
    """



"""
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
    poll_url = _video_poll_url(vg_cfg.poll_endpoint, base, handle.id)

    #
    #
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


def _video_poll_url(poll_endpoint: str, base: str, id: str) -> str:
    """

"""
    return _resolve_video_endpoint(base, poll_endpoint.replace("{id}", id))


def _resolve_video_endpoint(base: str, endpoint: str) -> str:
    """"""
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    return base + endpoint


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
    return cur if isinstance(cur, str) else ""


def _parse_video_poll(vg_cfg: VideoGenDef, body: bytes) -> tuple[VideoResponse, bool]:
    """










"""
    try:
        raw = json.loads(body)
    except ValueError as exc:
        raise APIError(
            message=f"unmarshal video poll response: {exc}", status_code=0
        ) from exc

    if vg_cfg.wire_shape == "VideoZhipu":
        status = raw.get("task_status") if isinstance(raw, dict) else None
        if status == "SUCCESS":
            return _video_result_from_zhipu(vg_cfg, raw), True
        if status == "FAIL":
            raise APIError(message="video generation failed", status_code=0)
        #
        return VideoResponse(), False

    if vg_cfg.wire_shape == "VideoGrok":
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
        #
        return VideoResponse(), False

    #
    #
    raise APIError(
        message=f"video poll: unsupported wire shape {vg_cfg.wire_shape!r}",
        status_code=0,
    )


def _video_result_from_grok(vg_cfg: VideoGenDef, raw: dict[str, Any]) -> VideoResponse:
    """

"""
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


def _video_result_from_zhipu(vg_cfg: VideoGenDef, raw: dict[str, Any]) -> VideoResponse:
    """


"""
    mime = _video_fallback_mime(vg_cfg)
    results = raw.get("video_result") if isinstance(raw, dict) else None
    if not isinstance(results, list) or not results:
        return VideoResponse()
    first = results[0]
    if not isinstance(first, dict):
        return VideoResponse()
    url = first.get("url")
    return VideoResponse(
        videos=[VideoData(mime_type=mime, url=url if isinstance(url, str) else "")]
    )


def _video_fallback_mime(vg_cfg: VideoGenDef) -> str:
    """
"""
    if vg_cfg.models:
        return vg_cfg.models[0].output_mime
    return "video/mp4"


def _normalize_video_parts(request: VideoRequest) -> list[Part]:
    """

"""
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
