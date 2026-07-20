"""










"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import os
import time
import uuid
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from ..errors import APIError, ValidationError
from ..http import do_get, do_post, do_sigv4_get, do_sigv4_post
from ..image import Part, _image_auth_headers
from ..middleware import fire_post, fire_pre, set_event_error
from ..providers.generated.middleware import Event, MiddlewareFn, MiddlewareOp
from ..providers.generated.providers import PROVIDERS, ProviderName
from ..providers.generated.request import AuthScheme, auth_scheme
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
    #
    #
    #
    #
    output_uri: str = ""


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
        headers=b.client.provider.headers,
    )
    if b.client.provider.base_url:
        provider.base_url = b.client.provider.base_url

    #
    #
    request = VideoRequest(model=b._model, output_uri=b._output_uri)
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
    model = _find_video_model(vg_cfg, request.model)
    if model is None:
        raise ValidationError(
            field="model",
            message=f"{request.model} is not a known video-generation model for {provider.name}",
        )

    for i, part in enumerate(parts):
        if part.lyrics:
            raise ValidationError(
                field=f"parts[{i}]",
                message="video generation does not accept lyrics parts",
            )
        if part.image is not None:
            #
            #
            #
            if not model.supports_image_to_video:
                raise ValidationError(
                    field=f"parts[{i}]",
                    message=f"{request.model} is a text-to-video-only model and does not accept image parts",
                )
            continue
        if not part.text:
            raise ValidationError(field=f"parts[{i}]", message="must have Text set")
    #
    #
    #
    if vg_cfg.requires_output_uri and not request.output_uri:
        raise ValidationError(
            field="output_uri",
            message=f"{provider.name} requires a caller output S3 URI; set output_uri on the request",
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
        base_url = _video_base_url(provider, cfg, vg_cfg)
        request_id = _dispatch_video_submit(
            provider,
            pname,
            cfg,
            vg_cfg,
            request.model,
            request.output_uri,
            parts,
            base_url,
            headers,
        )
    except Exception as exc:
        post_event = dataclasses.replace(
            base_event, duration=time.monotonic() - start
        )
        set_event_error(post_event, exc)
        fire_post(mws, post_event)
        raise

    fire_post(
        mws,
        dataclasses.replace(base_event, duration=time.monotonic() - start),
    )
    return VideoHandle(id=request_id, provider=provider, raw=raw, model=request.model)


def _dispatch_video_submit(
    provider: Provider,
    pname: ProviderName,
    cfg: Any,
    vg_cfg: VideoGenDef,
    model: str,
    output_uri: str,
    parts: list[Part],
    base_url: str,
    headers: dict[str, str],
) -> str:
    """

















"""
    #
    #
    post_headers = headers
    if vg_cfg.wire_shape == "VideoQwen":
        body: dict[str, Any] = {
            "model": model,
            "input": {"prompt": _join_prompt_text(parts)},
        }
        #
        #
        post_headers = {**headers, "X-DashScope-Async": "enable"}
    elif vg_cfg.wire_shape == "VideoPixVerse":
        #
        #
        #
        #
        #
        body = {
            "model": model,
            "prompt": _join_prompt_text(parts),
            "duration": 5,
            "quality": "540p",
            "aspect_ratio": "16:9",
        }
        post_headers = {**headers, "Ai-trace-id": _new_video_trace_id()}
    elif vg_cfg.wire_shape in ("VideoVeo", "VideoVertexVeo"):
        #
        #
        #
        #
        #
        body = {"instances": [{"prompt": _join_prompt_text(parts)}]}
    elif vg_cfg.wire_shape == "VideoBedrock":
        #
        #
        #
        #
        body = {
            "modelId": model,
            "modelInput": {
                "taskType": "TEXT_VIDEO",
                "textToVideoParams": {"text": _join_prompt_text(parts)},
            },
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": output_uri}},
        }
    else:
        body = {"model": model, "prompt": _join_prompt_text(parts)}
        #
        #
        #
        #
        #
        seed = _video_seed_image_url(parts)
        if seed:
            body["image"] = {"url": seed}
    json_body = json.dumps(body).encode("utf-8")
    #
    #
    #
    submit_url = _append_video_auth(
        base_url + vg_cfg.gen_endpoint.replace("{model}", model), provider, pname, cfg
    )
    if auth_scheme(pname) == AuthScheme.SIG_V4:
        #
        #
        region = os.environ.get(cfg.region_env_var, "")
        secret_key = os.environ.get(cfg.secret_key_env_var, "")
        session_token = os.environ.get(cfg.session_token_env_var, "")
        resp_body = do_sigv4_post(
            submit_url,
            json_body,
            provider.api_key,
            secret_key,
            session_token,
            region,
            cfg.service_name,
        )
    else:
        resp_body = do_post(
            submit_url,
            json_body,
            {**post_headers, "content-type": "application/json"},
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

    base = _video_base_url(p, cfg, vg_cfg)
    headers = _image_auth_headers(p, cfg, pname)
    #
    #
    #
    #
    if vg_cfg.wire_shape == "VideoPixVerse":
        headers = {**headers, "Ai-trace-id": _new_video_trace_id()}

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
    #
    #
    #
    #
    #
    #
    #
    #
    #
    sig_v4 = auth_scheme(pname) == AuthScheme.SIG_V4
    vertex_poll = vg_cfg.wire_shape == "VideoVertexVeo"
    region = secret_key = session_token = ""
    vertex_poll_body = b""
    if sig_v4:
        poll_url = base + vg_cfg.poll_endpoint.replace("{id}", quote(handle.id, safe=":"))
        region = os.environ.get(cfg.region_env_var, "")
        secret_key = os.environ.get(cfg.secret_key_env_var, "")
        session_token = os.environ.get(cfg.session_token_env_var, "")
    elif vertex_poll:
        poll_url = _append_video_auth(
            base + vg_cfg.poll_endpoint.replace("{model}", handle.model), p, pname, cfg
        )
        vertex_poll_body = json.dumps({"operationName": handle.id}).encode("utf-8")
    else:
        poll_url = _append_video_auth(
            _video_poll_url(vg_cfg.poll_endpoint, base, handle.id), p, pname, cfg
        )

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
        if sig_v4:
            resp_body = do_sigv4_get(
                poll_url,
                p.api_key,
                secret_key,
                session_token,
                region,
                cfg.service_name,
            )
        elif vertex_poll:
            resp_body = do_post(
                poll_url,
                vertex_poll_body,
                {**headers, "content-type": "application/json"},
            )
        else:
            resp_body = do_get(poll_url, headers)
        resp, done = _parse_video_poll(vg_cfg, resp_body)
        if done:
            #
            #
            #
            if vg_cfg.file_endpoint:
                resp = _resolve_video_file(base, vg_cfg, resp_body, headers)
            #
            #
            #
            #
            if vg_cfg.output_delivery == "DeliveryDownload":
                resp = _download_video_bytes(p, pname, cfg, resp, headers)
            if raw:
                try:
                    resp.raw = json.loads(resp_body)
                except ValueError:
                    resp.raw = None
            return resp
        time.sleep(poll_interval)


def _video_base_url(provider: Provider, cfg: Any, vg_cfg: VideoGenDef) -> str:
    """



"""
    if provider.base_url:
        return provider.base_url
    base = vg_cfg.video_base_url or cfg.base_url
    #
    #
    #
    if cfg.region_env_var:
        base = base.replace("{region}", os.environ.get(cfg.region_env_var, ""))
    return base


def _video_poll_url(poll_endpoint: str, base: str, id: str) -> str:
    """
"""
    return base + poll_endpoint.replace("{id}", id)


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
    #
    #
    #
    if isinstance(cur, str):
        return cur
    if isinstance(cur, int) and not isinstance(cur, bool):
        return str(cur)
    if isinstance(cur, float):
        return str(int(cur))
    return ""


def _new_video_trace_id() -> str:
    """

"""
    return str(uuid.uuid4())


def _parse_video_poll(vg_cfg: VideoGenDef, body: bytes) -> tuple[VideoResponse, bool]:
    """














"""
    try:
        raw = json.loads(body)
    except ValueError as exc:
        raise APIError(
            message=f"unmarshal video poll response: {exc}", status_code=0
        ) from exc

    if vg_cfg.wire_shape == "VideoQwen":
        output = raw.get("output") if isinstance(raw, dict) else None
        status = output.get("task_status") if isinstance(output, dict) else None
        if status == "SUCCEEDED":
            return _video_result_from_qwen(vg_cfg, raw), True
        if status in ("FAILED", "CANCELED"):
            raise APIError(message=f"video generation {status}", status_code=0)
        #
        return VideoResponse(), False

    if vg_cfg.wire_shape == "VideoTogether":
        status = raw.get("status") if isinstance(raw, dict) else None
        if status == "completed":
            return _video_result_from_together(vg_cfg, raw), True
        if status in ("failed", "cancelled"):
            raise APIError(message=f"video generation {status}", status_code=0)
        #
        return VideoResponse(), False

    if vg_cfg.wire_shape == "VideoZhipu":
        status = raw.get("task_status") if isinstance(raw, dict) else None
        if status == "SUCCESS":
            return _video_result_from_zhipu(vg_cfg, raw), True
        if status == "FAIL":
            raise APIError(message="video generation failed", status_code=0)
        #
        return VideoResponse(), False

    if vg_cfg.wire_shape == "VideoVidu":
        #
        #
        #
        state = raw.get("state") if isinstance(raw, dict) else None
        if state == "success":
            return _video_result_from_vidu(vg_cfg, raw), True
        if state == "failed":
            msg = raw.get("err_code") if isinstance(raw, dict) else None
            if not isinstance(msg, str) or not msg:
                msg = raw.get("message") if isinstance(raw, dict) else None
            if not isinstance(msg, str) or not msg:
                msg = "operation failed"
            raise APIError(
                message=f"video generation failed: {msg}", status_code=0
            )
        #
        return VideoResponse(), False

    if vg_cfg.wire_shape == "VideoPixVerse":
        #
        #
        #
        resp = raw.get("Resp") if isinstance(raw, dict) else None
        status = resp.get("status") if isinstance(resp, dict) else None
        if status == 1:
            return _video_result_from_pixverse(vg_cfg, raw), True
        if status in (7, 8):
            raise APIError(
                message=f"video generation failed (status {status})", status_code=0
            )
        #
        return VideoResponse(), False

    if vg_cfg.wire_shape == "VideoMinimax":
        #
        #
        #
        status = raw.get("status") if isinstance(raw, dict) else None
        if status == "Success":
            return VideoResponse(), True
        if status == "Fail":
            raise APIError(message="video generation failed", status_code=0)
        #
        return VideoResponse(), False

    if vg_cfg.wire_shape == "VideoVeo":
        #
        #
        #
        done = raw.get("done") if isinstance(raw, dict) else None
        if done is not True:
            return VideoResponse(), False
        err_obj = raw.get("error") if isinstance(raw, dict) else None
        if isinstance(err_obj, dict):
            msg = err_obj.get("message")
            if not isinstance(msg, str) or not msg:
                msg = "operation failed"
            raise APIError(
                message=f"video generation failed: {msg}", status_code=0
            )
        #
        #
        #
        result = _video_result_from_veo(vg_cfg, raw)
        if not result.videos or not result.videos[0].url:
            raise APIError(
                message="video generation: operation done but carried no video uri",
                status_code=0,
            )
        return result, True

    if vg_cfg.wire_shape == "VideoVertexVeo":
        #
        #
        #
        done = raw.get("done") if isinstance(raw, dict) else None
        if done is not True:
            return VideoResponse(), False
        err_obj = raw.get("error") if isinstance(raw, dict) else None
        if isinstance(err_obj, dict):
            msg = err_obj.get("message")
            if not isinstance(msg, str) or not msg:
                msg = "operation failed"
            raise APIError(
                message=f"video generation failed: {msg}", status_code=0
            )
        result = _video_result_from_vertex_veo(vg_cfg, raw)
        #
        #
        if not result.videos or not result.videos[0].bytes:
            raise APIError(
                message="video generation: operation done but carried no video bytes",
                status_code=0,
            )
        return result, True

    if vg_cfg.wire_shape == "VideoBedrock":
        #
        #
        #
        status = raw.get("status") if isinstance(raw, dict) else None
        if status == "Completed":
            #
            #
            #
            #
            result = _video_result_from_bedrock(vg_cfg, raw)
            if not result.videos or not result.videos[0].url:
                raise APIError(
                    message="video generation: completed but carried no output s3 uri",
                    status_code=0,
                )
            return result, True
        if status == "Failed":
            msg = raw.get("failureMessage") if isinstance(raw, dict) else None
            if not isinstance(msg, str) or not msg:
                msg = "operation failed"
            raise APIError(
                message=f"video generation failed: {msg}", status_code=0
            )
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


def _video_result_from_vidu(vg_cfg: VideoGenDef, raw: dict[str, Any]) -> VideoResponse:
    """


"""
    mime = _video_fallback_mime(vg_cfg)
    creations = raw.get("creations") if isinstance(raw, dict) else None
    if not isinstance(creations, list) or not creations:
        return VideoResponse()
    first = creations[0]
    if not isinstance(first, dict):
        return VideoResponse()
    url = first.get("url")
    return VideoResponse(
        videos=[VideoData(mime_type=mime, url=url if isinstance(url, str) else "")]
    )


def _video_result_from_pixverse(
    vg_cfg: VideoGenDef, raw: dict[str, Any]
) -> VideoResponse:
    """


"""
    mime = _video_fallback_mime(vg_cfg)
    resp = raw.get("Resp") if isinstance(raw, dict) else None
    if not isinstance(resp, dict):
        return VideoResponse()
    url = resp.get("url")
    return VideoResponse(
        videos=[VideoData(mime_type=mime, url=url if isinstance(url, str) else "")]
    )


def _video_result_from_together(
    vg_cfg: VideoGenDef, raw: dict[str, Any]
) -> VideoResponse:
    """


"""
    mime = _video_fallback_mime(vg_cfg)
    outputs = raw.get("outputs") if isinstance(raw, dict) else None
    if not isinstance(outputs, dict):
        return VideoResponse()
    url = outputs.get("video_url")
    return VideoResponse(
        videos=[VideoData(mime_type=mime, url=url if isinstance(url, str) else "")]
    )


def _video_result_from_qwen(
    vg_cfg: VideoGenDef, raw: dict[str, Any]
) -> VideoResponse:
    """


"""
    mime = _video_fallback_mime(vg_cfg)
    output = raw.get("output") if isinstance(raw, dict) else None
    if not isinstance(output, dict):
        return VideoResponse()
    url = output.get("video_url")
    return VideoResponse(
        videos=[VideoData(mime_type=mime, url=url if isinstance(url, str) else "")]
    )


def _resolve_video_file(
    base: str, vg_cfg: VideoGenDef, poll_body: bytes, headers: dict[str, str]
) -> VideoResponse:
    """




"""
    try:
        poll = json.loads(poll_body)
    except ValueError as exc:
        raise APIError(
            message=f"unmarshal video poll for file hop: {exc}", status_code=0
        ) from exc
    file_id = _video_file_id(poll.get("file_id") if isinstance(poll, dict) else None)
    if not file_id:
        raise APIError(
            message="video file hop: terminal poll carried no file_id", status_code=0
        )
    file_url = base + vg_cfg.file_endpoint.replace("{file_id}", file_id)
    file_body = do_get(file_url, headers)
    try:
        file_raw = json.loads(file_body)
    except ValueError as exc:
        raise APIError(
            message=f"unmarshal video file response: {exc}", status_code=0
        ) from exc
    return _video_result_from_minimax_file(vg_cfg, file_raw)


def _video_file_id(v: Any) -> str:
    """
"""
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    return ""


def _video_result_from_minimax_file(
    vg_cfg: VideoGenDef, raw: dict[str, Any]
) -> VideoResponse:
    """

"""
    mime = _video_fallback_mime(vg_cfg)
    file_obj = raw.get("file") if isinstance(raw, dict) else None
    if not isinstance(file_obj, dict):
        return VideoResponse()
    url = file_obj.get("download_url")
    return VideoResponse(
        videos=[VideoData(mime_type=mime, url=url if isinstance(url, str) else "")]
    )


def _video_result_from_veo(vg_cfg: VideoGenDef, raw: dict[str, Any]) -> VideoResponse:
    """




"""
    mime = _video_fallback_mime(vg_cfg)
    response = raw.get("response") if isinstance(raw, dict) else None
    gvr = response.get("generateVideoResponse") if isinstance(response, dict) else None
    samples = gvr.get("generatedSamples") if isinstance(gvr, dict) else None
    if not isinstance(samples, list) or not samples:
        return VideoResponse()
    first = samples[0]
    if not isinstance(first, dict):
        return VideoResponse()
    video = first.get("video")
    uri = video.get("uri") if isinstance(video, dict) else None
    return VideoResponse(
        videos=[VideoData(mime_type=mime, url=uri if isinstance(uri, str) else "")]
    )


def _video_result_from_vertex_veo(
    vg_cfg: VideoGenDef, raw: dict[str, Any]
) -> VideoResponse:
    """






"""
    mime = _video_fallback_mime(vg_cfg)
    response = raw.get("response") if isinstance(raw, dict) else None
    videos = response.get("videos") if isinstance(response, dict) else None
    if not isinstance(videos, list) or not videos:
        return VideoResponse()
    first = videos[0]
    if not isinstance(first, dict):
        return VideoResponse()
    m = first.get("mimeType")
    if isinstance(m, str) and m:
        mime = m
    b64 = first.get("bytesBase64Encoded")
    if not isinstance(b64, str) or not b64:
        return VideoResponse()
    decoded = base64.b64decode(b64)
    return VideoResponse(videos=[VideoData(mime_type=mime, bytes=decoded)])


def _video_result_from_bedrock(
    vg_cfg: VideoGenDef, raw: dict[str, Any]
) -> VideoResponse:
    """





"""
    mime = _video_fallback_mime(vg_cfg)
    odc = raw.get("outputDataConfig") if isinstance(raw, dict) else None
    s3 = odc.get("s3OutputDataConfig") if isinstance(odc, dict) else None
    uri = s3.get("s3Uri") if isinstance(s3, dict) else None
    return VideoResponse(
        videos=[VideoData(mime_type=mime, url=uri if isinstance(uri, str) else "")]
    )


def _append_video_auth(
    url: str, provider: Provider, pname: ProviderName, cfg: Any
) -> str:
    """



"""
    if auth_scheme(pname) != AuthScheme.QUERY_PARAM_KEY or not cfg.auth_query_param:
        return url
    sep = "&" if "?" in url else "?"
    return url + sep + cfg.auth_query_param + "=" + provider.api_key


def _download_video_bytes(
    provider: Provider,
    pname: ProviderName,
    cfg: Any,
    resp: VideoResponse,
    headers: dict[str, str],
) -> VideoResponse:
    """




"""
    for video in resp.videos:
        if not video.url:
            continue
        fetch_url = _append_video_auth(video.url, provider, pname, cfg)
        video.bytes = do_get(fetch_url, headers)
        video.url = ""
    return resp


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


def _video_seed_image_url(parts: list[Part]) -> str:
    """






"""
    seed = None
    for part in parts:
        if part.image is None:
            continue
        if seed is not None:
            raise ValidationError(
                field="parts",
                message="image-to-video conditions on a single seed frame; pass one image part",
            )
        seed = part.image
    if seed is None:
        return ""
    mime = seed.mime_type or "image/png"
    return f"data:{mime};base64,{base64.b64encode(seed.bytes).decode('ascii')}"
