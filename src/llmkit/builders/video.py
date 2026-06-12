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
import os
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from ..errors import APIError, ValidationError
from ..http import do_get, do_post, do_sigv4_get, do_sigv4_post
from ..image import Part, _image_auth_headers
from ..middleware import fire_post, fire_pre
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
    # output_uri is the caller-supplied destination S3 URI for output-uri
    # delivery providers (Bedrock Nova Reel writes the mp4 to the caller's own
    # S3 bucket). Required when the provider's config sets requires_output_uri;
    # ignored otherwise. Set it on the builder via Video.output_uri.
    output_uri: str = ""


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
    # VID-005: output-uri providers (Bedrock Nova Reel) write the video to the
    # caller's own S3 bucket, so the submit MUST carry a destination URI. Reject
    # pre-flight rather than letting the provider 400.
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
    """POST the submit body per wire shape (never by provider name) and
    return the provider-assigned poll handle id.

      - VideoGrok (xAI), VideoZhipu (CogVideoX), and VideoTogether share the
        simple {model, prompt} submit body. They differ only in which
        response field carries the poll handle: Grok returns it as
        request_id, Zhipu and Together as the top-level id.
      - VideoQwen (DashScope) nests the prompt under an ``input`` object
        ({model, input:{prompt}}) and requires the X-DashScope-Async: enable
        header.
      - VideoVeo carries the model in the submit PATH; body has no model field.
      - VideoBedrock (Nova Reel) nests the prompt under modelInput, carries the
        caller S3 URI under outputDataConfig, and is signed with SigV4 (not the
        bearer/query-param header map).

    The body and any per-shape headers are selected by wire shape; the poll
    handle id is always read from the config-declared dotted path (OQ7).
    """
    # Submit endpoint from the config-declared base + relative path (Option D);
    # handle id from the config-declared dotted path (OQ7).
    post_headers = headers
    if vg_cfg.wire_shape == "VideoQwen":
        body: dict[str, Any] = {
            "model": model,
            "input": {"prompt": _join_prompt_text(parts)},
        }
        # DashScope's async submit requires this header; set per-request only so
        # it never leaks into the shared auth-header map.
        post_headers = {**headers, "X-DashScope-Async": "enable"}
    elif vg_cfg.wire_shape == "VideoVeo":
        # Veo carries the model in the submit PATH (:predictLongRunning), not the
        # body — so the body has no model field. The prompt nests under
        # instances[]; the optional parameters object is omitted on the
        # prompt-only hot path.
        body = {"instances": [{"prompt": _join_prompt_text(parts)}]}
    elif vg_cfg.wire_shape == "VideoBedrock":
        # Nova Reel carries the model in the BODY (modelId, unlike the Converse
        # chat path) and writes the mp4 to the caller's S3 bucket. The optional
        # videoGenerationConfig {durationSeconds, fps, dimension, seed} is
        # omitted on the prompt-only hot path (provider defaults apply).
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
    json_body = json.dumps(body).encode("utf-8")
    # {model} in the submit endpoint is substituted with the per-call model
    # (Veo's :predictLongRunning path); a no-op for providers that carry the
    # model in the body. Query-param auth (Google ?key=) is appended last.
    submit_url = _append_video_auth(
        base_url + vg_cfg.gen_endpoint.replace("{model}", model), provider, pname, cfg
    )
    if auth_scheme(pname) == AuthScheme.SIG_V4:
        # Bedrock signs every request (SigV4); the bearer/query-param header map
        # does not apply. Region/secret/session come from the AWS env vars.
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

    base = _video_base_url(p, cfg, vg_cfg)
    headers = _image_auth_headers(p, cfg, pname)

    # Bedrock (SigV4) signs the poll GET and carries the handle ARN as a single
    # percent-encoded path segment (its ':' and '/' must not split into extra
    # segments). url.PathEscape's Python twin is quote(arn, safe=":"): it encodes
    # '/' to %2F (keeping one path segment) but leaves ':' literal, matching how
    # Bedrock's SigV4 canonicalizes the Converse model id's ':'. The signer
    # canonicalizes the escaped path, so the signed path equals the wire path.
    # Every other provider uses the verbatim {id} substitution and the bearer/
    # query-param auth path.
    sig_v4 = auth_scheme(pname) == AuthScheme.SIG_V4
    region = secret_key = session_token = ""
    if sig_v4:
        poll_url = base + vg_cfg.poll_endpoint.replace("{id}", quote(handle.id, safe=":"))
        region = os.environ.get(cfg.region_env_var, "")
        secret_key = os.environ.get(cfg.secret_key_env_var, "")
        session_token = os.environ.get(cfg.session_token_env_var, "")
    else:
        poll_url = _append_video_auth(
            _video_poll_url(vg_cfg.poll_endpoint, base, handle.id), p, pname, cfg
        )

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
        if sig_v4:
            resp_body = do_sigv4_get(
                poll_url,
                p.api_key,
                secret_key,
                session_token,
                region,
                cfg.service_name,
            )
        else:
            resp_body = do_get(poll_url, headers)
        resp, done = _parse_video_poll(vg_cfg, resp_body)
        if done:
            # Two-hop providers (vg_cfg.file_endpoint set, e.g. minimax): the
            # terminal poll carried a file reference, not a video URL — resolve
            # it with one more GET before returning.
            if vg_cfg.file_endpoint:
                resp = _resolve_video_file(base, vg_cfg, resp_body, headers)
            # Delivery dispatch (VID-005). Download-delivery providers (Veo)
            # returned a temporary fetch URI in VideoData.url; GET it and fill
            # VideoData.bytes (clearing url, per the source-XOR contract). Url-
            # and output-uri-delivery providers leave the url.
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
    """Resolve the base for the video API (Option D): an explicit per-client
    override wins (tests point it at a mock; users at a proxy), else the
    provider's distinct video base (vg_cfg.video_base_url) when the video host
    differs from chat, else the chat base. Endpoints are always relative paths
    joined to this base — never absolute — so the host stays overridable."""
    if provider.base_url:
        return provider.base_url
    base = vg_cfg.video_base_url or cfg.base_url
    # SigV4 hosts carry a {region} placeholder (Bedrock:
    # bedrock-runtime.{region}.amazonaws.com) resolved from the region env var;
    # a no-op for every provider without the placeholder.
    if cfg.region_env_var:
        base = base.replace("{region}", os.environ.get(cfg.region_env_var, ""))
    return base


def _video_poll_url(poll_endpoint: str, base: str, id: str) -> str:
    """Substitute {id} in the config poll template (an A-Box fact, OQ7) and
    join it to the resolved video base."""
    return base + poll_endpoint.replace("{id}", id)


def _lookup_handle_field(raw: Any, path: str) -> str:
    """Descend a dotted path (e.g. "id", "output.task_id") through the decoded
    submit response, returning the string leaf or "" if any segment is missing
    or the leaf is not a string."""
    if not path:
        return ""
    cur: Any = raw
    for seg in path.split("."):
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(seg)
    return cur if isinstance(cur, str) else ""


def _parse_video_poll(vg_cfg: VideoGenDef, body: bytes) -> tuple[VideoResponse, bool]:
    """Decode one poll response per wire shape. Returns (resp, done):

      - done=False when the job is still pending (caller keeps polling).
      - done=True with the finished VideoResponse when status is
        terminal-success.
      - raises when the job failed or expired.

    VideoGrok: {"status": "...", "video": {"url", "duration"}} or
    {"status": "failed", "error": {"code", "message"}}.
    VideoZhipu: {"task_status": "SUCCESS"|"FAIL"|"PROCESSING",
    "video_result": [{"url"}]}.
    VideoTogether: {"status": "completed"|"failed"|"cancelled"|"queued"|
    "in_progress", "outputs": {"video_url"}}.
    VideoQwen: {"output": {"task_status": "SUCCEEDED"|"FAILED"|"CANCELED"|
    "PENDING"|"RUNNING"|"UNKNOWN", "video_url"}}.
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
        # PENDING, RUNNING, UNKNOWN (or any non-terminal status)
        return VideoResponse(), False

    if vg_cfg.wire_shape == "VideoTogether":
        status = raw.get("status") if isinstance(raw, dict) else None
        if status == "completed":
            return _video_result_from_together(vg_cfg, raw), True
        if status in ("failed", "cancelled"):
            raise APIError(message=f"video generation {status}", status_code=0)
        # queued, in_progress (or any non-terminal status)
        return VideoResponse(), False

    if vg_cfg.wire_shape == "VideoZhipu":
        status = raw.get("task_status") if isinstance(raw, dict) else None
        if status == "SUCCESS":
            return _video_result_from_zhipu(vg_cfg, raw), True
        if status == "FAIL":
            raise APIError(message="video generation failed", status_code=0)
        # PROCESSING (or any non-terminal status)
        return VideoResponse(), False

    if vg_cfg.wire_shape == "VideoMinimax":
        # Two-hop: terminal-success yields a file_id, not a URL. Report done
        # with an empty result; _wait_video performs the file-retrieve hop
        # (gated on vg_cfg.file_endpoint) and fills the URL.
        status = raw.get("status") if isinstance(raw, dict) else None
        if status == "Success":
            return VideoResponse(), True
        if status == "Fail":
            raise APIError(message="video generation failed", status_code=0)
        # Queueing, Preparing, Processing (or any non-terminal status)
        return VideoResponse(), False

    if vg_cfg.wire_shape == "VideoVeo":
        # Operation-based LRO: poll until done=True (the long-running-operation
        # done flag, not a status string). A done op carrying an error object is
        # a terminal failure; otherwise the response holds the finished video.
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
        # A done op with neither error nor a usable uri must surface as an error,
        # not a silent zero-byte success: download delivery would otherwise GET
        # nothing and return a VideoData with empty bytes and empty url.
        result = _video_result_from_veo(vg_cfg, raw)
        if not result.videos or not result.videos[0].url:
            raise APIError(
                message="video generation: operation done but carried no video uri",
                status_code=0,
            )
        return result, True

    if vg_cfg.wire_shape == "VideoBedrock":
        # Bedrock async-invoke status (GetAsyncInvoke): Completed terminal-success,
        # Failed terminal-error (failureMessage), InProgress pending. On success
        # the provider wrote the mp4 to the caller's S3 bucket and echoes the URI.
        status = raw.get("status") if isinstance(raw, dict) else None
        if status == "Completed":
            # A Completed invocation that echoes no output s3 uri must surface as
            # an error, not a silent empty success (mirrors the Veo done+no-uri
            # guard): the caller would otherwise get a "successful" VideoResponse
            # whose url is empty and never find the mp4.
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
        # InProgress (or any non-terminal status)
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
        # pending (or any non-terminal status)
        return VideoResponse(), False

    # Unknown shape rejected (not defaulted to Grok): a forgotten poll arm
    # fails loud instead of hanging on a never-terminal status.
    raise APIError(
        message=f"video poll: unsupported wire shape {vg_cfg.wire_shape!r}",
        status_code=0,
    )


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


def _video_result_from_zhipu(vg_cfg: VideoGenDef, raw: dict[str, Any]) -> VideoResponse:
    """Extract the finished video from a Zhipu CogVideoX poll response. Zhipu
    uses url delivery: the finished video sits at video_result[0].url (no
    duration field on the result), so VideoData.url carries the temporary
    Zhipu-hosted URL and bytes stays empty."""
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


def _video_result_from_together(
    vg_cfg: VideoGenDef, raw: dict[str, Any]
) -> VideoResponse:
    """Extract the finished video from a Together poll response. Together uses
    url delivery: the finished video sits at outputs.video_url, so
    VideoData.url carries the temporary Together-hosted URL and bytes stays
    empty."""
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
    """Extract the finished video from a DashScope (Qwen) poll response. Qwen
    uses url delivery: the finished video sits at output.video_url, so
    VideoData.url carries the temporary DashScope-hosted URL and bytes stays
    empty."""
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
    """Perform the two-hop file-retrieve step for providers whose terminal poll
    yields a file reference rather than a finished video URL (vg_cfg.file_endpoint
    set, e.g. minimax): extract the file id from the terminal poll body, GET the
    file endpoint (joined to the resolved video base), and extract the finished
    reference. file-id and result locations are wire-shape-keyed (the transform);
    the endpoint is config."""
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
    """Read the minimax terminal poll's file_id, which the API may encode as a
    string or a (large) integer."""
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    return ""


def _video_result_from_minimax_file(
    vg_cfg: VideoGenDef, raw: dict[str, Any]
) -> VideoResponse:
    """Extract the finished video from a minimax file-retrieve response. minimax
    uses url delivery: the download URL sits at file.download_url, so
    VideoData.url carries it and bytes stays empty."""
    mime = _video_fallback_mime(vg_cfg)
    file_obj = raw.get("file") if isinstance(raw, dict) else None
    if not isinstance(file_obj, dict):
        return VideoResponse()
    url = file_obj.get("download_url")
    return VideoResponse(
        videos=[VideoData(mime_type=mime, url=url if isinstance(url, str) else "")]
    )


def _video_result_from_veo(vg_cfg: VideoGenDef, raw: dict[str, Any]) -> VideoResponse:
    """Extract the finished video reference from a Veo LRO poll response. Veo
    uses download delivery: the response carries a temporary Files-API download
    URI at response.generateVideoResponse.generatedSamples[0].video.uri. This
    places it in VideoData.url; the _wait_video download step
    (output_delivery=DeliveryDownload) then fetches the bytes into
    VideoData.bytes and clears url."""
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


def _video_result_from_bedrock(
    vg_cfg: VideoGenDef, raw: dict[str, Any]
) -> VideoResponse:
    """Extract the finished video reference from a Bedrock Nova Reel poll
    response. Bedrock uses output-uri delivery: the provider wrote the mp4 to
    the caller's own S3 bucket and the finished poll echoes the S3 URI at
    outputDataConfig.s3OutputDataConfig.s3Uri. The SDK surfaces it as
    VideoData.url with bytes empty — the _wait_video delivery step never
    downloads it (only DeliveryDownload fetches), so the caller fetches from S3
    with their own tooling (VID-005; ADR-034 open question 4)."""
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
    """Append the provider's query-param API key to a video URL when the
    provider authenticates that way (Google ?key=); a no-op for bearer-header
    providers (every other video provider). Picks ? or & based on whether the
    URL already carries a query string (the Files-API download URI arrives with
    ?alt=media)."""
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
    """Fetch the finished video for download-delivery providers
    (vg_cfg.output_delivery == DeliveryDownload, e.g. Veo). The poll result
    placed the temporary fetch URI in VideoData.url; this GETs each one
    (carrying the provider's query-param auth when applicable) and moves the
    payload into VideoData.bytes, clearing url so the source-XOR contract holds
    (VID-004): download delivery returns bytes, never a url."""
    for video in resp.videos:
        if not video.url:
            continue
        fetch_url = _append_video_auth(video.url, provider, pname, cfg)
        video.bytes = do_get(fetch_url, headers)
        video.url = ""
    return resp


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
