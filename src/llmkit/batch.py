""""""

from __future__ import annotations

import json
import time
from typing import Any

from .errors import APIError, ValidationError
from .http import do_get, do_multipart_post, do_post, merge_caller_headers
from .job import (
    LifecycleConfig,
    PollBody,
    classify_by_config,
    poll_job,
    _Classification,
)
from .middleware import fire_post, fire_pre, resolve_model, set_event_error
from .paths import extract_path
from .providers.generated.batch import BatchDef, BatchInputMode, batch_config
from .providers.generated.middleware import Event, MiddlewareOp
from .providers.generated.providers import PROVIDERS, ProviderSpec, ProviderName
from .providers.generated.request import AuthScheme, auth_scheme
from .structs import BatchHandle
from .types import Options, Provider, Request, Response

#
#
#
#
#
#
DEFAULT_POLL_DEADLINE = 600.0
DEFAULT_POLL_INTERVAL = 2.0


def submit_batch(
    provider: Provider,
    requests: list[Request],
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    stop_sequences: list[str] | None = None,
    seed: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    thinking_budget: int | None = None,
    reasoning_effort: str = "",
    caching: bool = False,
    cache_ttl: float = 0.0,
    middleware: list | None = None,
    safety_settings: list | None = None,
    request_timeout: float = 600.0,
    raw: bool = False,
) -> BatchHandle:
    """"""
    from .client import _build_request, _validate_provider  # avoid circular at import time

    _validate_provider(provider)
    cfg = PROVIDERS.get(provider.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {provider.name}")

    bc = batch_config(ProviderName(provider.name))
    if bc is None:
        raise ValidationError(field="provider", message=f"batching not supported: {provider.name}")
    if bc.lifecycle is None:
        raise ValidationError(field="provider", message=f"async batching not supported: {provider.name}")

    mws = list(middleware or [])
    opts = Options(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop_sequences=list(stop_sequences or []),
        seed=seed,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        thinking_budget=thinking_budget,
        reasoning_effort=reasoning_effort,
        caching=caching,
        cache_ttl=cache_ttl,
        middleware=mws,
        safety_settings=list(safety_settings or []),
        request_timeout=request_timeout,
    )
    base_event = Event(
        op=MiddlewareOp.BATCH_SUBMIT,
        provider=provider.name,
        model=resolve_model(provider, cfg),
    )
    start = time.monotonic()
    fire_pre(mws, base_event)

    def post_with(exc: BaseException | None) -> None:
        import dataclasses

        ev = dataclasses.replace(
            base_event,
            err="",
            duration=time.monotonic() - start,
        )
        if exc is not None:
            set_event_error(ev, exc)
        fire_post(mws, ev)

    base = provider.base_url or cfg.base_url
    headers = _build_auth_headers(provider, cfg)

    try:
        if bc.input_mode == BatchInputMode.FILE_REFERENCE_INPUT:
            jsonl = _build_batch_jsonl(requests, opts, provider, cfg, bc)
            file_id = _upload_batch_file(base, jsonl, bc, headers, request_timeout)
            body = {
                bc.input_field: file_id,
                "endpoint": bc.endpoint_path,
                "completion_window": bc.completion_window,
            }
            json_body = json.dumps(body).encode("utf-8")
        else:
            from .client import _append_beta

            body, beta_headers = _build_batch_body(requests, opts, provider, cfg, bc)
            #
            #
            #
            #
            for k, v in beta_headers.items():
                headers[k] = (
                    _append_beta(headers.get(k, ""), v) if k == "anthropic-beta" else v
                )
            json_body = json.dumps(body).encode("utf-8")

        create_url = base + bc.lifecycle.create_endpoint
        resp_body = do_post(create_url, json_body, headers, timeout=request_timeout)

        raw = json.loads(resp_body)
        batch_id = extract_path(raw, bc.lifecycle.response_id_path)
        if not batch_id:
            raise APIError(provider=provider.name, message="batch create: empty batch ID", status_code=0)
    except Exception as exc:
        post_with(exc)
        raise

    post_with(None)
    return BatchHandle(id=batch_id, provider=provider, raw=raw)


class _BatchAdapter:
    """

"""

    def __init__(
        self,
        lc: LifecycleConfig,
        handle: BatchHandle,
        base: str,
        bc: BatchDef,
        headers: dict[str, str],
        poll_url: str,
        request_timeout: float,
        raw: bool,
    ) -> None:
        self._lc = lc
        self._handle = handle
        self._base = base
        self._bc = bc
        self._headers = headers
        self._poll_url = poll_url
        self._request_timeout = request_timeout
        self._raw = raw

    def config(self) -> LifecycleConfig:
        return self._lc

    def poll(self) -> PollBody:
        resp_body = do_get(self._poll_url, self._headers, timeout=self._request_timeout)
        try:
            raw = json.loads(resp_body)
        except ValueError as exc:
            raise APIError(
                provider=self._handle.provider.name,
                message=f"unmarshal batch poll response: {exc}",
                status_code=0,
            ) from exc
        return PollBody(raw=raw)

    def classify(self, body: PollBody) -> _Classification:
        return classify_by_config(self._lc, body)

    def result(self, body: PollBody) -> list[Response]:
        #
        #
        #
        return _fetch_batch_results(
            self._handle,
            self._base,
            self._bc,
            self._headers,
            self._request_timeout,
            self._raw,
            status_raw=body.raw,
        )


def _new_batch_adapter(
    handle: BatchHandle,
    request_timeout: float,
    poll_interval: float,
    poll_deadline: float,
    raw: bool,
) -> _BatchAdapter:
    """



"""
    p = handle.provider
    cfg = PROVIDERS.get(p.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {p.name}")
    bc = batch_config(ProviderName(p.name))
    if bc is None or bc.lifecycle is None:
        raise APIError(
            provider=p.name,
            message=f"batch polling not available for {p.name}",
            status_code=0,
        )

    base = p.base_url or cfg.base_url
    headers = _build_auth_headers(p, cfg)

    if bc.lifecycle.polling_endpoint:
        poll_url = base + bc.lifecycle.polling_endpoint.replace("{id}", handle.id)
    else:
        poll_url = base + bc.lifecycle.create_endpoint + "/" + handle.id

    lc = LifecycleConfig(
        noun="batch",
        status_path=bc.lifecycle.polling_status_path,
        done_values=_non_empty(bc.lifecycle.polling_done_value),
        error_values=tuple(bc.lifecycle.polling_error_values),
        poll_interval=poll_interval,
        poll_timeout=poll_deadline,
    )
    return _BatchAdapter(
        lc, handle, base, bc, headers, poll_url, request_timeout, raw
    )


def _non_empty(*values: str) -> tuple[str, ...]:
    """
"""
    return tuple(v for v in values if v)


def wait_batch(
    handle: BatchHandle,
    *,
    request_timeout: float = 600.0,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    poll_deadline: float = DEFAULT_POLL_DEADLINE,
    raw: bool = False,
) -> list[Response]:
    """







"""
    adapter = _new_batch_adapter(
        handle, request_timeout, poll_interval, poll_deadline, raw or handle.raw
    )
    return poll_job(adapter)


def _build_batch_body(
    reqs: list[Request],
    opts: Options,
    provider: Provider,
    cfg: ProviderSpec,
    bc: BatchDef,
) -> tuple[dict[str, Any], dict[str, str]]:
    """



"""
    from .caching import apply_caching
    from .client import _append_beta, _build_request

    items: list[dict[str, Any]] = []
    beta_headers: dict[str, str] = {}
    for i, req in enumerate(reqs):
        req_body, req_headers = _build_request(provider, req, opts, cfg)
        beta = req_headers.get("anthropic-beta")
        if beta:
            beta_headers["anthropic-beta"] = _append_beta(
                beta_headers.get("anthropic-beta", ""), beta
            )
        if opts.caching:
            apply_caching(req_body, provider, opts, cfg)
        if bc.item_body_field:
            item = {
                "custom_id": f"req-{i}",
                bc.item_body_field: req_body,
            }
        else:
            item = req_body
        items.append(item)
    payload = {bc.request_wrapper: items} if bc.request_wrapper else {"requests": items}
    return payload, beta_headers


def _build_batch_jsonl(
    reqs: list[Request],
    opts: Options,
    provider: Provider,
    cfg: ProviderSpec,
    bc: BatchDef,
) -> bytes:
    from .caching import apply_caching
    from .client import _build_request

    lines: list[str] = []
    for i, req in enumerate(reqs):
        req_body, _ = _build_request(provider, req, opts, cfg)
        if opts.caching:
            apply_caching(req_body, provider, opts, cfg)
        line = {
            "custom_id": f"req-{i}",
            "method": "POST",
            "url": bc.endpoint_path,
            "body": req_body,
        }
        lines.append(json.dumps(line))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _upload_batch_file(
    base: str,
    jsonl: bytes,
    bc: BatchDef,
    headers: dict[str, str],
    timeout: float,
) -> str:
    upload_url = base + "/v1/files"
    fields = {"purpose": bc.file_purpose}
    resp_data, status_code = do_multipart_post(
        upload_url, "file", "batch_input.jsonl", jsonl, fields, headers, timeout=timeout
    )
    if status_code >= 400:
        raise APIError(
            status_code=status_code,
            message=resp_data.decode("utf-8", errors="replace"),
            retryable=status_code == 429 or status_code >= 500,
        )
    raw = json.loads(resp_data)
    file_id = extract_path(raw, "id")
    if not file_id:
        raise APIError(message="batch file upload: empty file ID", status_code=0)
    return file_id


def _fetch_batch_results(
    handle: BatchHandle,
    base: str,
    bc: BatchDef,
    headers: dict[str, str],
    timeout: float,
    raw: bool = False,
    status_raw: dict[str, Any] | None = None,
) -> list[Response]:
    """



"""
    from .client import _parse_response

    lc = bc.lifecycle
    assert lc is not None

    if lc.result_file_id_path:
        if status_raw is None:
            poll_url = base + lc.create_endpoint + "/" + handle.id
            status_body = do_get(poll_url, headers, timeout=timeout)
            status_raw = json.loads(status_body)
        file_id = extract_path(status_raw, lc.result_file_id_path)
        if not file_id:
            raise APIError(provider=handle.provider.name, message="batch results: empty output file ID", status_code=0)
        file_url = base + lc.file_content_endpoint.replace("{id}", file_id)
        resp_body = do_get(file_url, headers, timeout=timeout)
    elif lc.result_endpoint:
        result_url = base + lc.result_endpoint.replace("{id}", handle.id)
        resp_body = do_get(result_url, headers, timeout=timeout)
    else:
        raise APIError(
            provider=handle.provider.name,
            message=f"batch result endpoint not configured for {handle.provider.name}",
            status_code=0,
        )

    return _parse_batch_results(handle.provider.name, resp_body, bc, raw)


def _parse_batch_results(provider: str, data: bytes, bc: BatchDef, raw: bool = False) -> list[Response]:
    from .client import _parse_response

    out: list[Response] = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        response_bytes = line.encode("utf-8")
        inner_for_raw: Any = None
        if bc.result_body_path:
            try:
                wrapper = json.loads(line)
            except ValueError:
                continue
            inner = _navigate_map_path(wrapper, bc.result_body_path)
            if inner is None:
                continue
            inner_for_raw = inner
            response_bytes = json.dumps(inner).encode("utf-8")
        try:
            parsed = _parse_response(provider, response_bytes)
        except Exception:
            continue
        if raw:
            if inner_for_raw is not None:
                parsed.raw = inner_for_raw
            else:
                try:
                    parsed.raw = json.loads(line)
                except Exception:
                    parsed.raw = None
        out.append(parsed)
    return out


def _navigate_map_path(data: dict[str, Any], path: str) -> dict[str, Any] | None:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current if isinstance(current, dict) else None


def _build_auth_headers(p: Provider, cfg: ProviderSpec) -> dict[str, str]:
    headers: dict[str, str] = {}
    scheme = auth_scheme(ProviderName(p.name))
    if scheme == AuthScheme.BEARER_TOKEN:
        headers[cfg.auth_header] = cfg.auth_prefix + " " + p.api_key
    elif scheme == AuthScheme.HEADER_API_KEY:
        headers[cfg.auth_header] = p.api_key
    if cfg.required_header:
        headers[cfg.required_header] = cfg.required_header_value
    #
    merge_caller_headers(headers, p.headers)
    return headers
