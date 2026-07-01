"""Caching lifecycle: explicit (inline mutations) and resource (pre-flight request)."""

from __future__ import annotations

import json
import time
from typing import Any

from .errors import APIError, ValidationError
from .http import do_post, merge_caller_headers
from .middleware import fire_post, fire_pre, resolve_model
from .paths import extract_path
from .providers.generated.caching import CachingDef, CachingMode, caching_config
from .providers.generated.middleware import Event, MiddlewareOp
from .providers.generated.providers import PROVIDERS, ProviderSpec, ProviderName
from .providers.generated.request import AuthScheme, SystemPlacement, auth_scheme, system_placement
from .types import Options, Provider


def apply_caching(
    body: dict[str, Any],
    provider: Provider,
    opts: Options,
    cfg: ProviderSpec,
) -> None:
    """Mutate body to enable caching. Dispatches on the provider's CachingMode."""
    cc = caching_config(ProviderName(provider.name))
    if cc is None:
        raise ValidationError(field="caching", message=f"not supported by {provider.name}")

    if cc.mode == CachingMode.AUTOMATIC_CACHING:
        return
    if cc.mode == CachingMode.EXPLICIT_CACHING:
        _apply_explicit(body, cc, cfg)
        return
    if cc.mode == CachingMode.RESOURCE_CACHING:
        _apply_resource(body, provider, opts, cc, cfg)
        return
    raise ValidationError(field="caching", message=f"unknown caching mode: {cc.mode}")


def _apply_explicit(body: dict[str, Any], cc: CachingDef, cfg: ProviderSpec) -> None:
    control_type = cc.control_type or "ephemeral"
    placement = system_placement(ProviderName(cfg.name))

    if placement == SystemPlacement.TOP_LEVEL_FIELD:
        sys_val = body.get("system")
        if isinstance(sys_val, str) and sys_val:
            body["system"] = [
                {"type": "text", "text": sys_val, "cache_control": {"type": control_type}}
            ]
        return

    if placement == SystemPlacement.MESSAGE_IN_ARRAY:
        msgs = body.get("messages")
        if not isinstance(msgs, list):
            return
        for i in range(len(msgs) - 1, -1, -1):
            msg = msgs[i]
            if not isinstance(msg, dict) or msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": control_type}}
                ]
            return


def _apply_resource(
    body: dict[str, Any],
    provider: Provider,
    opts: Options,
    cc: CachingDef,
    cfg: ProviderSpec,
) -> None:
    if cc.lifecycle is None:
        raise ValidationError(field="caching", message="resource caching requires lifecycle config")

    lc = cc.lifecycle
    model = resolve_model(provider, cfg)

    base_event = Event(
        op=MiddlewareOp.CACHE_CREATE,
        provider=provider.name,
        model=resolve_model(provider, cfg),
    )
    start = time.monotonic()
    fire_pre(opts.middleware, base_event)

    ttl_secs = int(opts.cache_ttl) if opts.cache_ttl > 0 else int(cc.default_ttl or "0")
    ttl_str = f"{ttl_secs}s" if ttl_secs > 0 else (cc.default_ttl or "300s")

    create_body: dict[str, Any] = {
        "model": f"models/{model}",
        "ttl": ttl_str,
    }
    if "system_instruction" in body:
        create_body["contents"] = [
            {"role": "user", "parts": [{"text": "cache"}]},
        ]
        create_body["systemInstruction"] = body["system_instruction"]

    create_json = json.dumps(create_body).encode("utf-8")

    base = provider.base_url or cfg.base_url
    create_url = base + lc.create_endpoint
    scheme = auth_scheme(ProviderName(provider.name))
    if scheme == AuthScheme.QUERY_PARAM_KEY:
        create_url = create_url + "?" + cfg.auth_query_param + "=" + provider.api_key

    headers: dict[str, str] = {}
    if scheme == AuthScheme.BEARER_TOKEN:
        headers[cfg.auth_header] = cfg.auth_prefix + " " + provider.api_key
    elif scheme == AuthScheme.HEADER_API_KEY:
        headers[cfg.auth_header] = provider.api_key
    # ADR-052: additive; never clobbers the provider auth above.
    merge_caller_headers(headers, provider.headers)

    try:
        resp_body = do_post(create_url, create_json, headers, timeout=opts.request_timeout)
    except (APIError, Exception) as exc:
        _fire_post_err(opts.middleware, base_event, exc, start)
        raise

    try:
        raw = json.loads(resp_body)
    except ValueError as exc:
        _fire_post_err(opts.middleware, base_event, exc, start)
        raise

    resource_id = extract_path(raw, lc.response_id_path)
    if not resource_id:
        err = APIError(provider=provider.name, message="cache create: empty resource ID", status_code=0)
        _fire_post_err(opts.middleware, base_event, err, start)
        raise err

    body[lc.reference_field] = resource_id
    body.pop("system_instruction", None)

    post_event = Event(
        op=MiddlewareOp.CACHE_CREATE,
        provider=provider.name,
        model=resolve_model(provider, cfg),
        duration=time.monotonic() - start,
    )
    fire_post(opts.middleware, post_event)


def _fire_post_err(mws: list, base_event: Event, exc: BaseException, start: float) -> None:
    import dataclasses

    ev = dataclasses.replace(base_event, err=str(exc), duration=time.monotonic() - start)
    fire_post(mws, ev)
