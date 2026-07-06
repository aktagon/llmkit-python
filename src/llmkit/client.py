"""Public entry points: prompt, prompt_stream, upload_file. Mirrors go/llmkit.go."""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Any, Callable

from .caching import apply_caching
from .errors import APIError, ValidationError, parse_error
from .http import (
    do_multipart_post,
    do_post,
    do_sigv4_post,
    do_stream_post,
    merge_caller_headers,
)
from .middleware import fire_post, fire_pre, resolve_model
from .paths import (
    contains_value,
    deep_merge,
    extract_float_path,
    extract_int_path,
    extract_path,
    merge_into_parent,
    remove_additional_properties,
    set_additional_properties_false,
    set_nested_field,
)
from .providers.generated.caching import caching_config
from .providers.generated.middleware import Event, MiddlewareOp, Usage
from .providers.generated.options import (
    OptionKey,
    SupportedOptionDef,
    model_option_overrides,
    option_overrides,
    supported_options,
)
from .providers.generated.providers import PROVIDERS, ProviderName
from .providers.generated.request import (
    AuthScheme,
    SystemPlacement,
    auth_scheme,
    file_upload_config,
    structured_output,
    system_placement,
)
from .providers.generated.stream import stream_config
from .transforms import select_message_transform, select_tool_def_transform, to_internal
from .types import File, Options, Provider, Request, Response

StreamCallback = Callable[[str], None]


# ADR-055: the opt-in chat-protocol token for OpenAI's Responses API. Pass it to
# Text.protocol to POST the {input} envelope to /v1/responses instead of the
# default Chat Completions {messages} envelope to /v1/chat/completions. It is a
# plain string; c.text.protocol("responses") is equivalent.
Responses = "responses"

_PROTOCOL_WIRE_SHAPES = {Responses: "ChatResponsesOpenAI"}


def _protocol_wire_shape(token: str) -> str:
    """Map a public Protocol token to its llm:ChatWireShape local name. An empty
    string means the token is unknown (an empty token is handled by the caller)."""
    return _PROTOCOL_WIRE_SHAPES.get(token, "")


def resolve_chat_protocol(cfg, token: str):
    """Return cfg with endpoint + chat_wire_shape overridden for a non-default
    chat protocol opt-in (ADR-055 Text.protocol(...)). An empty token keeps the
    default (cfg unchanged). A provider that does not expose the requested
    protocol raises ValidationError(field="protocol") — the loud, uniform error
    the ADR requires — before any network call. cfg is a frozen dataclass; the
    override is a copy, so it never leaks to other calls.
    """
    if not token:
        return cfg
    want = _protocol_wire_shape(token)
    if not want:
        raise ValidationError(field="protocol", message=f"unknown protocol: {token}")
    for cp in cfg.chat_protocols:
        if cp.wire_shape == want:
            return dataclasses.replace(
                cfg, endpoint=cp.endpoint, chat_wire_shape=cp.wire_shape
            )
    raise ValidationError(
        field="protocol",
        message=f"provider {cfg.name!r} does not support protocol {token!r}",
    )


def prompt(
    provider: Provider,
    request: Request,
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
    safety_settings: list | None = None,
    caching: bool = False,
    cache_ttl: float = 0.0,
    middleware: list | None = None,
    request_timeout: float = 600.0,
    raw: bool = False,
    protocol: str = "",
) -> Response:
    """Send a one-shot request to an LLM provider."""
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
        safety_settings=list(safety_settings or []),
        caching=caching,
        cache_ttl=cache_ttl,
        middleware=list(middleware or []),
        request_timeout=request_timeout,
        raw=raw,
    )

    _validate_provider(provider)
    _validate_request(request)
    _validate_options(provider, opts)

    cfg = PROVIDERS.get(provider.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {provider.name}")

    # ADR-055: opt into a non-default chat protocol (Responses). Overrides the
    # endpoint + wire shape on this call's cfg copy; empty keeps the default.
    # Raises ValidationError(field="protocol") before any network call.
    cfg = resolve_chat_protocol(cfg, protocol)

    # Carrier-validate at the single boundary before firing middleware, so a
    # malformed message rejects without a dangling pre-hook (PIPE-008).
    msgs = to_internal(request.messages)

    base_event = Event(
        op=MiddlewareOp.LLM_REQUEST,
        provider=provider.name,
        model=resolve_model(provider, cfg),
    )
    start = time.monotonic()
    fire_pre(opts.middleware, base_event)

    body, headers = _build_request(provider, request, opts, cfg, msgs=msgs)

    if opts.caching:
        try:
            apply_caching(body, provider, opts, cfg)
        except Exception as exc:
            _fire_post_err(opts.middleware, base_event, exc, start)
            raise

    json_body = json.dumps(body).encode("utf-8")
    url = _build_url(provider, cfg)

    try:
        if auth_scheme(ProviderName(provider.name)) == AuthScheme.SIG_V4:
            region = os.environ.get(cfg.region_env_var, "")
            secret_key = os.environ.get(cfg.secret_key_env_var, "")
            session_token = os.environ.get(cfg.session_token_env_var, "")
            resp_body = do_sigv4_post(
                url,
                json_body,
                provider.api_key,
                secret_key,
                session_token,
                region,
                cfg.service_name,
                timeout=opts.request_timeout,
                custom_headers=provider.headers,
            )
        else:
            resp_body = do_post(url, json_body, headers, timeout=opts.request_timeout)
    except APIError as raw_api_err:
        err = parse_error(
            provider.name,
            raw_api_err.status_code,
            raw_api_err.message.encode("utf-8"),
            None,
        )
        _fire_post_err(opts.middleware, base_event, err, start)
        raise err from raw_api_err
    except Exception as exc:
        _fire_post_err(opts.middleware, base_event, exc, start)
        raise

    resp = _parse_response(provider.name, resp_body, cfg.chat_wire_shape)
    if opts.raw:
        try:
            resp.raw = json.loads(resp_body)
        except Exception:
            resp.raw = None
    post_event = Event(
        op=MiddlewareOp.LLM_REQUEST,
        provider=provider.name,
        model=resolve_model(provider, cfg),
        usage=resp.usage,
        duration=time.monotonic() - start,
    )
    fire_post(opts.middleware, post_event)
    return resp


def prompt_stream(
    provider: Provider,
    request: Request,
    on_chunk: StreamCallback,
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
    request_timeout: float = 600.0,
) -> Response:
    """Streaming variant of `prompt`. Calls on_chunk(text) for each delta; returns accumulated response."""
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
        middleware=list(middleware or []),
        request_timeout=request_timeout,
    )

    _validate_provider(provider)
    _validate_request(request)
    _validate_options(provider, opts)

    cfg = PROVIDERS.get(provider.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {provider.name}")

    stream_cfg = stream_config(ProviderName(provider.name))
    if stream_cfg is None:
        raise ValidationError(
            field="provider", message=f"streaming not supported: {provider.name}"
        )

    # Carrier-validate at the single boundary before firing middleware (PIPE-008).
    msgs = to_internal(request.messages)

    base_event = Event(
        op=MiddlewareOp.LLM_REQUEST,
        provider=provider.name,
        model=resolve_model(provider, cfg),
    )
    start = time.monotonic()
    fire_pre(opts.middleware, base_event)

    body, headers = _build_request(provider, request, opts, cfg, msgs=msgs)

    if opts.caching:
        try:
            apply_caching(body, provider, opts, cfg)
        except Exception as exc:
            _fire_post_err(opts.middleware, base_event, exc, start)
            raise

    if stream_cfg.param:
        body[stream_cfg.param] = True

    json_body = json.dumps(body).encode("utf-8")
    url = _build_url(provider, cfg)
    if stream_cfg.endpoint:
        url = _build_stream_url(provider, cfg, stream_cfg)

    chunks: list[str] = []

    def wrapped(chunk: str) -> None:
        chunks.append(chunk)
        on_chunk(chunk)

    try:
        usage, finish_reason = do_stream_post(
            url,
            json_body,
            headers,
            stream_cfg,
            wrapped,
            timeout=opts.request_timeout,
            finish_reason_path=cfg.stream_finish_reason_path,
        )
    except Exception as exc:
        _fire_post_err(opts.middleware, base_event, exc, start)
        raise

    post_event = Event(
        op=MiddlewareOp.LLM_REQUEST,
        provider=provider.name,
        model=resolve_model(provider, cfg),
        usage=usage,
        duration=time.monotonic() - start,
    )
    fire_post(opts.middleware, post_event)
    return Response(text="".join(chunks), usage=usage, finish_reason=finish_reason)


def upload_file(
    provider: Provider,
    source: str | os.PathLike[str] | bytes | bytearray,
    *,
    filename: str | None = None,
    mime_type: str = "",
    middleware: list | None = None,
    request_timeout: float = 600.0,
) -> File:
    """Upload a file to a provider and return a File reference.

    ``source`` may be:

    - ``str`` or ``os.PathLike`` — read the file from disk. The
      multipart filename defaults to ``os.path.basename(source)``;
      pass ``filename=`` to override.
    - ``bytes`` (or ``bytearray``) — upload the buffer directly.
      ``filename=`` is required.

    ``mime_type`` overrides the filename-extension–based detection
    used for the multipart Content-Type header.
    """
    if isinstance(source, (bytes, bytearray)):
        if not filename:
            raise ValidationError(
                field="filename", message="required when source is bytes"
            )
        data = bytes(source)
        name = filename
    else:
        path = os.fspath(source)
        with open(path, "rb") as f:
            data = f.read()
        name = filename or os.path.basename(path)

    _validate_provider(provider)
    cfg = PROVIDERS.get(provider.name)
    if cfg is None:
        raise ValidationError(field="provider", message=f"unknown: {provider.name}")
    fu = file_upload_config(ProviderName(provider.name))
    if fu is None:
        raise ValidationError(
            field="provider", message=f"file upload not supported: {provider.name}"
        )

    mws = list(middleware or [])
    base_event = Event(
        op=MiddlewareOp.UPLOAD,
        provider=provider.name,
        model=resolve_model(provider, cfg),
    )
    start = time.monotonic()
    fire_pre(mws, base_event)

    base = provider.base_url or cfg.base_url
    upload_url = base + fu.endpoint
    if auth_scheme(ProviderName(provider.name)) == AuthScheme.QUERY_PARAM_KEY:
        upload_url = upload_url + "?" + cfg.auth_query_param + "=" + provider.api_key

    headers: dict[str, str] = {}
    scheme = auth_scheme(ProviderName(provider.name))
    if scheme == AuthScheme.BEARER_TOKEN:
        headers[cfg.auth_header] = cfg.auth_prefix + " " + provider.api_key
    elif scheme == AuthScheme.HEADER_API_KEY:
        headers[cfg.auth_header] = provider.api_key
    if cfg.required_header:
        headers[cfg.required_header] = cfg.required_header_value
    if fu.beta_header:
        headers["anthropic-beta"] = fu.beta_header
    # ADR-052: additive; never clobbers the SDK headers above.
    merge_caller_headers(headers, provider.headers)

    extra_fields: dict[str, str] = {}
    if fu.extra_fields_json:
        try:
            parsed = json.loads(fu.extra_fields_json)
            if isinstance(parsed, dict):
                extra_fields = {str(k): str(v) for k, v in parsed.items()}
        except ValueError:
            pass

    if cfg.chat_wire_shape == "ChatGoogle":
        metadata = {"file": {"display_name": name}}
        extra_fields["metadata"] = json.dumps(metadata)
        headers["X-Goog-Upload-Protocol"] = "multipart"

    try:
        resp_body, status_code = do_multipart_post(
            upload_url,
            fu.field_name,
            name,
            data,
            extra_fields,
            headers,
            timeout=request_timeout,
            mime_type=mime_type,
        )
    except Exception as exc:
        _fire_post_err(mws, base_event, exc, start)
        raise

    if status_code >= 400:
        err = parse_error(provider.name, status_code, resp_body, None)
        _fire_post_err(mws, base_event, err, start)
        raise err

    try:
        raw = json.loads(resp_body)
    except ValueError as exc:
        _fire_post_err(mws, base_event, exc, start)
        raise

    from .paths import detect_mime_type

    file = File(mime_type=mime_type or detect_mime_type(name))
    if fu.response_id_path:
        file.id = extract_path(raw, fu.response_id_path)
    if fu.response_uri_path:
        file.uri = extract_path(raw, fu.response_uri_path)
    if fu.response_name_path:
        file.name = extract_path(raw, fu.response_name_path)
    if fu.response_mime_path:
        file.mime_type = extract_path(raw, fu.response_mime_path)

    post_event = Event(
        op=MiddlewareOp.UPLOAD,
        provider=provider.name,
        model=resolve_model(provider, cfg),
        duration=time.monotonic() - start,
    )
    fire_post(mws, post_event)
    return file


# =============================================================================
# Validation
# =============================================================================


def _validate_provider(p: Provider) -> None:
    if not p.api_key:
        raise ValidationError(field="api_key", message="required")


def _validate_request(req: Request) -> None:
    if not req.user and not req.messages:
        raise ValidationError(field="user", message="required")
    # The carrier invariant (ADR-026: each message holds at most one of
    # {text content, tool calls, tool result}) is enforced at the single
    # to_internal boundary (PIPE-008), not here.


def _validate_options(p: Provider, opts: Options) -> None:
    if p.name not in PROVIDERS:
        return
    supported = {o.key: o for o in supported_options(ProviderName(p.name))}

    def require(opt_key: OptionKey, field_name: str) -> None:
        if opt_key not in supported:
            raise ValidationError(
                field=field_name, message=f"not supported by {p.name}"
            )

    if opts.top_k is not None:
        require(OptionKey.TOP_K, "top_k")
    if opts.seed is not None:
        require(OptionKey.SEED, "seed")
    if opts.frequency_penalty is not None:
        require(OptionKey.FREQUENCY_PENALTY, "frequency_penalty")
    if opts.presence_penalty is not None:
        require(OptionKey.PRESENCE_PENALTY, "presence_penalty")
    if opts.thinking_budget is not None:
        require(OptionKey.THINKING_BUDGET, "thinking_budget")
    if opts.reasoning_effort:
        require(OptionKey.REASONING_EFFORT, "reasoning_effort")

    overrides = {o.key: o for o in option_overrides(ProviderName(p.name))}
    if opts.reasoning_effort and OptionKey.REASONING_EFFORT in overrides:
        ov = overrides[OptionKey.REASONING_EFFORT]
        allowed_csv = ",".join(ov.allowed_values)
        if allowed_csv and not contains_value(allowed_csv, opts.reasoning_effort):
            raise ValidationError(
                field="reasoning_effort",
                message=f"invalid value {opts.reasoning_effort!r}, must be one of: {allowed_csv}",
            )


# =============================================================================
# URL and request builders
# =============================================================================


def _build_url(p: Provider, cfg) -> str:
    base = p.base_url or cfg.base_url
    endpoint = cfg.endpoint

    if auth_scheme(ProviderName(p.name)) == AuthScheme.QUERY_PARAM_KEY:
        endpoint = endpoint + "?" + cfg.auth_query_param + "=" + p.api_key

    model = resolve_model(p, cfg)
    endpoint = endpoint.replace("{model}", model)
    endpoint = endpoint.replace("{apiKey}", p.api_key)

    if cfg.region_env_var:
        region = os.environ.get(cfg.region_env_var, "")
        base = base.replace("{region}", region)
    return base + endpoint


def _build_stream_url(p: Provider, cfg, stream_cfg) -> str:
    base = p.base_url or cfg.base_url
    endpoint = stream_cfg.endpoint
    model = resolve_model(p, cfg)
    endpoint = endpoint.replace("{model}", model)
    endpoint = endpoint.replace("{apiKey}", p.api_key)

    if auth_scheme(ProviderName(p.name)) == AuthScheme.QUERY_PARAM_KEY:
        sep = "&" if "?" in endpoint else "?"
        endpoint = endpoint + sep + cfg.auth_query_param + "=" + p.api_key
    return base + endpoint


def _resolve_option_key(
    pname: ProviderName,
    model: str,
    param: OptionKey,
    supported: dict[OptionKey, SupportedOptionDef],
) -> str | None:
    """Wire (JSON) key for ``param`` on ``(provider, model)``.

    Per-model overrides (ADR-024) outrank the provider default table: an exact
    model id wins outright, otherwise the longest-prefix glob wins, and failing
    any override the provider's default supported-options key is used. This is
    the single resolution path; both the max-tokens site and the general option
    loop call it (OPT-005).
    """
    best_key: str | None = None
    best_len = -1
    for ov in model_option_overrides(pname):
        if ov.key != param:
            continue
        if ov.matcher_kind == "id":
            if ov.matcher_value == model:
                return ov.json_key
        else:  # "pattern": literal prefix + single trailing '*'
            prefix = (
                ov.matcher_value[:-1]
                if ov.matcher_value.endswith("*")
                else ov.matcher_value
            )
            if model.startswith(prefix) and len(prefix) > best_len:
                best_key, best_len = ov.json_key, len(prefix)
    if best_len >= 0:
        return best_key
    mapping = supported.get(param)
    return mapping.json_key if mapping is not None else None


def _build_request(
    p: Provider,
    req: Request,
    opts: Options,
    cfg,
    tools: list | None = None,
    *,
    msgs=None,
):
    # msgs is the internal message sum (ADR-026 PIPE-007). The Text/batch/stream
    # paths convert their public Message list via to_internal at the single
    # carrier-validation boundary (PIPE-008); the Agent builds it directly from
    # its trusted history, with no lossy public-Message hop. When msgs is None
    # it is derived here (the common Text/batch call), so unit tests and batch
    # need no change; prompt/prompt_stream pass it in so the carrier check runs
    # before fire_pre (preserving the middleware contract).
    #
    # Deliberate scope limit (vs the TS slice, matching the Go slice): only
    # multi-turn history flows through the sum. The single-turn req.user path —
    # which also carries media (req.files/req.images) — is handled directly in
    # each message transform's elif branch, because _MsgText carries only
    # (role, text). Unifying it is tracked as a follow-up; see CLAUDE.md.
    #
    # tools is the Agent's tool set; Text/batch pass None, so the tool-def step
    # is a no-op there and their wire body stays identical (ADR-026 PIPE-005).
    if msgs is None:
        msgs = to_internal(req.messages)
    body: dict[str, Any] = {}
    headers: dict[str, str] = {}

    model = resolve_model(p, cfg)
    if cfg.model_in_body:
        body["model"] = model

    max_tokens = cfg.default_max_tokens  #gitleaks:allow int assignment; scanner high-entropy false positive
    if opts.max_tokens is not None:
        max_tokens = opts.max_tokens

    pname = ProviderName(p.name)
    supported = {o.key: o for o in supported_options(pname)}

    max_json_key = _resolve_option_key(pname, model, OptionKey.MAX_TOKENS, supported)
    if max_json_key is not None:
        body[max_json_key] = max_tokens

    placement = system_placement(ProviderName(p.name))
    if placement == SystemPlacement.TOP_LEVEL_FIELD:
        if req.system:
            body["system"] = req.system
    elif placement == SystemPlacement.SIBLING_OBJECT:
        if req.system:
            body["system_instruction"] = {"parts": [{"text": req.system}]}

    msg_transform = select_message_transform(cfg)
    msg_transform(body, msgs, req, cfg)

    if tools:
        select_tool_def_transform(cfg)(body, tools)

    if cfg.wraps_options_in:
        opt_body: dict[str, Any] = {}
        _add_options(body, opt_body, opts, p.name, model)
        if max_json_key is not None:
            set_nested_field(opt_body, max_json_key, max_tokens)
            body.pop(max_json_key.split(".", 1)[0], None)
        if opt_body:
            body[cfg.wraps_options_in] = opt_body
    else:
        _add_options(body, body, opts, p.name, model)

    if cfg.safety_settings_wire_path and opts.safety_settings:
        body[cfg.safety_settings_wire_path] = [
            {"category": s.category, "threshold": s.threshold}
            for s in opts.safety_settings
        ]

    if req.schema:
        _add_structured_output(body, headers, req.schema, p.name, cfg)

    # Files API beta (BUG-017): a document/source:file block referencing an
    # uploaded file requires the same anthropic-beta the upload used. Compose
    # with any existing value (e.g. structured output) rather than overwrite.
    if req.files:
        fu = file_upload_config(ProviderName(p.name))
        if fu is not None and fu.beta_header:
            headers["anthropic-beta"] = _append_beta(
                headers.get("anthropic-beta", ""), fu.beta_header
            )

    scheme = auth_scheme(ProviderName(p.name))
    if scheme == AuthScheme.BEARER_TOKEN:
        headers[cfg.auth_header] = cfg.auth_prefix + " " + p.api_key
    elif scheme == AuthScheme.HEADER_API_KEY:
        headers[cfg.auth_header] = p.api_key

    if cfg.required_header:
        headers[cfg.required_header] = cfg.required_header_value

    # ADR-052: additive; never clobbers the provider auth / required header above.
    merge_caller_headers(headers, p.headers)

    # ADR-055 Responses wire-shape body fixup: the Responses API names the
    # output-token cap max_output_tokens and rejects max_tokens with a 400
    # (live-verified 2026-07-02). Every other body field is shared with Chat
    # Completions, so this single rename is the only option-key divergence.
    # Behavior held by responses-openai.json, not the ontology.
    if cfg.chat_wire_shape == "ChatResponsesOpenAI" and "max_tokens" in body:
        body["max_output_tokens"] = body.pop("max_tokens")

    return body, headers


def _add_options(
    root: dict[str, Any], body: dict[str, Any], opts: Options, provider_name: str, model: str
) -> None:
    """Apply generation parameters to body, honouring dotted JSON keys + extra_fields.

    JSON keys may be dotted (e.g. "thinking.budget_tokens") for providers that
    require nested objects. Each option's per-provider OptionOverrideDef may
    also carry extra_fields_json — sibling JSON merged into the same parent
    path (e.g. {"type":"enabled"} alongside Anthropic's thinking.budget_tokens)
    — and root_extra_fields_json (ADR-029 THK-003) — JSON deep-merged at the
    request body ROOT (root differs from body for wraps_options_in providers),
    for options that imply a sibling object elsewhere in the body (e.g.
    {"thinking":{"type":"adaptive"}} alongside Anthropic's output_config.effort).
    """
    pname = ProviderName(provider_name)
    supported = {o.key: o for o in supported_options(pname)}
    overrides = {ov.key: ov for ov in option_overrides(pname)}

    def put(opt_key: OptionKey, value: Any) -> None:
        json_key = _resolve_option_key(pname, model, opt_key, supported)
        if json_key is None:
            return
        set_nested_field(body, json_key, value)
        ov = overrides.get(opt_key)
        if ov and ov.extra_fields_json:
            try:
                extras = json.loads(ov.extra_fields_json)
            except ValueError:
                return
            if isinstance(extras, dict):
                merge_into_parent(body, json_key, extras)
        if ov and ov.root_extra_fields_json:
            try:
                root_extras = json.loads(ov.root_extra_fields_json)
            except ValueError:
                return
            if isinstance(root_extras, dict):
                deep_merge(root, root_extras)

    if opts.temperature is not None:
        put(OptionKey.TEMPERATURE, opts.temperature)
    if opts.top_p is not None:
        put(OptionKey.TOP_P, opts.top_p)
    if opts.top_k is not None:
        put(OptionKey.TOP_K, opts.top_k)
    if opts.stop_sequences:
        put(OptionKey.STOP_SEQUENCES, opts.stop_sequences)
    if opts.seed is not None:
        put(OptionKey.SEED, opts.seed)
    if opts.frequency_penalty is not None:
        put(OptionKey.FREQUENCY_PENALTY, opts.frequency_penalty)
    if opts.presence_penalty is not None:
        put(OptionKey.PRESENCE_PENALTY, opts.presence_penalty)
    if opts.thinking_budget is not None:
        put(OptionKey.THINKING_BUDGET, opts.thinking_budget)
    if opts.reasoning_effort:
        put(OptionKey.REASONING_EFFORT, opts.reasoning_effort)


def _append_beta(existing: str, add: str) -> str:
    """Compose a comma-separated anthropic-beta header value.

    Multiple features that each require a beta (structured output, Files API)
    coexist instead of clobbering one another. Idempotent on repeats.
    """
    if not add:
        return existing
    if not existing:
        return add
    for v in existing.split(","):
        if v.strip() == add:
            return existing
    return existing + "," + add


def _add_structured_output(
    body: dict[str, Any], headers: dict[str, str], schema: str, provider_name: str, cfg
) -> None:
    so = structured_output(ProviderName(provider_name))
    if so is None:
        return
    try:
        parsed_schema = json.loads(schema)
    except ValueError:
        return

    if so.enforce_strict:
        set_additional_properties_false(parsed_schema)
    if so.remove_additional_props:
        remove_additional_properties(parsed_schema)

    if so.beta_header:
        headers["anthropic-beta"] = so.beta_header

    # SiblingOfFormat placement (Google): the format field carries the literal
    # format type (responseMimeType: "application/json") and the schema is an
    # independent sibling at schema_path (responseSchema), not nested in a wrapper.
    if so.schema_placement == "SiblingOfFormat":
        set_nested_field(body, so.format_field, so.format_type)
        set_nested_field(body, so.schema_path, parsed_schema)
        return

    path_parts = so.schema_path.split(".")
    if len(path_parts) == 1:
        format_obj = {
            "type": so.format_type,
            path_parts[0]: parsed_schema,
        }
        set_nested_field(body, so.format_field, format_obj)
    else:
        inner: dict[str, Any] = {
            "name": "response",
            path_parts[1]: parsed_schema,
        }
        if so.enforce_strict:
            inner["strict"] = True
        format_obj = {
            "type": so.format_type,
            path_parts[0]: inner,
        }
        set_nested_field(body, so.format_field, format_obj)


# =============================================================================
# Response parsing
# =============================================================================


def _parse_response(provider: str, body: bytes, chat_wire_shape: str = "") -> Response:
    try:
        raw = json.loads(body)
    except ValueError as exc:
        raise APIError(
            provider=provider,
            message=f"unmarshal response: {exc}",
            status_code=0,
        ) from exc

    # ADR-055: chat_wire_shape is the EFFECTIVE wire shape for this request (after
    # Text.protocol(...) resolution). Only ChatResponsesOpenAI diverges (the
    # output[] envelope); every other value uses the declared response paths.
    if chat_wire_shape == "ChatResponsesOpenAI":
        return _parse_responses_envelope(raw)

    cfg = PROVIDERS[provider]
    text = extract_path(raw, cfg.response_text_path)
    input_tokens = extract_int_path(raw, cfg.usage_input_path)
    output_tokens = extract_int_path(raw, cfg.usage_output_path)
    cache_write, cache_read = _extract_cache_usage(raw, provider)
    reasoning = (
        extract_int_path(raw, cfg.reasoning_tokens_path)
        if cfg.reasoning_tokens_path
        else 0
    )
    cost = (
        extract_float_path(raw, cfg.usage_cost_path) * cfg.usage_cost_scale
        if cfg.usage_cost_path
        else 0.0
    )
    finish_reason = (
        extract_path(raw, cfg.finish_reason_path) if cfg.finish_reason_path else ""
    )
    finish_message = (
        extract_path(raw, cfg.finish_message_path) if cfg.finish_message_path else ""
    )

    tokens = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_write=cache_write,
        cache_read=cache_read,
        reasoning=reasoning,
        cost=cost,
    )
    return Response(
        text=text,
        usage=tokens,
        finish_reason=finish_reason,
        finish_message=finish_message,
    )


def _parse_responses_envelope(raw: dict[str, Any]) -> Response:
    """Extract text + usage from OpenAI's Responses reply (ADR-055). Unlike Chat
    Completions (choices[].message.content), the reply is an output[] array whose
    message item carries content[] blocks of type "output_text"; usage is
    input_tokens/output_tokens with cached + reasoning sub-details. Live-anchored
    2026-07-02. Hand-coded per wire shape, symmetric with transform_responses_input.
    """
    return Response(
        text=_extract_responses_text(raw),
        usage=Usage(
            input=extract_int_path(raw, "usage.input_tokens"),
            output=extract_int_path(raw, "usage.output_tokens"),
            cache_read=extract_int_path(raw, "usage.input_tokens_details.cached_tokens"),
            reasoning=extract_int_path(
                raw, "usage.output_tokens_details.reasoning_tokens"
            ),
        ),
        finish_reason=extract_path(raw, "status"),
    )


def _extract_responses_text(raw: dict[str, Any]) -> str:
    """Walk the Responses output[] array for the first message item and return its
    first output_text block. Iterating (rather than a fixed output[0].content[0]
    path) tolerates a leading reasoning item."""
    output = raw.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict) or c.get("type") != "output_text":
                continue
            text = c.get("text")
            if isinstance(text, str):
                return text
    return ""


def _extract_cache_usage(raw: dict[str, Any], provider: str) -> tuple[int, int]:
    cc = caching_config(ProviderName(provider))
    if cc is None:
        return 0, 0
    write = extract_int_path(raw, cc.write_tokens_path) if cc.write_tokens_path else 0
    read = extract_int_path(raw, cc.read_tokens_path) if cc.read_tokens_path else 0
    return write, read


def _fire_post_err(
    mws: list, base_event: Event, exc: BaseException, start: float
) -> None:
    import dataclasses

    ev = dataclasses.replace(
        base_event, err=str(exc), duration=time.monotonic() - start
    )
    fire_post(mws, ev)
