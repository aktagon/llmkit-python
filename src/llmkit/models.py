"""Hand-coded catalogue runtime (ADR-019). The generated builder classes
in builders/catalogue.py delegate their terminal methods here.

Folds in the providers-namespace runtime (catalogue_providers_*) because
``llmkit.providers`` is the generated subpackage path and Python forbids
shadowing it with a sibling module.
"""

from __future__ import annotations

import asyncio
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

from .catalogue import catalogue_by_provider, compiled_in_models, ontology_capabilities
from .middleware import fire_post, fire_pre
from .providers.generated.middleware import Event, MiddlewareOp
from .providers.generated.models_parsers import (
    ParsedModelRecord,
    ParsedModelsPage,
    parse_anthropic_models_response,
    parse_google_models_response,
    parse_openai_cohort_models_response,
)
from .providers.generated.providers import (
    PROVIDERS,
    ProviderSpec,
)
from .providers.generated.provider_info import ProviderInfo, info
from .providers.generated.request import AuthScheme, auth_scheme
from .providers.generated.providers import ProviderName
from .structs import LiveResult, ModelInfo, ProviderError
from .types import Capability, Provider

if TYPE_CHECKING:
    from .builders.catalogue import Models, ScopedModels
    from .builders import Client


_SCOPE_BODY_PATTERN = re.compile(r"scope|permission", re.IGNORECASE)


class ErrModelsNotSupported(Exception):
    def __init__(self, message: str = "llmkit: provider does not expose a models endpoint") -> None:
        super().__init__(message)


class ErrModelsUnavailable(Exception):
    def __init__(self, message: str = "llmkit: provider models endpoint unavailable") -> None:
        super().__init__(message)


class ErrModelsScope(Exception):
    def __init__(self, message: str = "llmkit: api key lacks scope for models endpoint") -> None:
        super().__init__(message)


def classify_catalogue_error(exc: BaseException) -> str:
    """Map a caught exception to the wire-format discriminant carried in
    ProviderError.kind (ADR-019 Amendment 1). Unknown errors fall back
    to "unavailable" — safer than "scope" since scope implies a documented
    retry path."""
    if isinstance(exc, ErrModelsNotSupported):
        return "not_supported"
    if isinstance(exc, ErrModelsScope):
        return "scope"
    return "unavailable"


def catalogue_filter(cap_filter: Capability | None) -> list[ModelInfo]:
    """Walk the compiled-in slice and return records whose capabilities list
    contains cap_filter. Returns a fresh list so callers cannot mutate the
    module-level constant."""
    if not cap_filter:
        return list(compiled_in_models)
    return [m for m in compiled_in_models if cap_filter in m.capabilities]


def catalogue_lookup(id: str) -> ModelInfo | None:
    """Linear scan over the compiled-in slice. Returns None on miss."""
    for m in compiled_in_models:
        if m.id == id:
            return m
    return None


async def catalogue_run_live(models: "Models") -> LiveResult:
    """Fan out per-provider live calls and aggregate into LiveResult.
    Errors land in result.errors as typed ProviderError per Amendment 1.
    with_capability composes post-fetch."""
    from .builders.catalogue import ScopedModels as _ScopedModels

    pc = models.client.provider
    configured = models.client.providers.list()
    scoped_builders = [
        _ScopedModels(
            models.client,
            Provider(name=p.id, api_key=pc.api_key, base_url=pc.base_url),
            models.cap_filter,
        )
        for p in configured
    ]
    results = await asyncio.gather(
        *(scoped.list() for scoped in scoped_builders),
        return_exceptions=True,
    )

    all_models: list[ModelInfo] = []
    errors: dict[str, ProviderError] = {}
    for p, r in zip(configured, results):
        if isinstance(r, BaseException):
            errors[p.slug] = ProviderError(kind=classify_catalogue_error(r), message=str(r))
        else:
            all_models.extend(r)

    if models.cap_filter:
        all_models = [m for m in all_models if models.cap_filter in m.capabilities]
    all_models.sort(key=lambda m: (m.provider.name, m.id))
    return LiveResult(models=all_models, errors=errors)


async def catalogue_run_list(scoped: "ScopedModels") -> list[ModelInfo]:
    """Single-provider live HTTP. Paginates per the catalogue config until
    the parser reports no next cursor; enriches each record with the
    ontology-derived capability list. Middleware fires once per call (not
    per page) so observability stays at the call granularity."""
    cfg = catalogue_by_provider.get(scoped.target.name)
    if cfg is None:
        raise ErrModelsNotSupported()
    pcfg = PROVIDERS.get(scoped.target.name)
    if pcfg is None:
        raise ErrModelsNotSupported()

    base_event = Event(
        op=MiddlewareOp.MODELS_LIST,
        provider=scoped.target.name,
    )
    fire_pre([], base_event)
    start = time.monotonic()
    effective = _effective_provider(scoped)
    try:
        records = await asyncio.to_thread(
            _paginate_sync, effective, pcfg, cfg.endpoint, cfg.pagination, cfg.parser_kind
        )
    except BaseException as exc:
        post = Event(
            op=MiddlewareOp.MODELS_LIST,
            provider=scoped.target.name,
            err=str(exc),
            duration=time.monotonic() - start,
        )
        fire_post([], post)
        raise

    post = Event(
        op=MiddlewareOp.MODELS_LIST,
        provider=scoped.target.name,
        duration=time.monotonic() - start,
    )
    fire_post([], post)
    return _enrich(scoped, records)


async def catalogue_run_get(scoped: "ScopedModels", id: str) -> ModelInfo:
    """Single-provider live model fetch. URL shapes pinned in plan 025."""
    cfg = catalogue_by_provider.get(scoped.target.name)
    if cfg is None:
        raise ErrModelsNotSupported()
    if cfg.parser_kind in ("ParseVertexModels", "ParseBedrockModels"):
        raise ErrModelsNotSupported()
    pcfg = PROVIDERS.get(scoped.target.name)
    if pcfg is None:
        raise ErrModelsNotSupported()

    base_event = Event(
        op=MiddlewareOp.MODELS_LIST,
        provider=scoped.target.name,
        model=id,
    )
    fire_pre([], base_event)
    effective = _effective_provider(scoped)
    try:
        record = await asyncio.to_thread(
            _get_sync, effective, pcfg, cfg.endpoint, id, cfg.parser_kind
        )
    finally:
        fire_post([], Event(op=MiddlewareOp.MODELS_LIST, provider=scoped.target.name, model=id))
    return _enrich(scoped, [record])[0]


# === Providers-namespace runtime (hand-coded mirror of go/providers.go) ===


def catalogue_providers_list(client: "Client") -> list[ProviderInfo]:
    p = client.provider
    if p.name not in catalogue_by_provider:
        return []
    return [info(ProviderName(p.name))]


# === HTTP internals ===


def _effective_provider(scoped: "ScopedModels") -> Provider:
    """Materialise the Provider used for HTTP from the Client's stored
    credentials, not from the user-supplied scoped.target. The target
    carries only the provider name (used for parser dispatch); the
    base_url / api_key live on client.provider where with_base_url
    sets them."""
    pc = scoped.client.provider
    return Provider(
        name=scoped.target.name,
        api_key=pc.api_key,
        base_url=pc.base_url,
    )


def _paginate_sync(
    provider: Provider,
    pcfg: ProviderSpec,
    endpoint: str,
    pagination: str,
    parser_kind: str,
) -> list[ParsedModelRecord]:
    """Synchronous pagination loop. Runs in a worker thread per
    asyncio.to_thread so other live fan-out tasks proceed in parallel.
    urllib is blocking, which is why we don't call it on the event loop
    directly."""
    headers = _build_catalogue_headers(provider, pcfg)
    cursor = ""
    all_records: list[ParsedModelRecord] = []
    while True:
        req_url = _append_cursor(
            _build_catalogue_url(provider, pcfg, endpoint), pagination, cursor
        )
        body = _http_get(req_url, headers)
        page = _dispatch_parser(parser_kind, body)
        all_records.extend(page.records)
        if not page.next_cursor:
            return all_records
        cursor = page.next_cursor


def _get_sync(
    provider: Provider,
    pcfg: ProviderSpec,
    endpoint: str,
    id: str,
    parser_kind: str,
) -> ParsedModelRecord:
    headers = _build_catalogue_headers(provider, pcfg)
    url = _build_catalogue_url(provider, pcfg, f"{endpoint}/{id}")
    body = _http_get(url, headers)
    return _parse_single_record(parser_kind, body)


def _http_get(url: str, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, method="GET")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read() or b""
        status = exc.code
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ErrModelsUnavailable(
            f"llmkit: provider models endpoint unavailable: {exc}"
        ) from exc
    if status >= 200 and status < 300:
        return body
    if status == 403 and _SCOPE_BODY_PATTERN.search(body.decode("utf-8", "replace")):
        raise ErrModelsScope(
            f"llmkit: api key lacks scope for models endpoint (status {status})"
        )
    raise ErrModelsUnavailable(
        f"llmkit: provider models endpoint unavailable (status {status})"
    )


def _dispatch_parser(kind: str, body: bytes) -> ParsedModelsPage:
    if kind == "ParseAnthropicModels":
        return parse_anthropic_models_response(body)
    if kind == "ParseGoogleModels":
        return parse_google_models_response(body)
    if kind == "ParseOpenAICohortModels":
        return parse_openai_cohort_models_response(body)
    raise ErrModelsNotSupported()


def _parse_single_record(kind: str, body: bytes) -> ParsedModelRecord:
    text = body.decode("utf-8", "replace")
    if kind == "ParseAnthropicModels":
        page = parse_anthropic_models_response(f'{{"data":[{text}]}}'.encode())
    elif kind == "ParseGoogleModels":
        page = parse_google_models_response(f'{{"models":[{text}]}}'.encode())
    elif kind == "ParseOpenAICohortModels":
        page = parse_openai_cohort_models_response(f'{{"data":[{text}]}}'.encode())
    else:
        raise ErrModelsNotSupported()
    if not page.records:
        raise ErrModelsUnavailable(f"empty single-record response for {kind}")
    return page.records[0]


def _append_cursor(raw_url: str, pagination: str, cursor: str) -> str:
    if not cursor:
        return raw_url
    sep = "&" if "?" in raw_url else "?"
    if pagination == "CursorByLastID":
        return f"{raw_url}{sep}after_id={urllib.parse.quote(cursor, safe='')}"
    if pagination == "CursorOpaqueToken":
        return f"{raw_url}{sep}pageToken={urllib.parse.quote(cursor, safe='')}"
    return raw_url


def _build_catalogue_url(provider: Provider, pcfg: ProviderSpec, endpoint: str) -> str:
    base = provider.base_url or pcfg.base_url
    url = base + endpoint
    scheme = auth_scheme(ProviderName(provider.name))
    if scheme == AuthScheme.QUERY_PARAM_KEY:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{pcfg.auth_query_param}={urllib.parse.quote(provider.api_key, safe='')}"
    return url


def _build_catalogue_headers(provider: Provider, pcfg: ProviderSpec) -> dict[str, str]:
    headers: dict[str, str] = {}
    scheme = auth_scheme(ProviderName(provider.name))
    if scheme == AuthScheme.BEARER_TOKEN:
        headers[pcfg.auth_header] = pcfg.auth_prefix + " " + provider.api_key
    elif scheme == AuthScheme.HEADER_API_KEY:
        headers[pcfg.auth_header] = provider.api_key
    if pcfg.required_header:
        headers[pcfg.required_header] = pcfg.required_header_value
    return headers


def _enrich(scoped: "ScopedModels", records: list[ParsedModelRecord]) -> list[ModelInfo]:
    provider_name = scoped.target.name
    by_id = ontology_capabilities.get(provider_name, {})
    out: list[ModelInfo] = []
    for rec in records:
        info = ModelInfo(
            id=rec.id,
            provider=Provider(name=provider_name, api_key=""),
            capabilities=by_id.get(rec.id, []),
            display_name=rec.display_name or "",
            description=rec.description or "",
            context_window=rec.context_window or 0,
            max_output=rec.max_output or 0,
            created=rec.created or 0,
        )
        if getattr(scoped, "raw_flag", False):
            info.raw = rec.raw
        out.append(info)
    return out
