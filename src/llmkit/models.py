"""Hand-coded catalogue runtime (ADR-019). The generated builder classes
in builders/catalogue.py delegate their terminal methods here.

Folds in the providers-namespace runtime (catalogue_providers_*) because
``llmkit.providers`` is the generated subpackage path and Python forbids
shadowing it with a sibling module.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .catalogue import catalogue_by_provider, compiled_in_models
from .providers.generated.providers import ALL_PROVIDER_NAMES
from .structs import LiveResult, ModelInfo, ProviderError
from .types import Capability, Provider

if TYPE_CHECKING:
    from .builders.catalogue import Models, ScopedModels
    from .builders import Client


# Catalogue error sentinels (ADR-019). Provider live calls map to:
#   - ErrModelsNotSupported: provider lacks llm:hasModelsEndpoint
#     (no /v1/models route; nothing to fetch).
#   - ErrModelsScope: HTTP 403 whose body mentions scope (OpenAI's
#     api.model.read scope is the canonical case).
#   - ErrModelsUnavailable: any other non-2xx response or network
#     failure during a live HTTP call.
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
    Phase 3 wires real HTTP; this scaffold inherits the same shape so the
    builder surface is stable. with_capability composes post-fetch."""
    from .builders.catalogue import ScopedModels as _ScopedModels

    # Function-local import: this is the load-bearing cycle break. Module-
    # level imports flow builders/catalogue.py -> models.py (one direction);
    # this single function-local import handles the reverse edge so
    # both sides can keep their module-top imports clean.
    configured = models.client.providers.list()
    scoped_builders = [_ScopedModels(models.client, p, models.cap_filter) for p in configured]
    results = await asyncio.gather(
        *(scoped.list() for scoped in scoped_builders),
        return_exceptions=True,
    )

    all_models: list[ModelInfo] = []
    errors: dict[str, ProviderError] = {}
    for p, r in zip(configured, results):
        if isinstance(r, BaseException):
            # ADR-019 Amendment 1: structured discriminant + message.
            errors[p.name] = ProviderError(kind=classify_catalogue_error(r), message=str(r))
        else:
            all_models.extend(r)

    if models.cap_filter:
        all_models = [m for m in all_models if models.cap_filter in m.capabilities]
    all_models.sort(key=lambda m: (m.provider.name, m.id))
    return LiveResult(models=all_models, errors=errors)


async def catalogue_run_list(scoped: "ScopedModels") -> list[ModelInfo]:
    """Single-provider live HTTP — Phase 3 stub."""
    if scoped.target.name not in catalogue_by_provider:
        raise ErrModelsNotSupported()
    raise ErrModelsUnavailable()


async def catalogue_run_get(scoped: "ScopedModels", id: str) -> ModelInfo:
    """Single-provider live model fetch — Phase 3 stub."""
    _ = id
    if scoped.target.name not in catalogue_by_provider:
        raise ErrModelsNotSupported()
    raise ErrModelsUnavailable()


# === Providers-namespace runtime (hand-coded mirror of go/providers.go) ===


def catalogue_providers_list(client: "Client") -> list[Provider]:
    """Eligibility test per ADR-019: credentials configured on this Client
    AND llm:hasModelsEndpoint declared in the ontology. A Python Client
    carries one provider's credentials, so the result is either a
    single-element list (when its provider has a catalogue endpoint) or
    empty."""
    p = client.provider
    if p.name not in catalogue_by_provider:
        return []
    return [Provider(name=p.name, api_key=p.api_key, base_url=p.base_url)]


def catalogue_providers_supported() -> list[Provider]:
    """Every provider the SDK was built to support — independent of Client
    credentials. Sorted by name for deterministic callers."""
    # ProviderName(str, Enum) -> str() returns "ProviderName.ANTHROPIC"
    # on Python <3.11; .value always returns the wire string ("anthropic").
    names = sorted(n.value for n in ALL_PROVIDER_NAMES)
    return [Provider(name=n, api_key="") for n in names]
