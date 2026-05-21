"""





"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .catalogue import catalogue_by_provider, compiled_in_models
from .providers.generated.providers import ALL_PROVIDER_NAMES
from .structs import LiveResult, ModelInfo
from .types import Capability, Provider

if TYPE_CHECKING:
    from .builders.catalogue import Models, ScopedModels
    from .builders import Client


#
#
#
#
#
#
#
class ErrModelsNotSupported(Exception):
    def __init__(self, message: str = "llmkit: provider does not expose a models endpoint") -> None:
        super().__init__(message)


class ErrModelsUnavailable(Exception):
    def __init__(self, message: str = "llmkit: provider models endpoint unavailable") -> None:
        super().__init__(message)


class ErrModelsScope(Exception):
    def __init__(self, message: str = "llmkit: api key lacks scope for models endpoint") -> None:
        super().__init__(message)


def catalogue_filter(cap_filter: Capability | None) -> list[ModelInfo]:
    """

"""
    if not cap_filter:
        return list(compiled_in_models)
    return [m for m in compiled_in_models if cap_filter in m.capabilities]


def catalogue_lookup(id: str) -> ModelInfo | None:
    """"""
    for m in compiled_in_models:
        if m.id == id:
            return m
    return None


async def catalogue_run_live(models: "Models") -> LiveResult:
    """

"""
    from .builders.catalogue import ScopedModels as _ScopedModels

    #
    #
    #
    #
    configured = models.client.providers.list()
    scoped_builders = [_ScopedModels(models.client, p, models.cap_filter) for p in configured]
    results = await asyncio.gather(
        *(scoped.list() for scoped in scoped_builders),
        return_exceptions=True,
    )

    all_models: list[ModelInfo] = []
    errors: dict[str, str] = {}
    for p, r in zip(configured, results):
        if isinstance(r, BaseException):
            errors[p.name] = str(r)
        else:
            all_models.extend(r)

    if models.cap_filter:
        all_models = [m for m in all_models if models.cap_filter in m.capabilities]
    all_models.sort(key=lambda m: (m.provider.name, m.id))
    return LiveResult(models=all_models, errors=errors)


async def catalogue_run_list(scoped: "ScopedModels") -> list[ModelInfo]:
    """"""
    if scoped.target.name not in catalogue_by_provider:
        raise ErrModelsNotSupported()
    raise ErrModelsUnavailable()


async def catalogue_run_get(scoped: "ScopedModels", id: str) -> ModelInfo:
    """"""
    _ = id
    if scoped.target.name not in catalogue_by_provider:
        raise ErrModelsNotSupported()
    raise ErrModelsUnavailable()


#


def catalogue_providers_list(client: "Client") -> list[Provider]:
    """



"""
    p = client.provider
    if p.name not in catalogue_by_provider:
        return []
    return [Provider(name=p.name, api_key=p.api_key, base_url=p.base_url)]


def catalogue_providers_supported() -> list[Provider]:
    """
"""
    #
    #
    names = sorted(n.value for n in ALL_PROVIDER_NAMES)
    return [Provider(name=n, api_key="") for n in names]
