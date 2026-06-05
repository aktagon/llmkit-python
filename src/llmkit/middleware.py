""""""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import MiddlewareVetoError
from .providers.generated.middleware import Event, MiddlewareFn, MiddlewarePhase
from .providers.generated.providers import ProviderConfig

if TYPE_CHECKING:
    from .types import Provider


def fire_pre(mws: list[MiddlewareFn], base: Event) -> None:
    """"""
    if not mws:
        return
    ev = _copy_event(base)
    ev.phase = MiddlewarePhase.PRE
    for mw in mws:
        err = mw(ev)
        if err is not None:
            raise MiddlewareVetoError(cause=err)


def fire_post(mws: list[MiddlewareFn], base: Event) -> None:
    """"""
    if not mws:
        return
    ev = _copy_event(base)
    ev.phase = MiddlewarePhase.POST
    for mw in mws:
        try:
            mw(ev)
        except Exception:
            #
            #
            pass


def resolve_model(provider: Provider, cfg: ProviderConfig) -> str:
    """




"""
    if provider.model:
        return provider.model
    if cfg.local:
        #
        #
        from .models import resolve_local_default

        return resolve_local_default(provider, cfg)
    return cfg.default_model


def _copy_event(e: Event) -> Event:
    """"""
    import dataclasses

    return dataclasses.replace(e)
