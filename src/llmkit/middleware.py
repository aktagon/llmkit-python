"""Handwritten middleware helpers: fire_pre, fire_post, resolve_model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import MiddlewareVetoError
from .providers.generated.middleware import Event, MiddlewareFn, MiddlewarePhase
from .providers.generated.providers import ProviderConfig

if TYPE_CHECKING:
    from .types import Provider


def fire_pre(mws: list[MiddlewareFn], base: Event) -> None:
    """Run pre-phase middlewares. First non-None return aborts with MiddlewareVetoError."""
    if not mws:
        return
    ev = _copy_event(base)
    ev.phase = MiddlewarePhase.PRE
    for mw in mws:
        err = mw(ev)
        if err is not None:
            raise MiddlewareVetoError(cause=err)


def fire_post(mws: list[MiddlewareFn], base: Event) -> None:
    """Run post-phase middlewares. Return values are discarded (observation only)."""
    if not mws:
        return
    ev = _copy_event(base)
    ev.phase = MiddlewarePhase.POST
    for mw in mws:
        try:
            mw(ev)
        except Exception:
            # Post-phase hooks never veto, but we also shouldn't crash the caller.
            # Swallowing here matches Go's `_ = m(ctx, ev)` discard semantics.
            pass


def resolve_model(provider: Provider, cfg: ProviderConfig) -> str:
    """Return the caller-specified model or the provider default.

    For machine-local daemons (cfg.local, ADR-031 / BUG-009c) the default
    resolves from the daemon's live listing — the registry constant is a
    guess that 404s when not pulled. Cloud defaults are curated constants
    and stay registry-resolved."""
    if provider.model:
        return provider.model
    if cfg.local:
        # Function-local import: models.py imports fire_pre/fire_post
        # from this module, so a top-level import would be circular.
        from .models import resolve_local_default

        return resolve_local_default(provider, cfg)
    return cfg.default_model


def _copy_event(e: Event) -> Event:
    """Shallow copy so each phase can set phase/usage/err independently."""
    import dataclasses

    return dataclasses.replace(e)
