"""Handwritten middleware helpers: fire_pre, fire_post, resolve_model."""

from __future__ import annotations

from .errors import MiddlewareVetoError
from .providers.generated.middleware import Event, MiddlewareFn, MiddlewarePhase
from .providers.generated.providers import ProviderConfig


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


def resolve_model(provider_model: str, cfg: ProviderConfig) -> str:
    """Return caller-specified model or provider default."""
    return provider_model or cfg.default_model


def _copy_event(e: Event) -> Event:
    """Shallow copy so each phase can set phase/usage/err independently."""
    import dataclasses

    return dataclasses.replace(e)
