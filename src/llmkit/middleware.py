"""Handwritten middleware helpers: fire_pre, fire_post, set_event_error, resolve_model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .errors import APIError, MiddlewareVetoError, ValidationError
from .providers.generated.middleware import Event, MiddlewareFn, MiddlewarePhase
from .providers.generated.providers import ProviderSpec

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


def set_event_error(ev: Event, exc: BaseException) -> None:
    """Erase a typed error onto a post-phase Event: ``err`` (the human string)
    and ``err_type`` (the ADR-071 structural kind the OTLP builder reads
    verbatim). Classification happens here — the one seam where the typed
    exception still exists — never by re-parsing the message string."""
    ev.err = str(exc)
    if isinstance(exc, APIError):
        ev.err_type = "api_error"
    elif isinstance(exc, ValidationError):
        ev.err_type = "validation_error"
    else:
        # Transport, decoding, veto, unknown: the stable catch-all kind.
        ev.err_type = "error"


def resolve_model(provider: Provider, cfg: ProviderSpec) -> str:
    """Return the caller-specified model or the provider's curated default.

    Local daemons declare no default — what a daemon serves is runtime
    inventory, not a registry fact (ADR-031). Both empty raises
    immediately instead of guessing a model the daemon may not have."""
    if provider.model:
        return provider.model
    if not cfg.default_model:
        raise ValidationError(
            field="model",
            message=(
                f'no model chosen and "{provider.name}" declares no default; '
                "pick one (models.live() lists what the daemon serves)"
            ),
        )
    return cfg.default_model


def _copy_event(e: Event) -> Event:
    """Shallow copy so each phase can set phase/usage/err independently."""
    import dataclasses

    return dataclasses.replace(e)
