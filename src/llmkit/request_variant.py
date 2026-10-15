"""Request-variant selection (ADR-096) — mirror of go/request_variant.go."""

from __future__ import annotations

from collections.abc import Sequence

from .errors import ValidationError
from .providers.generated.image_gen import (
    TEXT_ONLY,
    RequestVariant,
)


def select_request_variant(
    provider: str,
    variants: tuple[RequestVariant, ...],
    carries: Sequence[str],
) -> RequestVariant:
    """Pick the config's wire request whose carried inputs equal carries, in
    any order. A first-and-last-frame request carries two inputs. A config
    with no variant for those inputs cannot send the request, so the caller
    gets a ValidationError on "parts"."""
    for v in variants:
        if len(v.carries) == len(carries) and set(v.carries) == set(carries):
            return v
    raise ValidationError(
        field="parts", message=f"{provider} has no {'+'.join(carries)} request"
    )


def parts_carry(image_count: int, image_input: str) -> tuple[str, ...]:
    """The carried input of a request with image_count image parts:
    image_input when it holds at least one, else TextOnly. The capability
    names what its image parts are: WithReferenceImages for image,
    WithSeedFrame for video."""
    return (image_input,) if image_count > 0 else (TEXT_ONLY,)
