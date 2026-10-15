"""Request-variant selection (ADR-096) — mirror of go/request_variant.go."""

from __future__ import annotations

from .errors import ValidationError
from .providers.generated.image_gen import (
    TEXT_ONLY,
    RequestVariant,
)


def select_request_variant(
    provider: str,
    variants: tuple[RequestVariant, ...],
    image_count: int,
    image_carries: str,
) -> RequestVariant:
    """Pick the config's wire request by the inputs the request carries:
    image_carries when it holds at least one image part, else TextOnly. The
    capability names what its image parts are: WithReferenceImages for image,
    WithSeedFrame for video. A config with no variant for those inputs cannot
    send the request, so the caller gets a ValidationError on "parts"."""
    carries = image_carries if image_count > 0 else TEXT_ONLY
    for v in variants:
        if v.carries == carries:
            return v
    raise ValidationError(field="parts", message=f"{provider} has no {carries} request")
