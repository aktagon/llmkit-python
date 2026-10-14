"""Request-variant selection (ADR-096) — mirror of go/request_variant.go."""

from __future__ import annotations

from .errors import ValidationError
from .providers.generated.image_gen import (
    TEXT_ONLY,
    WITH_REFERENCE_IMAGES,
    RequestVariant,
)


def select_request_variant(
    provider: str, variants: tuple[RequestVariant, ...], image_count: int
) -> RequestVariant:
    """Pick the config's wire request by the inputs the request carries:
    WithReferenceImages when it holds at least one image part, else TextOnly.
    A config with no variant for those inputs cannot send the request, so the
    caller gets a ValidationError on "parts"."""
    carries = WITH_REFERENCE_IMAGES if image_count > 0 else TEXT_ONLY
    for v in variants:
        if v.carries == carries:
            return v
    raise ValidationError(field="parts", message=f"{provider} has no {carries} request")
