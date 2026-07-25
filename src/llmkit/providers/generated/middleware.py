# Code generated — DO NOT EDIT.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


@dataclass
class Usage:
    """Token consumption metrics for an LLM call.

    A dimension is either reported — carrying a value that may legitimately be
    zero — or not reported at all. The two are different claims: a provider
    that says it used no cached tokens and a provider that never mentions
    caching are not the same fact, and neither is a zero.
    """

    input: int | None = None
    output: int | None = None
    cache_write: int | None = None
    cache_read: int | None = None
    reasoning: int | None = None
    # cost is the provider-reported request cost in USD (ADR-027). Not a TokenDimension — a distinct monetary field. Only OpenRouter (the request must opt in with usage: {include: true}) and xAI report it. Providers whose usageCostPath is empty never report cost, and the field is then ABSENT, not 0.0 — an unreported cost is not a free request (ADR-081 AVAIL-007).
    cost: float | None = None


class MiddlewarePhase(str, Enum):
    PRE = "pre"
    POST = "post"


class MiddlewareOp(str, Enum):
    LLM_REQUEST = "llm_request"
    TOOL_CALL = "tool_call"
    CACHE_CREATE = "cache_create"
    UPLOAD = "upload"
    BATCH_SUBMIT = "batch_submit"
    IMAGE_GENERATION = "image_generation"
    MUSIC_GENERATION = "music_generation"
    VIDEO_GENERATION = "video_generation"
    MODELS_LIST = "models_list"
    SPEECH_GENERATION = "speech_generation"
    TRANSCRIPTION = "transcription"


@dataclass
class Event:
    """Observation and veto carrier passed to middleware hooks."""
    # Always set.
    op: MiddlewareOp = MiddlewareOp.LLM_REQUEST
    # Always set. Internal-only (drives pre/post dispatch); not an OTEL attribute.
    phase: MiddlewarePhase = MiddlewarePhase.PRE
    # Always set.
    provider: str = ""
    # Always set.
    model: str = ""
    # Only set when Op=tool_call. Internal-only.
    tool: str = ""
    # Only set when Op=tool_call, Phase=pre. Mutation by middleware is observed by the tool. Internal-only.
    args: dict[str, Any] = field(default_factory=dict)
    # Only set when Op=tool_call, Phase=post. Internal-only.
    result: str = ""
    # Set for Op=llm_request, Phase=post. Expanded to gen_ai.usage.* via otelUsageAttribute on each TokenDimension, not a single attribute. Its optional dimensions are SHARED with the response the middleware observes (ADR-081): read them, do not write through them — mutating one rewrites what the caller receives.
    usage: Usage | None = None
    # Set in Phase=post when the operation failed. Human-readable; telemetry never re-parses it (ADR-071).
    err: str | None = None
    # Set in Phase=post when the operation failed: one of api_error | validation_error | error, stamped structurally from the typed error at the erasure seam (ADR-071). The OTLP builder reads this verbatim.
    err_type: str = ""
    # Set in Phase=post. Internal-only (maps to span duration, not a gen_ai attribute).
    duration: float = 0.0


# MiddlewareFn is the user-supplied pre/post hook. Pre-phase non-None return
# vetoes the operation; post-phase return value is ignored.
MiddlewareFn = Callable[[Event], Exception | None]
