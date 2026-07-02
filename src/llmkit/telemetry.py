"""Opt-in, OTEL-aligned telemetry (ADR-054).

Attach a :class:`Telemetry` to a client with :func:`with_telemetry` to export an
OTEL GenAI-aligned span over OTLP/HTTP (JSON) on every provider call that fires
middleware — success and rejection alike. Off unless attached; an empty endpoint
is a :class:`ValidationError` (the honest-contract lineage: no enabled-but-no-sink
state). A sibling of the ADR-052 base-URL / custom-header runtime overrides — a
handwritten config value, not modelled in the ontology.

The pure :func:`build_otlp_traces` builder is asserted value-identical across all
four SDKs against the shared goldens at
``codegen/testdata/wire/telemetry/v1/`` (TEL-011).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass

from .errors import ValidationError
from .middleware import _copy_event
from .providers.generated.middleware import Event, MiddlewareFn, MiddlewarePhase
from .providers.generated.telemetry import (
    OTEL_ATTR_ERR,
    OTEL_ATTR_MODEL,
    OTEL_ATTR_OP,
    OTEL_ATTR_PROVIDER,
    OTEL_USAGE_INPUT,
    OTEL_USAGE_OUTPUT,
    TELEMETRY_OPERATION_NAME,
    TELEMETRY_SEMCONV_VERSION,
    TELEMETRY_TRACES_PATH,
)

# Builder namespaces on Client that carry a middleware seam. Speech and
# Transcription have no middleware runtime yet (ADR-049/051) and are covered
# when that seam lands — the same deferral the Go reference documents.
_TELEMETRY_BUILDERS = ("text", "image", "music", "video", "agent", "upload")

# The export POST is bounded so a slow/hung collector never stalls a call.
_EXPORT_TIMEOUT_SECONDS = 5.0


@dataclass
class Telemetry:
    """Opt-in observability config (ADR-054).

    ``endpoint`` is the OTLP/HTTP collector base URL (mandatory); the exporter
    POSTs proto3-JSON to ``endpoint`` + ``/v1/traces``. ``headers`` are added to
    every export POST (e.g. authorization). ``capture_content`` gates tier-2
    message payloads (default False for privacy); the middleware Event does not
    carry payloads yet, so this reserves the semantics for a deferred follow-up.
    """

    endpoint: str
    headers: dict[str, str] | None = None
    capture_content: bool = False


def with_telemetry(client, telemetry: Telemetry):
    """Enable opt-in telemetry on ``client``; returns the same client for chaining.

    Mirrors the Go ``Client.WithTelemetry``: the exporter rides the middleware
    seam, so every capability path that fires middleware emits one OTEL span on
    the post phase. Empty endpoint is fail-loud — a :class:`ValidationError`
    naming ``telemetry.endpoint`` is raised immediately (Python validates at
    attach time rather than deferring to first call).
    """
    if not telemetry.endpoint:
        raise ValidationError(
            field="telemetry.endpoint",
            message="endpoint is required when telemetry is enabled",
        )
    mw = make_telemetry_middleware(telemetry)
    # Inject into every builder prototype that carries a middleware seam. Chain
    # clones copy the prototype's slice reference, so this reaches every call.
    for name in _TELEMETRY_BUILDERS:
        builder = getattr(client, name, None)
        if builder is not None and hasattr(builder, "_middleware"):
            builder._middleware = [*builder._middleware, mw]
    return client


def make_telemetry_middleware(telemetry: Telemetry) -> MiddlewareFn:
    """Build the export hook. The post phase exports fail-open: a telemetry
    failure never propagates or blocks the call."""

    def _hook(event: Event) -> Exception | None:
        if event.phase != MiddlewarePhase.POST:
            return None
        # Fire-and-forget on a daemon thread (FU-2): a slow/hung collector must
        # never block the caller, and daemon=True means it never holds up
        # interpreter exit. _export is itself fail-open. One thread per export
        # for now; a shared worker + bounded channel is the FU-6 upgrade.
        # Snapshot the event first: fire_post shares one Event across all post
        # hooks, so a later hook could mutate it while this thread reads it (Go
        # copies by value and Rust clones scalars for the same reason).
        threading.Thread(
            target=_export, args=(telemetry, _copy_event(event)), daemon=True
        ).start()
        return None

    return _hook


def _export(telemetry: Telemetry, event: Event) -> None:
    """Serialize the post-phase Event to an OTLP traces payload and POST it.

    Fail-open: every error (bad endpoint, timeout, malformed value) is swallowed.
    """
    try:
        operation_name = TELEMETRY_OPERATION_NAME.get(event.op, event.op.value)
        input_tokens = event.usage.input if event.usage is not None else 0
        output_tokens = event.usage.output if event.usage is not None else 0
        error_type = _error_type(event)
        now = str(time.time_ns())
        payload = build_otlp_traces(
            operation_name,
            event.provider,
            event.model,
            input_tokens,
            output_tokens,
            error_type,
            os.urandom(16).hex(),
            os.urandom(8).hex(),
            now,
            now,
        )

        headers = {"Content-Type": "application/json"}
        headers.update(telemetry.headers or {})
        url = telemetry.endpoint.rstrip("/") + TELEMETRY_TRACES_PATH

        req = urllib.request.Request(url, data=payload, method="POST")
        for key, value in headers.items():
            req.add_header(key, value)
        with urllib.request.urlopen(req, timeout=_EXPORT_TIMEOUT_SECONDS) as resp:
            resp.read()
    except Exception:
        # Fail-open: telemetry never affects the caller.
        pass


def _error_type(event: Event) -> str:
    """Map a post-phase Event to a stable OTEL ``error.type`` value.

    The Python middleware Event carries the error as a string (not a typed
    exception), so classification collapses to a single stable token; the exact
    provider error code (e.g. ``rate_limit_exceeded``) is not recoverable here.
    """
    return "error" if event.err else ""


def build_otlp_traces(
    operation_name: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    error_type: str,
    trace_id: str,
    span_id: str,
    start_nano: str,
    end_nano: str,
) -> bytes:
    """Pure, deterministic OTLP/HTTP traces payload builder.

    Given the call's primitives plus injectable span identity + timing, returns
    the exact JSON bytes the exporter POSTs. int64 fields (times, token counts)
    render as strings and traceId/spanId as hex, per the OTLP/JSON spec — the
    parity fixtures call this with fixed inputs so all four SDKs are asserted
    value-identical.
    """
    attributes = [
        {"key": OTEL_ATTR_OP, "value": {"stringValue": operation_name}},
        {"key": OTEL_ATTR_PROVIDER, "value": {"stringValue": provider}},
        {"key": OTEL_ATTR_MODEL, "value": {"stringValue": model}},
    ]
    if input_tokens > 0:
        attributes.append(
            {"key": OTEL_USAGE_INPUT, "value": {"intValue": str(input_tokens)}}
        )
    if output_tokens > 0:
        attributes.append(
            {"key": OTEL_USAGE_OUTPUT, "value": {"intValue": str(output_tokens)}}
        )
    if error_type != "":
        attributes.append(
            {"key": OTEL_ATTR_ERR, "value": {"stringValue": error_type}}
        )

    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": f"{operation_name} {model}",
        "kind": 3,
        "startTimeUnixNano": start_nano,
        "endTimeUnixNano": end_nano,
        "attributes": attributes,
    }
    if error_type != "":
        span["status"] = {"code": 2}

    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "llmkit"}}
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {
                            "name": "llmkit",
                            "version": TELEMETRY_SEMCONV_VERSION,
                        },
                        "spans": [span],
                    }
                ],
            }
        ]
    }
    return json.dumps(payload).encode("utf-8")
