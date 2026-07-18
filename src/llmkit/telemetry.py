"""Opt-in, OTEL-aligned telemetry (ADR-059, superseding ADR-054's transport half).

Attach a :class:`Telemetry` to a client with :func:`add_telemetry`: on every
provider call that fires middleware — success and rejection alike — llmkit builds
an OTEL GenAI-aligned OTLP span (proto3 JSON) and hands the finished bytes to the
``export`` callback. llmkit does no telemetry network I/O and spawns no thread;
what the callback does — enqueue into an OTEL SDK, POST, drop — plus all batching,
backpressure, and shutdown is the caller's concern. Off unless attached; a missing
``export`` is a :class:`ValidationError` (the honest-contract lineage: no
enabled-but-no-sink state). Use :func:`http_export` for a batteries POST. A sibling
of the ADR-052 base-URL / custom-header runtime overrides — a handwritten config
value, not modelled in the ontology.

The pure :func:`build_otlp_traces` builder is asserted value-identical across all
four SDKs against the shared goldens at
``codegen/testdata/wire/telemetry/v1/`` (TEL-011).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from .errors import ValidationError
from .providers.generated.middleware import Event, MiddlewareFn, MiddlewarePhase
from .providers.generated.telemetry import (
    OTEL_ATTR_ERR_TYPE,
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

# The batteries http_export POST is bounded so a slow/hung collector never
# stalls a call for longer than this.
_EXPORT_TIMEOUT_SECONDS = 5.0


@dataclass
class Telemetry:
    """Opt-in observability config (ADR-059).

    ``export`` receives the finished OTLP/HTTP proto3-JSON bytes for one span,
    called synchronously on the post phase (mandatory). Use :func:`http_export`
    for the batteries POST, or supply your own to bridge into an existing OTEL
    stack. ``capture_content`` gates tier-2 message payloads (default False for
    privacy); the middleware Event does not carry payloads yet, so this reserves
    the semantics for a deferred follow-up.
    """

    export: Callable[[bytes], None]
    capture_content: bool = False


def add_telemetry(client, telemetry: Telemetry):
    """Enable opt-in telemetry on ``client``; returns the same client for chaining.

    Mirrors the Go ``Client.AddTelemetry``: the builder rides the middleware
    seam, so every capability path that fires middleware emits one OTEL span on
    the post phase. A missing ``export`` callback is fail-loud — a
    :class:`ValidationError` naming ``telemetry.export`` is raised immediately
    (Python validates at attach time rather than deferring to first call).
    """
    if not callable(getattr(telemetry, "export", None)):
        raise ValidationError(
            field="telemetry.export",
            message="export is required when telemetry is enabled (use http_export for a batteries POST)",
        )
    mw = make_telemetry_middleware(telemetry)
    # Inject into every builder prototype that carries a middleware seam. Chain
    # clones copy the prototype's slice reference, so this reaches every call.
    for name in _TELEMETRY_BUILDERS:
        builder = getattr(client, name, None)
        if builder is not None and hasattr(builder, "_middleware"):
            builder._middleware = [*builder._middleware, mw]
    # Client-scoped seam: the models/catalogue runtime has no per-builder
    # middleware chain, so it fires the client list (HANDOFF-036 A3).
    if hasattr(client, "_middleware"):
        client._middleware = [*client._middleware, mw]
    return client


def make_telemetry_middleware(telemetry: Telemetry) -> MiddlewareFn:
    """Build the export hook. The post phase builds the OTLP payload and calls
    ``export`` SYNCHRONOUSLY (ADR-059) — no thread. Fail-open: a raising callback
    is swallowed so telemetry never propagates or blocks the call."""

    def _hook(event: Event) -> Exception | None:
        if event.phase != MiddlewarePhase.POST:
            return None
        try:
            telemetry.export(_build_payload(event))
        except Exception:
            # Fail-open: telemetry never affects the caller.
            pass
        return None

    return _hook


def _build_payload(event: Event) -> bytes:
    """Production wrapper: stamp span identity + timing, then render the Event."""
    now = str(time.time_ns())
    return _build_payload_at(
        event, os.urandom(16).hex(), os.urandom(8).hex(), now, now
    )


def _build_payload_at(
    event: Event,
    trace_id: str,
    span_id: str,
    start_nano: str,
    end_nano: str,
) -> bytes:
    """Pure event-level payload builder: render a post-phase Event to OTLP
    traces bytes with injected span identity + timing (the telemetry-error
    golden drives it end-to-end). ``error.type`` is ``event.err_type``
    verbatim — stamped structurally at the erasure seam (ADR-071), never
    re-derived here from the message string."""
    operation_name = TELEMETRY_OPERATION_NAME.get(event.op, event.op.value)
    input_tokens = event.usage.input if event.usage is not None else 0
    output_tokens = event.usage.output if event.usage is not None else 0
    return build_otlp_traces(
        operation_name,
        event.provider,
        event.model,
        input_tokens,
        output_tokens,
        event.err_type,
        trace_id,
        span_id,
        start_nano,
        end_nano,
    )


def http_export(
    endpoint: str, headers: dict[str, str] | None = None
) -> Callable[[bytes], None]:
    """Return an ``export`` callback that POSTs each OTLP payload to
    ``endpoint`` + ``/v1/traces`` with a bounded timeout, fail-open (every
    network error is swallowed). It spawns no background worker and needs no
    close.

    Low-volume only: the POST is SYNCHRONOUS on the request path, so a slow or
    hung collector adds up to ``_EXPORT_TIMEOUT_SECONDS`` of latency to the call.
    For high volume, hand your own callback that enqueues into your OTEL SDK's
    batch processor instead.
    """
    url = endpoint.rstrip("/") + TELEMETRY_TRACES_PATH
    base_headers = {"Content-Type": "application/json"}
    if headers:
        base_headers.update(headers)

    def _post(payload: bytes) -> None:
        try:
            req = urllib.request.Request(url, data=payload, method="POST")
            for key, value in base_headers.items():
                req.add_header(key, value)
            with urllib.request.urlopen(
                req, timeout=_EXPORT_TIMEOUT_SECONDS
            ) as resp:
                resp.read()
        except Exception:
            # Fail-open: telemetry never affects the caller.
            pass

    return _post


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
            {"key": OTEL_ATTR_ERR_TYPE, "value": {"stringValue": error_type}}
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
