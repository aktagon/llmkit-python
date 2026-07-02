"""











"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass

from .errors import ValidationError
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

#
#
#
_TELEMETRY_BUILDERS = ("text", "image", "music", "video", "agent", "upload")

#
_EXPORT_TIMEOUT_SECONDS = 5.0


@dataclass
class Telemetry:
    """






"""

    endpoint: str
    headers: dict[str, str] | None = None
    capture_content: bool = False


def with_telemetry(client, telemetry: Telemetry):
    """






"""
    if not telemetry.endpoint:
        raise ValidationError(
            field="telemetry.endpoint",
            message="endpoint is required when telemetry is enabled",
        )
    mw = make_telemetry_middleware(telemetry)
    #
    #
    for name in _TELEMETRY_BUILDERS:
        builder = getattr(client, name, None)
        if builder is not None and hasattr(builder, "_middleware"):
            builder._middleware = [*builder._middleware, mw]
    return client


def make_telemetry_middleware(telemetry: Telemetry) -> MiddlewareFn:
    """
"""

    def _hook(event: Event) -> Exception | None:
        if event.phase != MiddlewarePhase.POST:
            return None
        _export(telemetry, event)
        return None

    return _hook


def _export(telemetry: Telemetry, event: Event) -> None:
    """


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
        #
        pass


def _error_type(event: Event) -> str:
    """




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
    """






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
