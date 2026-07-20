"""















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

#
#
#
_TELEMETRY_BUILDERS = ("text", "image", "music", "video", "agent", "upload")

#
#
_EXPORT_TIMEOUT_SECONDS = 5.0


@dataclass
class Telemetry:
    """







"""

    export: Callable[[bytes], None]
    capture_content: bool = False


def add_telemetry(client, telemetry: Telemetry):
    """






"""
    if not callable(getattr(telemetry, "export", None)):
        raise ValidationError(
            field="telemetry.export",
            message="export is required when telemetry is enabled (use http_export for a batteries POST)",
        )
    mw = make_telemetry_middleware(telemetry)
    #
    #
    for name in _TELEMETRY_BUILDERS:
        builder = getattr(client, name, None)
        if builder is not None and hasattr(builder, "_middleware"):
            builder._middleware = [*builder._middleware, mw]
    #
    #
    if hasattr(client, "_middleware"):
        client._middleware = [*client._middleware, mw]
    return client


def make_telemetry_middleware(telemetry: Telemetry) -> MiddlewareFn:
    """

"""

    def _hook(event: Event) -> Exception | None:
        if event.phase != MiddlewarePhase.POST:
            return None
        try:
            telemetry.export(_build_payload(event))
        except Exception:
            #
            pass
        return None

    return _hook


def _build_payload(event: Event) -> bytes:
    """"""
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
    """



"""
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
    """








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
            #
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
