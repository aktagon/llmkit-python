"""HTTP transport: JSON POST, multipart upload, SSE streaming. stdlib only."""

from __future__ import annotations

import io
import json
import mimetypes
import os
import urllib.error
import urllib.request
from typing import Any, Callable


def merge_caller_headers(headers: dict[str, str], caller: dict[str, str]) -> None:
    """ADR-052: add caller-supplied custom headers (Client.add_header) that are
    NOT already present (case-insensitively). Call AFTER the SDK-set headers
    (auth, required) so those can never be clobbered — HTTP header names are
    case-insensitive, so a caller "authorization" must not shadow the
    provider's "Authorization". The caller can still add a new gateway header.
    """
    existing = {k.lower() for k in headers}
    for k, v in caller.items():
        if k.lower() not in existing:
            headers[k] = v

from .errors import APIError
from .paths import detect_mime_type, extract_int_path, extract_path
from .providers.generated.middleware import Usage
from .providers.generated.stream import StreamDef


def do_get(
    url: str,
    headers: dict[str, str],
    timeout: float = 600.0,
) -> bytes:
    """GET and return the response bytes. Raises APIError on 4xx/5xx."""
    req = urllib.request.Request(url, method="GET")
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if resp.status >= 400:
                raise APIError(
                    status_code=resp.status,
                    message=data.decode("utf-8", errors="replace"),
                    retryable=resp.status == 429 or resp.status >= 500,
                )
            return data
    except urllib.error.HTTPError as exc:
        data = exc.read()
        raise APIError(
            status_code=exc.code,
            message=data.decode("utf-8", errors="replace"),
            retryable=exc.code == 429 or exc.code >= 500,
        ) from exc


def do_post(
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float = 600.0,
) -> bytes:
    """POST JSON and return the response bytes. Raises APIError on 4xx/5xx."""
    data, status_code, resp_headers = _do_post_raw(url, body, headers, timeout)
    if status_code >= 400:
        err = APIError(
            status_code=status_code,
            message=data.decode("utf-8", errors="replace"),
            retryable=status_code == 429 or status_code >= 500,
        )
        # Attach the raw body for callers that want to parse provider error shape.
        err.type = ""  # keep dataclass shape consistent
        raise err
    return data


def _do_post_raw(
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float,
) -> tuple[bytes, int, dict[str, str]]:
    """Raw POST: returns (body, status_code, headers) without raising on HTTP errors."""
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.status, dict(resp.headers.items())
    except urllib.error.HTTPError as exc:
        return exc.read(), exc.code, dict(exc.headers.items()) if exc.headers else {}


def do_sigv4_post(
    url: str,
    body: bytes,
    access_key: str,
    secret_key: str,
    session_token: str,
    region: str,
    service: str,
    timeout: float = 600.0,
    custom_headers: dict[str, str] | None = None,
) -> bytes:
    """POST signed with AWS SigV4. Raises APIError on 4xx/5xx.

    custom_headers are caller-supplied custom headers (Client.add_header,
    ADR-052); added AFTER signing so they ride along without altering the AWS
    signature (extra unsigned headers are permitted; a gateway in front of
    Bedrock can read them)."""
    from .sigv4 import sign_sigv4

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    headers = sign_sigv4(url, body, access_key, secret_key, session_token, region, service)
    for key, value in {**(custom_headers or {}), **headers}.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        data = exc.read()
        raise APIError(
            status_code=exc.code,
            message=data.decode("utf-8", errors="replace"),
            retryable=exc.code == 429 or exc.code >= 500,
        ) from exc


def do_sigv4_get(
    url: str,
    access_key: str,
    secret_key: str,
    session_token: str,
    region: str,
    service: str,
    timeout: float = 600.0,
    custom_headers: dict[str, str] | None = None,
) -> bytes:
    """GET signed with AWS SigV4 (empty body). Raises APIError on 4xx/5xx.

    Used by the Bedrock video poll: the handle ARN is carried as one percent-
    encoded path segment so its ':' and '/' do not split into extra segments,
    and the signer canonicalizes the escaped path so the signed path equals the
    wire path. custom_headers (Client.add_header, ADR-052) are added after
    signing so they do not alter the AWS signature."""
    from .sigv4 import sign_sigv4

    req = urllib.request.Request(url, method="GET")
    headers = sign_sigv4(url, b"", access_key, secret_key, session_token, region, service, method="GET")
    for key, value in {**(custom_headers or {}), **headers}.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        data = exc.read()
        raise APIError(
            status_code=exc.code,
            message=data.decode("utf-8", errors="replace"),
            retryable=exc.code == 429 or exc.code >= 500,
        ) from exc


def _escape_quotes(value: str) -> str:
    """Mirror Go stdlib mime/multipart escapeQuotes and additionally strip
    CR/LF: a quote or newline in a caller-controlled field name or filename
    must not break out of the Content-Disposition part header
    (HANDOFF-036 A2)."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")


def do_multipart_post(
    url: str,
    field_name: str,
    filename: str,
    data: bytes,
    fields: dict[str, str],
    headers: dict[str, str],
    timeout: float = 600.0,
    mime_type: str = "",
) -> tuple[bytes, int]:
    """POST multipart/form-data. Returns (body, status_code); does NOT raise on 4xx/5xx.

    If ``mime_type`` is empty, Content-Type for the file part is derived
    from the filename extension via :func:`detect_mime_type`.
    """
    boundary = "----llmkit-python-" + os.urandom(16).hex()
    buf = io.BytesIO()
    for key, value in fields.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(f'Content-Disposition: form-data; name="{_escape_quotes(key)}"\r\n\r\n'.encode())
        buf.write(value.encode("utf-8"))
        buf.write(b"\r\n")

    if not mime_type:
        mime_type = detect_mime_type(filename)
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(
        f'Content-Disposition: form-data; name="{_escape_quotes(field_name)}"; filename="{_escape_quotes(filename)}"\r\n'.encode()
    )
    buf.write(f"Content-Type: {mime_type}\r\n\r\n".encode())
    buf.write(data)
    buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())

    body = buf.getvalue()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.status
    except urllib.error.HTTPError as exc:
        return exc.read(), exc.code


def do_multipart_post_multi(
    url: str,
    files: list[tuple[str, str, str, bytes]],
    fields: dict[str, str],
    headers: dict[str, str],
    timeout: float = 600.0,
) -> tuple[bytes, int]:
    """POST multipart/form-data with one or more file parts plus zero-or-more
    plain string fields. ``files`` items are ``(field_name, filename, mime_type, data)``;
    field_name may end in "[]" when the API expects an array (e.g. OpenAI's "image[]").
    Returns ``(body, status_code)``; does NOT raise on 4xx/5xx.
    """
    boundary = "----llmkit-python-" + os.urandom(16).hex()
    buf = io.BytesIO()
    for key, value in fields.items():
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(f'Content-Disposition: form-data; name="{_escape_quotes(key)}"\r\n\r\n'.encode())
        buf.write(value.encode("utf-8"))
        buf.write(b"\r\n")
    for field_name, filename, mime_type, data in files:
        if not mime_type:
            mime_type = detect_mime_type(filename)
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(
            f'Content-Disposition: form-data; name="{_escape_quotes(field_name)}"; filename="{_escape_quotes(filename)}"\r\n'.encode()
        )
        buf.write(f"Content-Type: {mime_type}\r\n\r\n".encode())
        buf.write(data)
        buf.write(b"\r\n")
    buf.write(f"--{boundary}--\r\n".encode())

    body = buf.getvalue()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.status
    except urllib.error.HTTPError as exc:
        return exc.read(), exc.code


def _parse_stream_finish_path(p: str) -> tuple[str, str]:
    """Split ADR-013 stream-finish locator into (event_name, json_path)."""
    if not p:
        return "", ""
    idx = p.find(":")
    if idx >= 0:
        return p[:idx], p[idx + 1 :]
    return "", p


def do_stream_post(
    url: str,
    body: bytes,
    headers: dict[str, str],
    stream_cfg: StreamDef,
    callback: Callable[[str], None],
    timeout: float = 600.0,
    finish_reason_path: str = "",
) -> tuple[Usage, str]:
    """POST a streaming request and dispatch SSE events to `callback`.

    Returns ``(usage, finish_reason)``. ``finish_reason`` follows ADR-013:
    captured from the parsed event/data body via ``finish_reason_path``
    (``event_name:json.path`` form or bare ``json.path``); empty when the
    provider declares no stream-time path or no signal arrived.
    """
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        data = exc.read()
        raise APIError(
            status_code=exc.code,
            message=data.decode("utf-8", errors="replace"),
            retryable=exc.code == 429 or exc.code >= 500,
        ) from exc

    usage = Usage()
    finish_event, finish_json_path = _parse_stream_finish_path(finish_reason_path)
    finish_reason = ""
    current_event = ""
    with resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")

            if line.startswith("event: "):
                current_event = line[len("event: "):]
                continue

            if not line.startswith("data: "):
                continue

            data_str = line[len("data: "):]

            # Data-level done sentinel (e.g., OpenAI "[DONE]") is literal,
            # not JSON — bail before parsing.
            if stream_cfg.done_signal and data_str == stream_cfg.done_signal:
                break

            try:
                parsed = json.loads(data_str)
            except ValueError:
                if (
                    stream_cfg.uses_event_types
                    and stream_cfg.done_event
                    and current_event == stream_cfg.done_event
                ):
                    break
                continue
            if not isinstance(parsed, dict):
                if (
                    stream_cfg.uses_event_types
                    and stream_cfg.done_event
                    and current_event == stream_cfg.done_event
                ):
                    break
                continue

            # ADR-013: capture finish-reason BEFORE the event-level done
            # break — Anthropic carries stop_reason on the message_stop
            # event body and dropping the parse would discard it.
            if finish_json_path and (finish_event == "" or finish_event == current_event):
                value = extract_path(parsed, finish_json_path)
                if value and value != "FINISH_REASON_UNSPECIFIED":
                    finish_reason = value

            if (
                stream_cfg.uses_event_types
                and stream_cfg.done_event
                and current_event == stream_cfg.done_event
            ):
                break

            if stream_cfg.uses_event_types:
                if current_event == stream_cfg.content_event:
                    text = extract_path(parsed, stream_cfg.delta_text_path)
                    if text:
                        callback(text)
                if current_event == stream_cfg.usage_event and stream_cfg.usage_output_path:
                    usage.output = extract_int_path(parsed, stream_cfg.usage_output_path)
            else:
                text = extract_path(parsed, stream_cfg.delta_text_path)
                if text:
                    callback(text)
                if stream_cfg.usage_input_path:
                    value = extract_int_path(parsed, stream_cfg.usage_input_path)
                    if value > 0:
                        usage.input = value
                if stream_cfg.usage_output_path:
                    value = extract_int_path(parsed, stream_cfg.usage_output_path)
                    if value > 0:
                        usage.output = value

            current_event = ""
    return usage, finish_reason
