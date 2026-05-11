"""HTTP transport: JSON POST, multipart upload, SSE streaming. stdlib only."""

from __future__ import annotations

import io
import json
import mimetypes
import os
import urllib.error
import urllib.request
from typing import Any, Callable

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
) -> bytes:
    """POST signed with AWS SigV4. Raises APIError on 4xx/5xx."""
    from .sigv4 import sign_sigv4

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    headers = sign_sigv4(url, body, access_key, secret_key, session_token, region, service)
    for key, value in headers.items():
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
        buf.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        buf.write(value.encode("utf-8"))
        buf.write(b"\r\n")

    if not mime_type:
        mime_type = detect_mime_type(filename)
    buf.write(f"--{boundary}\r\n".encode())
    buf.write(
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode()
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
        buf.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        buf.write(value.encode("utf-8"))
        buf.write(b"\r\n")
    for field_name, filename, mime_type, data in files:
        if not mime_type:
            mime_type = detect_mime_type(filename)
        buf.write(f"--{boundary}\r\n".encode())
        buf.write(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode()
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


def do_stream_post(
    url: str,
    body: bytes,
    headers: dict[str, str],
    stream_cfg: StreamDef,
    callback: Callable[[str], None],
    timeout: float = 600.0,
) -> Usage:
    """POST a streaming request and dispatch SSE events to `callback`. Returns accumulated usage."""
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

            if stream_cfg.done_signal and data_str == stream_cfg.done_signal:
                break
            if (
                stream_cfg.uses_event_types
                and stream_cfg.done_event
                and current_event == stream_cfg.done_event
            ):
                break

            try:
                parsed = json.loads(data_str)
            except ValueError:
                continue
            if not isinstance(parsed, dict):
                continue

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
    return usage
