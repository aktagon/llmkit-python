"""Unit tests for llmkit.http.do_sigv4_post — POSTs signed with AWS
SigV4 (Bedrock invoke path). Mock HTTP server checks the canonical
SigV4 header set is present and the body is forwarded verbatim, then
exercises the error path on a 4xx response."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from llmkit.errors import APIError
from llmkit.http import do_sigv4_post


class _SigV4Server:
    """Mock server that records request headers + body and serves a canned response."""

    def __init__(self, response_body: bytes, status_code: int = 200) -> None:
        self.response_body = response_body
        self.status_code = status_code
        self.received_headers: dict[str, str] = {}
        self.received_body: bytes = b""
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):  # silence noise
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                outer.received_body = self.rfile.read(length)
                outer.received_headers = {k.lower(): v for k, v in self.headers.items()}
                self.send_response(outer.status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(outer.response_body)))
                self.end_headers()
                self.wfile.write(outer.response_body)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_SigV4Server":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


# ---------- happy path ----------


def test_do_sigv4_post_attaches_sigv4_headers_and_forwards_body() -> None:
    body = b'{"prompt":"hello"}'
    response = b'{"completion":"hi there"}'
    with _SigV4Server(response) as server:
        url = f"{server.url}/model/anthropic.claude-sonnet-4-20250514-v1:0/invoke"
        got = do_sigv4_post(
            url,
            body,
            access_key="AKIA-TEST",
            secret_key="SECRET-TEST",
            session_token="",
            region="us-east-1",
            service="bedrock",
        )

    assert got == response
    # Forwarded body is verbatim.
    assert server.received_body == body

    # SigV4 canonical header set is present.
    auth = server.received_headers.get("authorization", "")
    assert auth.startswith("AWS4-HMAC-SHA256 ")
    # Authorization carries Credential / SignedHeaders / Signature parts.
    for marker in ("Credential=AKIA-TEST/", "SignedHeaders=", "Signature="):
        assert marker in auth
    # The credential scope encodes us-east-1/bedrock/aws4_request.
    assert "/us-east-1/bedrock/aws4_request" in auth

    assert "x-amz-date" in server.received_headers
    assert "x-amz-content-sha256" in server.received_headers
    # No session token requested → header omitted.
    assert "x-amz-security-token" not in server.received_headers


def test_do_sigv4_post_includes_security_token_when_provided() -> None:
    body = b'{"x":1}'
    with _SigV4Server(b"{}") as server:
        do_sigv4_post(
            url=f"{server.url}/invoke",
            body=body,
            access_key="AKIA",
            secret_key="SECRET",
            session_token="FwoGZ-temp-credentials",
            region="us-west-2",
            service="bedrock",
        )

    assert server.received_headers.get("x-amz-security-token") == "FwoGZ-temp-credentials"
    # The token is also folded into SignedHeaders so it's part of the signature.
    auth = server.received_headers["authorization"]
    assert "x-amz-security-token" in auth


# ---------- error path ----------


def test_do_sigv4_post_raises_apierror_on_4xx() -> None:
    err_body = b'{"message":"AccessDenied"}'
    with _SigV4Server(err_body, status_code=403) as server:
        with pytest.raises(APIError) as exc:
            do_sigv4_post(
                f"{server.url}/invoke",
                b"{}",
                access_key="AKIA",
                secret_key="SECRET",
                session_token="",
                region="us-east-1",
                service="bedrock",
            )

    assert exc.value.status_code == 403
    assert "AccessDenied" in exc.value.message
    assert exc.value.retryable is False  # 403 is not retryable


def test_do_sigv4_post_marks_5xx_retryable() -> None:
    with _SigV4Server(b"server error", status_code=503) as server:
        with pytest.raises(APIError) as exc:
            do_sigv4_post(
                f"{server.url}/invoke",
                b"{}",
                access_key="AKIA",
                secret_key="SECRET",
                session_token="",
                region="us-east-1",
                service="bedrock",
            )

    assert exc.value.status_code == 503
    assert exc.value.retryable is True
