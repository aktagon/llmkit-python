"""AWS Signature Version 4 signing. stdlib only (hashlib, hmac, urllib)."""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
from urllib.parse import urlparse, parse_qsl


def sign_sigv4(
    url: str,
    body: bytes,
    access_key: str,
    secret_key: str,
    session_token: str,
    region: str,
    service: str,
    method: str = "POST",
) -> dict[str, str]:
    """Return the SigV4 headers for a request, matching Go sigv4.go output.

    ``method`` defaults to POST (the chat path); the Bedrock video poll signs a
    GET with an empty body. The canonical path is the ESCAPED path (what goes on
    the wire) — mirroring go canonicalURI — so a percent-encoded path segment
    (e.g. the GetAsyncInvoke ARN encoded as one segment) canonicalizes to the
    same bytes the server receives. A no-op for the chat Converse path: its model
    id's ':' is not escaped, so the escaped path equals the decoded path there.
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    datestamp = now.strftime("%Y%m%d")
    amzdate = now.strftime("%Y%m%dT%H%M%SZ")

    parsed = urlparse(url)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    # urlsplit/urlparse leaves percent-encoding intact in .path, so .path is the
    # escaped (wire) path — exactly what AWS canonicalizes.
    path = parsed.path or "/"

    payload_hash = _sha256_hex(body)

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Host": host,
        "X-Amz-Date": amzdate,
        "X-Amz-Content-Sha256": payload_hash,
    }
    if session_token:
        headers["X-Amz-Security-Token"] = session_token

    signed_headers, canonical_headers = _build_canonical_headers(headers, host)

    canonical_request = "\n".join(
        [
            method,
            path,
            _canonical_query_string(parsed.query),
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )

    credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amzdate,
            credential_scope,
            _sha256_hex(canonical_request.encode("utf-8")),
        ]
    )

    signing_key = _derive_signing_key(secret_key, datestamp, region, service)
    signature = _hmac_sha256(signing_key, string_to_sign.encode("utf-8")).hex()

    headers["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return headers


def _canonical_query_string(query: str) -> str:
    if not query:
        return ""
    pairs = parse_qsl(query, keep_blank_values=True)
    pairs.sort(key=lambda kv: kv[0])
    return "&".join(f"{k}={v}" for k, v in pairs)


def _build_canonical_headers(headers: dict[str, str], host: str) -> tuple[str, str]:
    selected: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower == "host" or lower == "content-type" or lower.startswith("x-amz-"):
            selected[lower] = value.strip()
    selected.setdefault("host", host)

    keys = sorted(selected.keys())
    canonical = "".join(f"{k}:{selected[k]}\n" for k in keys)
    signed = ";".join(keys)
    return signed, canonical


def _derive_signing_key(secret_key: str, datestamp: str, region: str, service: str) -> bytes:
    k_date = _hmac_sha256(("AWS4" + secret_key).encode("utf-8"), datestamp.encode("utf-8"))
    k_region = _hmac_sha256(k_date, region.encode("utf-8"))
    k_service = _hmac_sha256(k_region, service.encode("utf-8"))
    return _hmac_sha256(k_service, b"aws4_request")


def _hmac_sha256(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
