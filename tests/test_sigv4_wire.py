"""SigV4 canonical-request wire driver (CR-002): sign the two production-shaped
Bedrock requests with an injected clock and assert the canonical request,
string-to-sign, and Authorization header byte-identically against the shared
golden at ``codegen/testdata/wire/sigv4/v1/<fixture>.json``. The golden is
minted from botocore (external authority — see the PROVENANCE.md beside the
goldens), and the same fixed inputs are hard-coded in every SDK's driver; the
cross-SDK comparator cross-checks the per-SDK artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llmkit.sigv4 import _sign_parts

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "codegen" / "testdata" / "wire" / "sigv4" / "v1"
ARTIFACT_ROOT = REPO_ROOT / "target" / "wire" / "sigv4"

# The frozen signing clock shared by every SDK driver: 2026-07-18T00:00:00Z.
SIGV4_WIRE_NOW = datetime(2026, 7, 18, 0, 0, 0, tzinfo=timezone.utc)

ACCESS_KEY = "AKIDEXAMPLE"
SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"  # AWS docs canonical example #gitleaks:allow
SESSION_TOKEN = "IQoJb3JpZ2luX2VjEXAMPLETOKEN"  # AWS docs example creds #gitleaks:allow


def _assert_sigv4_golden(fixture: str, canonical_request: str, string_to_sign: str, authorization: str) -> None:
    artifact = {
        "canonicalRequest": canonical_request,
        "stringToSign": string_to_sign,
        "authorization": authorization,
    }
    out_dir = ARTIFACT_ROOT / fixture
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "python.json").write_text(json.dumps(artifact, indent=2))

    golden = json.loads((GOLDEN_DIR / f"{fixture}.json").read_text())
    assert artifact["canonicalRequest"] == golden["canonicalRequest"]
    assert artifact["stringToSign"] == golden["stringToSign"]
    assert artifact["authorization"] == golden["authorization"]


def test_sigv4_wire_chat_post_matches_golden() -> None:
    # Mirrors do_sigv4_post for the Bedrock Converse chat path: POST,
    # Content-Type signed, session token present, model id ':' literal in the
    # path.
    _, canonical_request, string_to_sign, authorization = _sign_parts(
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1:0/converse",
        b'{"messages":[{"role":"user","content":[{"text":"Hello, Bedrock"}]}]}',
        ACCESS_KEY,
        SECRET_KEY,
        SESSION_TOKEN,
        "us-east-1",
        "bedrock",
        "POST",
        "application/json",
        SIGV4_WIRE_NOW,
    )
    _assert_sigv4_golden("sigv4-chat-post", canonical_request, string_to_sign, authorization)


def test_sigv4_wire_poll_get_matches_golden() -> None:
    # Mirrors do_sigv4_get for the Bedrock async-invoke poll: GET, empty body
    # (empty-string SHA-256 payload hash), no Content-Type, no session token,
    # and the invocation ARN percent-encoded as ONE path segment ('/' -> %2F,
    # ':' literal) so the signed path equals the wire path.
    _, canonical_request, string_to_sign, authorization = _sign_parts(
        "https://bedrock-runtime.us-west-2.amazonaws.com/async-invoke/arn:aws:bedrock:us-west-2:123456789012:async-invoke%2Fabc123xyz",
        b"",
        ACCESS_KEY,
        SECRET_KEY,
        "",
        "us-west-2",
        "bedrock",
        "GET",
        "",
        SIGV4_WIRE_NOW,
    )
    _assert_sigv4_golden("sigv4-poll-get", canonical_request, string_to_sign, authorization)
