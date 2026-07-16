"""Cross-SDK catalogue request-URL conformance driver — Python (ADR-067 Fix B).

The REQUEST-side sibling of test_response_wire.py. Where the response suite
locks the /models PARSE seam, this locks the URL/auth-assembly seam: for a fixed
(provider, cursor), every SDK's catalogue-list path must assemble a
byte-identical {method, url, headers}.

The driver calls the SAME URL/header-assembly seam the paginate loop uses
(_build_catalogue_url + _append_cursor + _build_catalogue_headers). The
cursor_param comes from the generated catalogue_by_provider config, NOT from
inputs.json — so this exercises the generated config.

inputs.json supplies (provider, cursor) + the shared apiKey; each golden
codegen/testdata/wire/catalogue/v1/<case>.json is the expected outbound request.
This driver drops target/wire/catalogue/<case>/python.json and asserts it
value-equals the golden; codegen/test_cross_sdk_catalogue.py compares all five.
"""

from __future__ import annotations

import json
from pathlib import Path

from llmkit.catalogue import catalogue_by_provider
from llmkit.models import (
    _append_cursor,
    _build_catalogue_headers,
    _build_catalogue_url,
)
from llmkit.providers.generated.providers import PROVIDERS
from llmkit.types import Provider

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOGUE_DIR = REPO_ROOT / "codegen" / "testdata" / "wire" / "catalogue" / "v1"
ARTIFACT_ROOT = REPO_ROOT / "target" / "wire" / "catalogue"


def _write_and_assert(case: str, req_url: str, headers: dict[str, str]) -> None:
    artifact = {"method": "GET", "url": req_url, "headers": headers}
    out_dir = ARTIFACT_ROOT / case
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "python.json").write_text(json.dumps(artifact, indent=2))

    golden = json.loads((CATALOGUE_DIR / f"{case}.json").read_text())
    assert artifact == golden, f"catalogue {case} differs from shared golden"


def test_catalogue_wire() -> None:
    inputs = json.loads((CATALOGUE_DIR / "inputs.json").read_text())
    api_key = inputs["apiKey"]
    for case, spec in inputs["cases"].items():
        name = spec["provider"]
        provider = Provider(name=name, api_key=api_key)
        pcfg = PROVIDERS[name]
        cfg = catalogue_by_provider[name]

        req_url = _append_cursor(
            _build_catalogue_url(provider, pcfg, cfg.endpoint),
            cfg.cursor_param,
            spec["cursor"],
        )
        headers = _build_catalogue_headers(provider, pcfg)
        _write_and_assert(case, req_url, headers)
