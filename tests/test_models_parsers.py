"""
"""

from __future__ import annotations

from pathlib import Path

from llmkit.providers.generated.models_parsers import (
    parse_anthropic_models_response,
    parse_google_models_response,
    parse_openai_cohort_models_response,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "codegen" / "fixtures" / "models"


def _load(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


def test_parse_anthropic_fixture_records_and_metadata() -> None:
    page = parse_anthropic_models_response(_load("anthropic.json"))
    assert len(page.records) == 9
    first = page.records[0]
    assert first.id != ""
    assert first.display_name != ""
    assert first.context_window > 0
    assert first.max_output > 0


def test_parse_anthropic_round_trips_raw() -> None:
    page = parse_anthropic_models_response(_load("anthropic.json"))
    assert page.records[0].raw is not None


def test_parse_openai_cohort_fixture_records_and_no_pagination() -> None:
    page = parse_openai_cohort_models_response(_load("openai.json"))
    assert len(page.records) == 124
    assert page.next_cursor == ""
    assert page.records[0].id != ""
    assert page.records[0].created > 0


def test_parse_google_fixture_strips_models_prefix() -> None:
    page = parse_google_models_response(_load("google.json"))
    assert len(page.records) == 50
    for r in page.records:
        assert r.id != ""
        assert not r.id.startswith("models/")
    assert any(r.context_window > 0 for r in page.records)
