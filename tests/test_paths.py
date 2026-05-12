"""Unit tests for llmkit.paths — pure-function helpers used by request
shaping, response parsing, and schema massaging. All tests use real
domain shapes (image MIME paths, OpenAI-style JSON schemas, data URIs)
rather than placeholder strings."""

from __future__ import annotations

import pytest

from llmkit.paths import (
    contains_value,
    detect_mime_type,
    extract_int_path,
    extract_path,
    merge_into_parent,
    parse_data_uri,
    remove_additional_properties,
    set_additional_properties_false,
    set_nested_field,
)


# ---------- detect_mime_type ----------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("report.pdf", "application/pdf"),
        ("data.json", "application/json"),
        ("notes.txt", "text/plain"),
        ("README.md", "text/markdown"),
        ("rows.csv", "text/csv"),
        ("photo.png", "image/png"),
        ("photo.jpg", "image/jpeg"),
        ("photo.jpeg", "image/jpeg"),
        ("animated.gif", "image/gif"),
        ("photo.webp", "image/webp"),
        # Case insensitivity matters: macOS Finder defaults to lowercase
        # but iOS Camera Roll exports uppercase JPG.
        ("PHOTO.JPG", "image/jpeg"),
        # Path with directories.
        ("/some/dir/report.pdf", "application/pdf"),
        # Unknown extensions fall back to octet-stream.
        ("binary.bin", "application/octet-stream"),
        ("noextension", "application/octet-stream"),
    ],
)
def test_detect_mime_type(filename: str, expected: str) -> None:
    assert detect_mime_type(filename) == expected


# ---------- parse_data_uri ----------


def test_parse_data_uri_base64_png() -> None:
    uri = "data:image/png;base64,iVBORw0KGgo="
    mime, payload = parse_data_uri(uri)
    assert mime == "image/png"
    assert payload == "iVBORw0KGgo="


def test_parse_data_uri_no_base64_marker() -> None:
    # data: URIs without base64 still parse — mime is the literal part
    # before the comma.
    uri = "data:text/plain,hello"
    mime, payload = parse_data_uri(uri)
    assert mime == "text/plain"
    assert payload == "hello"


def test_parse_data_uri_non_data_uri_returned_verbatim() -> None:
    uri = "https://example.com/image.png"
    mime, payload = parse_data_uri(uri)
    assert mime == ""
    assert payload == uri


def test_parse_data_uri_malformed_no_comma() -> None:
    uri = "data:image/png;base64"
    mime, payload = parse_data_uri(uri)
    assert mime == ""
    assert payload == uri


# ---------- contains_value ----------


def test_contains_value_present() -> None:
    assert contains_value("png,jpeg,webp", "jpeg") is True


def test_contains_value_absent() -> None:
    assert contains_value("png,jpeg,webp", "gif") is False


def test_contains_value_whitespace_trimmed() -> None:
    # Provider-config CSVs sometimes carry whitespace around commas.
    assert contains_value("png, jpeg , webp", "jpeg") is True


def test_contains_value_empty_csv() -> None:
    assert contains_value("", "anything") is False


# ---------- set_additional_properties_false ----------


def test_set_additional_properties_false_on_object_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    set_additional_properties_false(schema)
    assert schema["additionalProperties"] is False
    # Required is auto-filled when missing — OpenAI strict mode requires it.
    assert set(schema["required"]) == {"name", "age"}


def test_set_additional_properties_false_preserves_existing_required() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }
    set_additional_properties_false(schema)
    # User-supplied `required` is not stomped.
    assert schema["required"] == ["name"]


def test_set_additional_properties_false_recurses_into_nested_objects() -> None:
    schema = {
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    set_additional_properties_false(schema)
    assert schema["properties"]["address"]["additionalProperties"] is False


def test_set_additional_properties_false_recurses_into_arrays() -> None:
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
        },
    }
    set_additional_properties_false(schema)
    assert schema["items"]["additionalProperties"] is False


def test_set_additional_properties_false_skips_non_dict() -> None:
    # No raise on a non-dict — defensive idempotence.
    set_additional_properties_false("not a schema")  # type: ignore[arg-type]
    set_additional_properties_false(42)  # type: ignore[arg-type]


# ---------- remove_additional_properties ----------


def test_remove_additional_properties_drops_top_level() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"type": "integer"}},
    }
    remove_additional_properties(schema)
    assert "additionalProperties" not in schema


def test_remove_additional_properties_recurses_into_nested() -> None:
    schema = {
        "type": "object",
        "properties": {
            "inner": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            }
        },
    }
    remove_additional_properties(schema)
    assert "additionalProperties" not in schema["properties"]["inner"]


def test_remove_additional_properties_recurses_into_array_items() -> None:
    schema = {
        "type": "array",
        "items": {"additionalProperties": False, "type": "object"},
    }
    remove_additional_properties(schema)
    assert "additionalProperties" not in schema["items"]


def test_remove_additional_properties_skips_non_dict() -> None:
    # No raise on non-dict.
    remove_additional_properties(None)  # type: ignore[arg-type]


# ---------- extract_path / extract_int_path (already used widely;
#            keeping a sanity case to lock in behavior) ----------


def test_extract_path_simple_dotted_lookup() -> None:
    data = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
    assert (
        extract_path(data, "candidates[0].content.parts[0].text") == "hello"
    )


def test_extract_int_path_returns_int_or_zero() -> None:
    data = {"usage": {"input_tokens": 42}}
    assert extract_int_path(data, "usage.input_tokens") == 42
    assert extract_int_path(data, "usage.missing") == 0


# ---------- set_nested_field / merge_into_parent (already used widely) ----------


def test_set_nested_field_creates_intermediate_maps() -> None:
    body: dict = {}
    set_nested_field(body, "thinking.budget_tokens", 1024)
    assert body == {"thinking": {"budget_tokens": 1024}}


def test_merge_into_parent_into_existing_dict() -> None:
    body = {"thinking": {"budget_tokens": 1024}}
    merge_into_parent(body, "thinking.budget_tokens", {"type": "enabled"})
    assert body == {"thinking": {"budget_tokens": 1024, "type": "enabled"}}
