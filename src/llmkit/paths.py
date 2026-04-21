"""Dot-notation path helpers for JSON-shaped dicts, plus MIME sniffing and data URIs."""

from __future__ import annotations

import os
import re
from typing import Any

_INDEX_RE = re.compile(r"^(?P<field>.+)\[(?P<idx>\d+)\]$")


def extract_path(data: Any, path: str) -> str:
    """Navigate dot-notation path with array index support. Returns stringified leaf.

    Examples: "content[0].text", "choices[0].message.content", "usage.input_tokens".
    """
    if not path:
        return ""
    current: Any = data
    for part in path.split("."):
        m = _INDEX_RE.match(part)
        if m:
            field = m.group("field")
            idx = int(m.group("idx"))
            if isinstance(current, dict):
                current = current.get(field)
            else:
                return ""
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            else:
                return ""
        else:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return ""
    if current is None:
        return ""
    if isinstance(current, str):
        return current
    return str(current)


def extract_int_path(data: Any, path: str) -> int:
    """Like extract_path but returns an int (0 on miss)."""
    if not path:
        return 0
    current: Any = data
    for part in path.split("."):
        m = _INDEX_RE.match(part)
        if m:
            field = m.group("field")
            idx = int(m.group("idx"))
            if isinstance(current, dict):
                current = current.get(field)
            else:
                return 0
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            else:
                return 0
        else:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return 0
    if isinstance(current, bool):
        return int(current)
    if isinstance(current, (int, float)):
        return int(current)
    return 0


def detect_mime_type(path: str) -> str:
    """Map file extension to MIME type (subset matching the Go handwritten table)."""
    ext = os.path.splitext(path)[1].lower()
    mapping = {
        ".pdf": "application/pdf",
        ".json": "application/json",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mapping.get(ext, "application/octet-stream")


def parse_data_uri(uri: str) -> tuple[str, str]:
    """Split a `data:<mime>;base64,<payload>` URI. Returns ("", uri) on non-data URIs."""
    if not uri.startswith("data:"):
        return "", uri
    rest = uri[len("data:"):]
    parts = rest.split(",", 1)
    if len(parts) != 2:
        return "", uri
    meta, data = parts
    mime = meta.removesuffix(";base64")
    return mime, data


def set_nested_field(body: dict[str, Any], path: str, value: Any) -> None:
    """Set a value at a dot-notation path in a nested dict, creating maps as needed."""
    parts = path.split(".")
    if len(parts) == 1:
        body[parts[0]] = value
        return
    current = body
    for part in parts[:-1]:
        existing = current.get(part)
        if isinstance(existing, dict):
            current = existing
        else:
            new_map: dict[str, Any] = {}
            current[part] = new_map
            current = new_map
    current[parts[-1]] = value


def set_additional_properties_false(schema: Any) -> None:
    """Recursively set additionalProperties=false and auto-fill required on object schemas."""
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        props = schema.get("properties")
        if isinstance(props, dict):
            if "required" not in schema:
                schema["required"] = list(props.keys())
            for value in props.values():
                set_additional_properties_false(value)
    items = schema.get("items")
    if items is not None:
        set_additional_properties_false(items)


def remove_additional_properties(schema: Any) -> None:
    """Recursively delete additionalProperties from JSON schema."""
    if not isinstance(schema, dict):
        return
    schema.pop("additionalProperties", None)
    props = schema.get("properties")
    if isinstance(props, dict):
        for value in props.values():
            remove_additional_properties(value)
    items = schema.get("items")
    if items is not None:
        remove_additional_properties(items)


def contains_value(csv: str, value: str) -> bool:
    """Return True if the comma-separated `csv` contains `value` (whitespace-trimmed)."""
    return value in {token.strip() for token in csv.split(",")}
