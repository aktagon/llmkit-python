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


def _navigate(data: Any, path: str) -> Any:
    """Walk a dotted path with array-index support, returning None on any miss.

    Shared by the optional extractors below; the two returning-zero extractors
    keep their own walks so their behaviour is unchanged.
    """
    current: Any = data
    for part in path.split("."):
        m = _INDEX_RE.match(part)
        if m:
            if not isinstance(current, dict):
                return None
            current = current.get(m.group("field"))
            idx = int(m.group("idx"))
            if not isinstance(current, list) or idx >= len(current):
                return None
            current = current[idx]
        else:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
    return current


def opt_int_path(data: Any, path: str) -> int | None:
    """extract_int_path's honest form (ADR-081 AVAIL-001).

    None when the provider declares no location for this dimension (empty
    path) or the location is absent from the body; the value — which may be a
    genuine zero — when the provider reported one.

    This is where the ambiguity used to be manufactured. extract_int_path
    answers "unreported" and "reported as zero" with the same 0, and every
    Usage dimension flowed through it, so the lie was created once and copied
    everywhere.
    """
    if not path:
        return None
    current = _navigate(data, path)
    if isinstance(current, bool):
        return int(current)
    if isinstance(current, (int, float)):
        return int(current)
    return None


def opt_float_path(data: Any, path: str) -> float | None:
    """opt_int_path for the fractional ADR-027 cost field."""
    if not path:
        return None
    current = _navigate(data, path)
    if isinstance(current, bool):
        return None
    if isinstance(current, (int, float)):
        return float(current)
    return None


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


def set_wire_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Place value at a dot-notation path with array index support
    ("choices[0].message.content"), creating intermediate maps and array elements
    as it descends. It is the navigate-or-create inverse of extract_path and walks
    the identical generated path strings.

    An empty path (the provider declares no location for this field) or an empty
    value is a no-op: there is nothing to write, and materializing a zero would
    invent a field the provider never sent.
    """
    if not path or _is_empty_wire_value(value):
        return
    parts = path.split(".")
    current = data
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        m = _INDEX_RE.match(part)
        if m is None:
            if last:
                current[part] = value
                return
            current = _child_map(current, part)
            continue
        field_name = m.group("field")
        idx = int(m.group("idx"))
        arr = current.get(field_name)
        if not isinstance(arr, list):
            arr = []
            current[field_name] = arr
        while len(arr) <= idx:
            arr.append(None)
        if last:
            arr[idx] = value
            return
        elem = arr[idx]
        if not isinstance(elem, dict):
            elem = {}
            arr[idx] = elem
        current = elem


def _child_map(parent: dict[str, Any], field_name: str) -> dict[str, Any]:
    """Return parent[field_name] as a dict, creating it when absent or mistyped."""
    child = parent.get(field_name)
    if not isinstance(child, dict):
        child = {}
        parent[field_name] = child
    return child


def _is_empty_wire_value(value: Any) -> bool:
    """Report whether value is NOT REPORTED, which is the one case the encoder
    must not write: materializing a value there would invent a field the
    provider never sent.

    A reported ZERO is written, and that is the change ADR-081 forces here. The
    old rule dropped every zero because the type could not tell the two apart,
    so an explicit ``cached_tokens: 0`` round-tripped to a body that omitted the
    field — an asymmetry the SYM-006 fixed point could not see, because decoding
    the omission produced the same 0 it started from.
    """
    if isinstance(value, str):
        return value == ""
    return value is None


def merge_into_parent(body: dict[str, Any], path: str, extras: dict[str, Any]) -> None:
    """Merge extras into the dict that contains the leaf of path.

    For "a.b.c" extras land in body["a"]["b"]; for "x" they land in body itself.
    Used to attach static sibling fields (OptionOverrideDef.extra_fields_json)
    next to a provider-specific dotted JSON key.
    """
    parts = path.split(".")
    if len(parts) == 1:
        body.update(extras)
        return
    current = body
    for part in parts[:-1]:
        nxt = current.get(part)
        if not isinstance(nxt, dict):
            return
        current = nxt
    current.update(extras)


def deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Merge src into dst recursively: dicts merge per key, scalars overwrite.

    Used for OptionOverrideDef.root_extra_fields_json (ADR-029) so e.g.
    {"thinking":{"type":"adaptive"}} composes with an existing thinking
    object rather than replacing it.
    """
    for k, v in src.items():
        dv = dst.get(k)
        if isinstance(v, dict) and isinstance(dv, dict):
            deep_merge(dv, v)
        else:
            dst[k] = v


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
