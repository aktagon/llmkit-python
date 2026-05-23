#

"""


"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ParsedModelRecord:
    """


"""
    id: str = ""
    display_name: str = ""
    description: str = ""
    created: int = 0
    context_window: int = 0
    max_output: int = 0
    raw: Any | None = None


@dataclass
class ParsedModelsPage:
    """
"""
    records: list[ParsedModelRecord] = field(default_factory=list)
    next_cursor: str = ""


def _parse_iso8601_best(s: str) -> int:
    """

"""
    if not s:
        return 0
    try:
        #
        normalized = s.replace("Z", "+00:00") if s.endswith("Z") else s
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return 0


def parse_anthropic_models_response(body: bytes | str) -> ParsedModelsPage:
    """"""
    envelope = json.loads(body)
    data = envelope.get("data") or []
    records: list[ParsedModelRecord] = []
    for wire in data:
        max_out = wire.get("max_output_tokens") or wire.get("max_tokens") or 0
        records.append(
            ParsedModelRecord(
                id=str(wire.get("id") or ""),
                display_name=str(wire.get("display_name") or ""),
                context_window=int(wire.get("max_input_tokens") or 0),
                max_output=int(max_out or 0),
                created=_parse_iso8601_best(str(wire.get("created_at") or "")),
                raw=wire,
            )
        )
    next_cursor = ""
    if envelope.get("has_more") and envelope.get("last_id"):
        next_cursor = str(envelope["last_id"])
    return ParsedModelsPage(records=records, next_cursor=next_cursor)


def parse_openai_cohort_models_response(body: bytes | str) -> ParsedModelsPage:
    """


"""
    parsed = json.loads(body)
    if isinstance(parsed, list):
        data = parsed
    else:
        data = parsed.get("data") or []
    records = [
        ParsedModelRecord(
            id=str(wire.get("id") or ""),
            created=int(wire.get("created") or 0),
            raw=wire,
        )
        for wire in data
    ]
    return ParsedModelsPage(records=records, next_cursor="")


def parse_google_models_response(body: bytes | str) -> ParsedModelsPage:
    """
"""
    envelope = json.loads(body)
    data = envelope.get("models") or []
    prefix = "models/"
    records: list[ParsedModelRecord] = []
    for wire in data:
        name = str(wire.get("name") or "")
        if name.startswith(prefix):
            name = name[len(prefix):]
        records.append(
            ParsedModelRecord(
                id=name,
                display_name=str(wire.get("displayName") or ""),
                description=str(wire.get("description") or ""),
                context_window=int(wire.get("inputTokenLimit") or 0),
                max_output=int(wire.get("outputTokenLimit") or 0),
                raw=wire,
            )
        )
    return ParsedModelsPage(records=records, next_cursor=str(envelope.get("nextPageToken") or ""))
