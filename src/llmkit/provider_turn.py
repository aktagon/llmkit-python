"""Verbatim assistant-turn capture (ADR-085).

llmkit keeps a canonical projection of every turn — role, content, tool calls —
and used to rebuild the next request's assistant turn from it. That projection is
lossy in three ways (ADR-085 §1): it has nowhere to put reasoning, it drops
assistant prose that accompanies a tool call, and it has no slot for per-part
provider metadata. The fix is not a richer projection but a second
representation alongside it: keep the provider's own bytes for the turn and send
those back unchanged.

Nothing here parses the payload. The only structure this file reads is the path
down to the turn — everything at and below it is carried as the provider wrote
it.

Python-specific note. json.loads gives a dict with no memory of the source text,
and json.dumps would then re-render it (different separators, non-ASCII escaped).
So capture SLICES the response text: the walk below finds the byte range of the
value at the path and returns that substring, using json.JSONDecoder.raw_decode
to measure each value it has to skip over. The stdlib parser stays the authority
on what a JSON value is; this file only tracks where one ends.
"""

from __future__ import annotations

import json

from .providers.generated.providers import ProviderSpec
from .structs import ProviderTurn
from .transforms import _Msg, _MsgTurn

_DECODER = json.JSONDecoder()
_WHITESPACE = " \t\n\r"


def split_path_segment(part: str) -> tuple[str, int]:
    """Split one dot-notation path segment into a field name and an array index:
    "choices[0]" -> ("choices", 0); "message" -> ("message", -1).

    
    single parser here, so capture cannot drift from the path facts it reads. A
    malformed segment yields the whole segment as a field name and no index,
    which resolves to "no such field" one step later — never a silently widened
    match to the whole array, which is what a negative index would read back as.
    """
    bracket = part.find("[")
    if bracket == -1 or not part.endswith("]"):
        return part, -1
    inner = part[bracket + 1 : -1]
    # Reject "+1", "-1", " 1" and "" — str.isdigit is the check int() is not.
    if not inner.isdigit():
        return part, -1
    return part[:bracket], int(inner)


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in _WHITESPACE:
        i += 1
    return i


def _span_of_member(text: str, start: int, field: str) -> tuple[int, int] | None:
    """Return the (start, end) span of `field`'s value in the JSON object that
    begins at `start`, or None when this is not an object or has no such key."""
    i = _skip_ws(text, start)
    if i >= len(text) or text[i] != "{":
        return None
    i += 1
    while True:
        i = _skip_ws(text, i)
        if i >= len(text) or text[i] == "}":
            return None
        if text[i] != '"':
            return None
        try:
            # A JSON key is a JSON string, so the same decoder measures it.
            key, i = _DECODER.raw_decode(text, i)
            i = _skip_ws(text, i)
            if i >= len(text) or text[i] != ":":
                return None
            value_start = _skip_ws(text, i + 1)
            _, value_end = _DECODER.raw_decode(text, value_start)
        except ValueError:
            return None
        if key == field:
            return value_start, value_end
        i = _skip_ws(text, value_end)
        if i >= len(text) or text[i] != ",":
            return None
        i += 1


def _span_of_element(text: str, start: int, index: int) -> tuple[int, int] | None:
    """Return the (start, end) span of element `index` in the JSON array that
    begins at `start`, or None when this is not an array or is too short."""
    i = _skip_ws(text, start)
    if i >= len(text) or text[i] != "[":
        return None
    i += 1
    position = 0
    while True:
        i = _skip_ws(text, i)
        if i >= len(text) or text[i] == "]":
            return None
        try:
            _, value_end = _DECODER.raw_decode(text, i)
        except ValueError:
            return None
        if position == index:
            return i, value_end
        position += 1
        i = _skip_ws(text, value_end)
        if i >= len(text) or text[i] != ",":
            return None
        i += 1


def extract_raw_json_path(text: str, path: str) -> str | None:
    """Return the VERBATIM JSON text of the value at `path`, or None when the
    path does not resolve.

    The distinction from paths.extract_path is the whole point: that walker
    descends a dict that has already been through json.loads, so re-encoding its
    result emits Python's rendering of the value (its separators, its
    non-ASCII escaping) rather than the provider's.
    """
    if not path:
        return None
    start, end = 0, len(text)
    for part in path.split("."):
        field, index = split_path_segment(part)
        if field:
            span = _span_of_member(text, start, field)
            if span is None:
                return None
            start, end = span
        if index >= 0:
            span = _span_of_element(text, start, index)
            if span is None:
                return None
            start, end = span
    return text[start:end]


def assistant_turn_path(cfg: ProviderSpec, chat_wire_shape: str) -> str:
    """Where one replayable assistant turn sits in a response body for this
    provider under this wire shape, or "" when the shape declares no position.

    An empty result is a DECLARED absence, not a missing lookup: ChatBedrock
    carries assistantTurnUnanchored rather than a path, because nobody has
    probed what an assistant turn looks like on Converse (ADR-085 OQ-5). The
    
    "declared unanchored" and never "somebody forgot".
    """
    for protocol in cfg.chat_protocols:
        if protocol.wire_shape == chat_wire_shape:
            return protocol.assistant_turn_path
    return ""


def _effective_chat_wire_shape(cfg: ProviderSpec, chat_wire_shape: str) -> str:
    """The shape a response was produced under. An empty argument means the
    caller did not route through Text.protocol(...) — the batch path passes ""
    because batch is Chat-Completions-only (ADR-055) — so the provider's default
    shape applies."""
    return chat_wire_shape or cfg.chat_wire_shape


def capture_provider_turn(
    body: bytes | str, cfg: ProviderSpec, chat_wire_shape: str
) -> ProviderTurn | None:
    """Lift the assistant turn out of a response body, or return None when this
    shape declares no turn position or the body carries nothing there."""
    shape = _effective_chat_wire_shape(cfg, chat_wire_shape)
    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else body
    wire = extract_raw_json_path(text, assistant_turn_path(cfg, shape))
    # A JSON null at the path is the provider declining to send a turn, not a
    # turn whose content is null — Google nulls candidates[0].content on a safety
    # block, and OpenAI-compatible proxies null choices[0].message on a content
    # filter. A bare `null` is four characters of a perfectly good JSON value, so
    # an emptiness check alone captures it and the next request appends a bare
    # null to the message array, which is a 400.
    if wire is None or not wire.strip() or wire.strip() == "null":
        return None
    return ProviderTurn(wire_shape=shape, wire=wire)


def resolve_turns(msgs: list[_Msg], cfg: ProviderSpec) -> list[_Msg]:
    """The RSN-006 boundary: a captured payload is replayed only under the shape
    that produced it, and a mismatch drops it and reconstructs the turn from the
    canonical projection instead.

    One unconditional rule, applied once per request where cfg and the message
    list first meet, so no transform has to remember the check. The draft ADR
    made this branch on whether the provider mandates the echo and raised an
    error on the mandating ones; RESEARCH-017 measured that set to be empty, so
    only the drop arm was ever reachable.

    Dropping is the safe direction here, and the measurement is why: every probed
    provider ACCEPTS a request with the payload omitted, while a mangled payload
    is the single 400 anywhere in the matrix. Replaying an Anthropic block array
    into Google's contents array would be exactly that mangling.
    """
    out: list[_Msg] = []
    for m in msgs:
        # Two conditions, not one. Matching the shape is not enough: the shape
        # must also DECLARE a turn position. A payload claiming an unanchored
        # shape can only come from caller-supplied or load_history()ed data, and
        # the transform for such a shape has no replay arm — so without the
        # second check, a history carrying ProviderTurn(wire_shape="ChatBedrock")
        # reaches a builder with nowhere to put it.
        if isinstance(m, _MsgTurn) and not (
            m.shape == cfg.chat_wire_shape and assistant_turn_path(cfg, m.shape)
        ):
            out.append(m.fallback)
        else:
            out.append(m)
    return out
