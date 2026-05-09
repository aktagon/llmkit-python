"""Phase 3 slice 1 — wires Text.prompt against the legacy ``prompt`` API.

Codegen-emitted ``Text.prompt`` delegates to ``text_prompt(self, msg)``
via PYTHON_BUILDER_SKIP_TERMINALS. The bridge is sync→async: legacy
``llmkit.client.prompt`` is synchronous, so we wrap it in
``asyncio.to_thread`` to keep the typed-builder API uniformly async.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..client import prompt as legacy_prompt
from ..types import Message, Provider, Request, Response

if TYPE_CHECKING:
    from . import Text


def _build_provider(b: "Text") -> Provider:
    p = Provider(
        name=b.client.provider.name,
        api_key=b.client.provider.api_key,
    )
    if b._model:
        p.model = b._model
    if b.client.provider.base_url:
        p.base_url = b.client.provider.base_url
    return p


def _build_request(b: "Text", final_text: str) -> Request:
    req = Request()
    if b._system:
        req.system = b._system

    # Concatenate accumulated text Parts + final prompt.
    parts_text: list[str] = []
    for p in b._parts:
        if p.text:
            parts_text.append(p.text)
    if final_text:
        parts_text.append(final_text)
    user = "".join(parts_text)

    # Legacy Request: messages + user are mutually exclusive in the
    # downstream body builder. Append the final user turn to messages
    # when history is present; otherwise use the simpler user field.
    if b._history:
        msgs = list(b._history)
        if user:
            msgs.append(Message(role="user", content=user))
        req.messages = msgs
    elif user:
        req.user = user
    if b._schema:
        req.schema = b._schema
    return req


async def text_prompt(b: "Text", msg: str) -> Response:
    provider = _build_provider(b)
    request = _build_request(b, msg)
    kwargs: dict = {}
    if b._max_tokens is not None:
        kwargs["max_tokens"] = b._max_tokens
    if b._temperature is not None:
        kwargs["temperature"] = b._temperature
    if b._top_p is not None:
        kwargs["top_p"] = b._top_p
    if b._top_k is not None:
        kwargs["top_k"] = b._top_k
    if b._frequency_penalty is not None:
        kwargs["frequency_penalty"] = b._frequency_penalty
    if b._presence_penalty is not None:
        kwargs["presence_penalty"] = b._presence_penalty
    if b._seed is not None:
        kwargs["seed"] = b._seed
    if b._stop_sequences:
        kwargs["stop_sequences"] = list(b._stop_sequences)
    if b._thinking_budget is not None:
        kwargs["thinking_budget"] = b._thinking_budget
    if b._reasoning_effort:
        kwargs["reasoning_effort"] = b._reasoning_effort
    if b._caching:
        kwargs["caching"] = True
    if b._middleware:
        kwargs["middleware"] = list(b._middleware)
    return await asyncio.to_thread(legacy_prompt, provider, request, **kwargs)
