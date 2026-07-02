"""Phase 3 slice 2a — wires Text.batch + Text.submit_batch + BatchHandle.wait.

The codegen-emitted Text.batch / Text.submit_batch methods delegate to
``text_batch(self, ...prompts)`` and ``text_submit_batch(self, ...)``
(see PYTHON_BUILDER_SKIP_TERMINALS in codegen/generate.py).

BatchHandle is promoted to a typed-builder-owned class with a
``wait()`` method. Mirrors the TS slice 2a approach: legacy
``llmkit.batch.BatchHandle`` is a plain dataclass; we wrap legacy
results into the new class so callers get ``handle.wait()`` for free,
matching Go's ``BatchHandle.Wait`` value-receiver shape.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..batch import (
    prompt_batch as legacy_prompt_batch,
    submit_batch as legacy_submit_batch,
    wait_batch as legacy_wait_batch,
)
from ..errors import ValidationError
from ..providers.generated.providers import ProviderName
from ..structs import BatchHandle as _BatchHandleData
from ..types import Provider, Request, Response

if TYPE_CHECKING:
    from . import Text


class BatchHandle(_BatchHandleData):
    """Typed-builder BatchHandle. Inherits the ontology-generated data
    shape (id, provider, raw) and adds a ``wait()`` method so callers
    can chain ``handle = await text.submit_batch(...); await handle.wait()``
    without reaching for the ``wait_batch`` free function."""

    async def wait(
        self, *, poll_interval: float = 2.0, request_timeout: float = 600.0
    ) -> list[Response]:
        return await asyncio.to_thread(
            legacy_wait_batch,
            self,
            poll_interval=poll_interval,
            request_timeout=request_timeout,
            raw=self.raw,
        )


def _provider_for(b: "Text") -> Provider:
    p = Provider(
        name=b.client.provider.name,
        api_key=b.client.provider.api_key,
        headers=b.client.provider.headers,
    )
    if b._model:
        p.model = b._model
    if b.client.provider.base_url:
        p.base_url = b.client.provider.base_url
    return p


def _build_request_for(b: "Text", prompt: str) -> Request:
    """Mirror of the TS buildRequest / Go buildRequest — builds a
    legacy ``Request`` from chained config + a final user message.
    Local copy here (not imported from text.py) so batch.py stays
    self-contained for the file-by-file phase 3 layout."""
    req = Request()
    if b._system:
        req.system = b._system
    # Concatenate accumulated text Parts + final prompt.
    parts_text: list[str] = []
    for p in b._parts:
        if p.text:
            parts_text.append(p.text)
    if prompt:
        parts_text.append(prompt)
    user = "".join(parts_text)
    if b._history:
        msgs = list(b._history)
        if user:
            from ..types import Message

            msgs.append(Message(role="user", content=user))
        req.messages = msgs
    elif user:
        req.user = user
    if b._schema:
        req.schema = b._schema
    return req


def _option_kwargs(b: "Text") -> dict:
    """Mirror of text.py's option-threading. Every chain-set field on the
    Text builder is propagated into the underlying batch call so the wire
    body carries the same knobs that the one-shot ``Text.prompt`` path
    sends. ADR-012 REQ-PROP-003 forbids drift between helpers."""
    # ADR-055: Protocol (e.g. Responses) is prompt-only in slice 1. Reject a
    # non-default protocol loudly rather than silently sending a Chat
    # Completions batch — the honest handling of a deferred-capability field
    # (REQ-PROP-003: read the field, don't silently drop it).
    if b._protocol:
        raise ValidationError(
            field="protocol",
            message="protocol (e.g. Responses) is only supported on the prompt terminal, not batch (ADR-055)",
        )
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
    if b._safety_settings:
        kwargs["safety_settings"] = list(b._safety_settings)
    if b._raw:
        kwargs["raw"] = True
    return kwargs


async def text_batch(b: "Text", *prompts: str) -> list[Response]:
    provider = _provider_for(b)
    requests = [_build_request_for(b, p) for p in prompts]
    return await asyncio.to_thread(
        legacy_prompt_batch,
        provider,
        requests,
        **_option_kwargs(b),
    )


async def text_submit_batch(b: "Text", *prompts: str) -> BatchHandle:
    provider = _provider_for(b)
    requests = [_build_request_for(b, p) for p in prompts]
    legacy = await asyncio.to_thread(
        legacy_submit_batch,
        provider,
        requests,
        **_option_kwargs(b),
    )
    return BatchHandle(id=legacy.id, provider=legacy.provider, raw=b._raw)
