"""Wires the ``Text.batch`` execution-mode terminal + BatchHandle.wait/poll.

Batch is a text execution mode (parallel to ``stream``): the codegen-emitted
``Text.batch`` method delegates to ``text_batch(self, *prompts)``. The terminal
queues the batch and RETURNS a handle without blocking; the blocking one-liner
is the explicit compose ``(await c.text.batch(...)).wait()``.

BatchHandle is a typed-builder-owned class with ``wait()`` / ``poll()``
methods. It MUST NOT be awaitable (AJU-007): a result-resolving thenable
would silently run a minutes-long job on a stray ``await``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..batch import (
    DEFAULT_POLL_DEADLINE,
    DEFAULT_POLL_INTERVAL,
    _new_batch_adapter,
    submit_batch as legacy_submit_batch,
)
from ..errors import ValidationError
from ..job import JobStatus, poll_engine_once, poll_job_async
from ..structs import BatchHandle as _BatchHandleData
from ..types import Provider, Response
from .text import _build_request

if TYPE_CHECKING:
    from . import Text


class BatchHandle(_BatchHandleData):
    """Typed-builder BatchHandle. Inherits the ontology-generated data
    shape (id, provider, raw) and adds ``wait()`` + ``poll()`` methods so callers
    can chain ``handle = await c.text.batch(...); await handle.wait()``
    without reaching for the ``wait_batch`` free function.

    AJU-007: this handle is deliberately NOT awaitable (no ``__await__``) —
    a result-resolving thenable would run a minutes-long job on a stray
    ``await``. The blocking path is the explicit ``(await c.text.batch(...)).wait()``."""

    async def wait(
        self,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        request_timeout: float = 600.0,
        poll_deadline: float = DEFAULT_POLL_DEADLINE,
    ) -> list[Response]:
        """Block until the batch finishes. A thin loop over ``poll`` (ADR-063
        POLL-003) via the shared engine — the between-poll wait is a cancellable
        ``asyncio.sleep`` so ``asyncio.CancelledError`` propagates (S06)."""
        adapter = _new_batch_adapter(
            self, request_timeout, poll_interval, poll_deadline, self.raw
        )
        return await poll_job_async(adapter)

    async def poll(
        self,
        *,
        request_timeout: float = 600.0,
        poll_deadline: float = DEFAULT_POLL_DEADLINE,
    ) -> JobStatus[list[Response]]:
        """Perform exactly ONE provider round-trip and return the normalized
        JobStatus (ADR-063 POLL-001). On a completed batch JobStatus.result carries
        the ordered responses (two-hop fetch inline); a provider-reported failure
        yields JobState.FAILED with the status on JobStatus.cause."""
        adapter = _new_batch_adapter(
            self, request_timeout, DEFAULT_POLL_INTERVAL, poll_deadline, self.raw
        )
        return await poll_engine_once(adapter)


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


async def text_batch(b: "Text", *prompts: str) -> BatchHandle:
    """Queue a batch and return a handle without blocking. The chain's
    accumulated config (system, max_tokens, schema, ...) applies to every prompt
    in the variadic. The chain's ``.raw()`` opt-in is remembered on the returned
    BatchHandle so ``handle.wait()`` honors it (ADR-014). The blocking one-liner
    is the explicit compose ``(await c.text.batch(...)).wait()``."""
    provider = _provider_for(b)
    # Reuse the prompt path's request builder so batch is byte-identical to
    # Text.prompt (images + files + history + schema), matching the Go/TS/Rust
    # batchInputs -> buildRequest reuse. ADR-012 REQ-PROP-003: one builder, no
    # drift between the prompt and batch wire bodies.
    requests = [_build_request(b, p) for p in prompts]
    legacy = await asyncio.to_thread(
        legacy_submit_batch,
        provider,
        requests,
        **_option_kwargs(b),
    )
    return BatchHandle(id=legacy.id, provider=legacy.provider, raw=b._raw)
