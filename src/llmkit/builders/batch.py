"""









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
    """






"""

    async def wait(
        self,
        *,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        request_timeout: float = 600.0,
        poll_deadline: float = DEFAULT_POLL_DEADLINE,
    ) -> list[Response]:
        """

"""
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
        """


"""
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
    """


"""
    #
    #
    #
    #
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
    """



"""
    provider = _provider_for(b)
    #
    #
    #
    #
    requests = [_build_request(b, p) for p in prompts]
    legacy = await asyncio.to_thread(
        legacy_submit_batch,
        provider,
        requests,
        **_option_kwargs(b),
    )
    return BatchHandle(id=legacy.id, provider=legacy.provider, raw=b._raw)
