"""
















"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ..client import prompt_stream as legacy_prompt_stream
from ..errors import ValidationError
from ..types import Response
from .text import _build_provider, _build_request

if TYPE_CHECKING:
    from . import Text


_DONE = object()


class TextStream:
    """




"""

    def __init__(self, b: "Text", msg: str) -> None:
        self._b = b
        self._msg = msg
        self._response: Response | None = None
        self._error: BaseException | None = None
        self._consumed = False

    @property
    def response(self) -> Response | None:
        """"""
        return self._response

    @property
    def error(self) -> BaseException | None:
        """"""
        return self._error

    def __aiter__(self) -> AsyncIterator[str]:
        if self._consumed:
            raise RuntimeError(
                "TextStream is single-use; create a new stream to iterate again"
            )
        self._consumed = True
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        b = self._b
        #
        #
        #
        if b._protocol:
            raise ValidationError(
                field="protocol",
                message="protocol (e.g. Responses) is only supported on the prompt terminal, not stream (ADR-055)",
            )
        provider = _build_provider(b)
        request = _build_request(b, self._msg)
        #
        #
        #
        #
        #
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

        loop = asyncio.get_running_loop()
        #
        #
        #
        #
        #
        #
        #
        #
        #
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)

        def on_chunk(chunk: str) -> None:
            fut = asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            fut.result()

        async def producer() -> None:
            try:
                resp = await asyncio.to_thread(
                    legacy_prompt_stream, provider, request, on_chunk, **kwargs
                )
                #
                #
                #
                #
                #
                self._response = resp
                await queue.put(_DONE)
            except BaseException as exc:
                self._error = exc
                await queue.put(exc)

        task = asyncio.create_task(producer())
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item  # type: ignore[misc]
        finally:
            #
            #
            #
            if not task.done():
                task.cancel()


def text_stream(b: "Text", msg: str) -> TextStream:
    return TextStream(b, msg)
