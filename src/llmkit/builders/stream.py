"""Trailing-handle stream wrapper for ``*Text.stream``.

The legacy ``prompt_stream`` is sync and callback-driven, returning the
final ``Response`` once the SSE socket drains. Async-iterator semantics
on the typed builder need a running task that yields chunks on demand.
We bridge with ``asyncio.Queue`` + ``asyncio.to_thread``: producer runs
on a worker thread, chunks land in the queue via the callback (using
``run_coroutine_threadsafe`` to cross the thread boundary), the async
iterator pulls from the queue, and once the producer task completes we
publish the final ``Response`` on ``self._response``.

Usage::

    stream = c.text.system("...").stream("hi")
    async for chunk in stream:
        print(chunk, end="")
    print(stream.response.tokens)  # populated after iteration ends
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ..client import prompt_stream as legacy_prompt_stream
from ..types import Response
from .text import _build_provider, _build_request

if TYPE_CHECKING:
    from . import Text


_DONE = object()


class TextStream:
    """Async-iterable wrapper carrying the final Response after iteration.

    ``response`` is ``None`` before iteration completes; populated to a
    ``Response`` instance once the producer task finishes (success or
    error). ``error`` is the terminal exception, if any.
    """

    def __init__(self, b: "Text", msg: str) -> None:
        self._b = b
        self._msg = msg
        self._response: Response | None = None
        self._error: BaseException | None = None
        self._consumed = False

    @property
    def response(self) -> Response | None:
        """Accumulated response (text + tokens) once iteration completes."""
        return self._response

    @property
    def error(self) -> BaseException | None:
        """Terminal exception, if any."""
        return self._error

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        if self._consumed:
            return
        self._consumed = True

        b = self._b
        provider = _build_provider(b)
        request = _build_request(b, self._msg)
        kwargs: dict = {}
        if b._max_tokens is not None:
            kwargs["max_tokens"] = b._max_tokens
        if b._temperature is not None:
            kwargs["temperature"] = b._temperature
        if b._caching:
            kwargs["caching"] = True
        if b._middleware:
            kwargs["middleware"] = list(b._middleware)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)

        def on_chunk(chunk: str) -> None:
            fut = asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            fut.result()

        async def producer() -> None:
            try:
                resp = await asyncio.to_thread(
                    legacy_prompt_stream, provider, request, on_chunk, **kwargs
                )
                # Stash the final response BEFORE the sentinel so the
                # consumer sees it on its next `response` access after
                # iteration ends (no race: queue ordering is fine, but
                # python's MAYBE-released-before-sentinel is also fine
                # because we publish on the producer task's completion).
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
            # Consumer broke early — best-effort cancel the producer
            # task. The worker thread itself can't be killed; the legacy
            # prompt_stream will keep draining the HTTP socket.
            if not task.done():
                task.cancel()


def text_stream(b: "Text", msg: str) -> TextStream:
    return TextStream(b, msg)
