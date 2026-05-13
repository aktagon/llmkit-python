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
        if self._consumed:
            raise RuntimeError(
                "TextStream is single-use; create a new stream to iterate again"
            )
        self._consumed = True
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        b = self._b
        provider = _build_provider(b)
        request = _build_request(b, self._msg)
        # ADR-012 REQ-PROP-003: every chain-set field must propagate
        # through every helper. text.py and batch.py both read this same
        # set; stream.py must too. _build_provider/_build_request
        # (imported from text.py) cover _model/_history/_parts/_schema —
        # tracked as delegated suppressions.
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
        # Bounded queue gives natural backpressure when the consumer
        # is slower than the producer — without it a fast SSE producer
        # could buffer megabytes of pending chunks. Known limitation:
        # if the consumer breaks out of iteration mid-stream and the
        # queue is full, the worker thread can park indefinitely on
        # `fut.result()` (the producer asyncio task is cancelled but
        # that doesn't kill the worker). Tracked as a 1.0.x follow-up;
        # workaround for callers is to consume the iterator to
        # completion or use `asyncio.timeout(...)` around the loop.
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
