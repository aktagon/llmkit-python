"""Phase 3 slice 2b — bridges legacy sync ``prompt_stream`` (callback)
to typed-builder ``async for chunk in text.stream(...)``.

The legacy API is sync and blocks until the entire stream is consumed,
calling a user-supplied ``on_chunk`` for each delta. Async-iterator
semantics on the typed builder need a running task that yields chunks
on demand. We bridge with ``asyncio.Queue`` + ``asyncio.to_thread``:
the producer runs on a worker thread, chunks land in the queue via the
callback (using ``loop.call_soon_threadsafe`` to cross the thread
boundary), the async generator pulls from the queue.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ..client import prompt_stream as legacy_prompt_stream
from .text import _build_provider, _build_request

if TYPE_CHECKING:
    from . import Text


_DONE = object()


async def text_stream(b: "Text", msg: str) -> AsyncIterator[str]:
    provider = _build_provider(b)
    request = _build_request(b, msg)
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
    # maxsize=64 caps memory if a hostile/buggy provider streams
    # faster than the consumer drains. on_chunk runs on a worker
    # thread, so it uses run_coroutine_threadsafe(...).result() to
    # block the worker thread until queue.put completes — providing
    # real backpressure across the thread boundary. Matches Go
    # chan(64) and TS bounded-queue semantics.
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)

    def on_chunk(chunk: str) -> None:
        # Called from the worker thread; cross back to the loop and
        # block the worker until the put succeeds (queue not full).
        fut = asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
        fut.result()

    async def producer() -> None:
        # We're on the loop thread here. Use queue.put (await) so the
        # bounded-queue backpressure applies to the sentinel as well.
        # call_soon_threadsafe(put_nowait, ...) would silently raise
        # QueueFull when the queue is at capacity and the consumer
        # would hang waiting for a sentinel that never arrived.
        try:
            await asyncio.to_thread(
                legacy_prompt_stream, provider, request, on_chunk, **kwargs
            )
            await queue.put(_DONE)
        except BaseException as exc:
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
        # Consumer broke early — best-effort cancel the producer task.
        # The worker thread can't be killed; the legacy prompt_stream
        # will continue until the HTTP socket drains. The task wrapper
        # is cancellable, which is enough to release the awaiting loop.
        if not task.done():
            task.cancel()
