"""









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
    queue: asyncio.Queue = asyncio.Queue()

    def on_chunk(chunk: str) -> None:
        #
        loop.call_soon_threadsafe(queue.put_nowait, chunk)

    async def producer() -> None:
        try:
            await asyncio.to_thread(
                legacy_prompt_stream, provider, request, on_chunk, **kwargs
            )
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)
        except BaseException as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)

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
        #
        if not task.done():
            task.cancel()
