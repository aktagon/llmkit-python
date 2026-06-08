"""Owns Music.generate translation (ADR-033). The typed-builder method is
the only public entry point for music generation; the internal
generate_music helper in ../music.py holds the runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..image import Part
from ..music import (
    MusicRequest,
    generate_music as run_music_generation,
)
from ..structs import MusicResponse
from ..types import Provider

if TYPE_CHECKING:
    from . import Music


async def music_generate(b: "Music", msg: str) -> MusicResponse:
    provider = Provider(
        name=b.client.provider.name,
        api_key=b.client.provider.api_key,
    )
    if b.client.provider.base_url:
        provider.base_url = b.client.provider.base_url

    # Mirror go/music_builder.go: chain-accumulated parts plus an optional
    # trailing text part from generate(msg). The XOR (prompt vs parts) is
    # enforced by _normalize_music_parts in the runtime — both empty errors.
    request = MusicRequest(model=b._model)
    if b._parts:
        if msg:
            request.parts = [*b._parts, Part(text=msg)]
        else:
            request.parts = list(b._parts)
    elif msg:
        request.prompt = msg

    kwargs: dict = {}
    if b._middleware:
        kwargs["middleware"] = list(b._middleware)
    if b._raw:
        kwargs["raw"] = True

    return await asyncio.to_thread(
        run_music_generation, provider, request, **kwargs
    )
