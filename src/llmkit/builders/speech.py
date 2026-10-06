"""Owns Speech.generate translation (ADR-049). The typed-builder method is
the only public entry point for speech generation; the internal
generate_speech helper in ../speech.py holds the runtime."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..speech import (
    SpeechRequest,
    generate_speech as run_speech_generation,
)
from ..structs import SpeechResponse
from ..types import provider_from_config

if TYPE_CHECKING:
    from . import Speech


async def speech_generate(b: "Speech", msg: str) -> SpeechResponse:
    provider = provider_from_config(b.client.provider)

    request = SpeechRequest(model=b._model, voice=b._voice, text=msg)

    kwargs: dict = {}
    if b._middleware:
        kwargs["middleware"] = list(b._middleware)

    return await asyncio.to_thread(
        run_speech_generation, provider, request, **kwargs
    )
