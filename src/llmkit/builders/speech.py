"""

"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..speech import (
    SpeechRequest,
    generate_speech as run_speech_generation,
)
from ..structs import SpeechResponse
from ..types import Provider

if TYPE_CHECKING:
    from . import Speech


async def speech_generate(b: "Speech", msg: str) -> SpeechResponse:
    provider = Provider(
        name=b.client.provider.name,
        api_key=b.client.provider.api_key,
        headers=b.client.provider.headers,
    )
    if b.client.provider.base_url:
        provider.base_url = b.client.provider.base_url

    request = SpeechRequest(model=b._model, voice=b._voice, text=msg)

    return await asyncio.to_thread(run_speech_generation, provider, request)
