"""


"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..image import (
    ImageRequest,
    ImageResponse,
    Part,
    generate_image as run_image_generation,
)
from ..types import Provider

if TYPE_CHECKING:
    from . import Image


async def image_generate(b: "Image", msg: str) -> ImageResponse:
    provider = Provider(
        name=b.client.provider.name,
        api_key=b.client.provider.api_key,
    )
    if b.client.provider.base_url:
        provider.base_url = b.client.provider.base_url

    request = ImageRequest(model=b._model)
    #
    #
    #
    if b._parts:
        if msg:
            request.parts = [*b._parts, Part(text=msg)]
        else:
            request.parts = list(b._parts)
    elif msg:
        request.prompt = msg

    kwargs: dict = {}
    if b._aspect_ratio:
        kwargs["aspect_ratio"] = b._aspect_ratio
    if b._image_size:
        kwargs["image_size"] = b._image_size
    if b._include_text:
        kwargs["include_text"] = True
    if b._middleware:
        kwargs["middleware"] = list(b._middleware)

    return await asyncio.to_thread(
        run_image_generation, provider, request, **kwargs
    )
