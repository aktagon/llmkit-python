"""Phase 3 slice 1 — wires Image.generate against legacy ``generate_image``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..image import (
    ImageRequest,
    ImageResponse,
    Part,
    generate_image as legacy_generate_image,
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
    # XOR rule: prompt or parts, never both. If chain accumulated parts,
    # append msg as final text Part and use the parts path; otherwise
    # use the prompt sugar path.
    if b._parts:
        from ..image import Text as TextPart

        if msg:
            request.parts = [*b._parts, TextPart(msg)]
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
        legacy_generate_image, provider, request, **kwargs
    )
