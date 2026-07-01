"""




"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..client import upload_file as _upload_file
from ..types import File, Provider

if TYPE_CHECKING:
    from . import Upload


async def upload_run(b: "Upload") -> File:
    has_bytes = bool(b._bytes)
    has_path = bool(b._path)

    if not has_bytes and not has_path:
        raise ValueError(
            "Upload: exactly one of bytes() or path() must be set"
        )
    if has_bytes and has_path:
        raise ValueError(
            "Upload: bytes() and path() are mutually exclusive"
        )
    if has_bytes and not b._filename:
        raise ValueError(
            "Upload: filename() is required when bytes() is set"
        )

    provider = Provider(
        name=b.client.provider.name,
        api_key=b.client.provider.api_key,
        headers=b.client.provider.headers,
    )
    if b.client.provider.base_url:
        provider.base_url = b.client.provider.base_url

    source: bytes | str = b._bytes if has_bytes else b._path
    kwargs: dict = {}
    if b._middleware:
        kwargs["middleware"] = list(b._middleware)
    if b._mime_type:
        kwargs["mime_type"] = b._mime_type
    if b._filename:
        kwargs["filename"] = b._filename

    return await asyncio.to_thread(_upload_file, provider, source, **kwargs)
