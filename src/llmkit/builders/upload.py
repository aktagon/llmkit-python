"""Phase 3 slice 2a — wires Upload.run against legacy ``upload_file``.

Python legacy ``upload_file(provider, path, ...)`` is path-based
(matching Go, inverse of TS). So in Python the Path branch is the
wired path and Bytes is deferred — symmetric to Go's slice 2a where
Path is wired and Bytes deferred.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..client import upload_file as legacy_upload_file
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
    if has_bytes:
        raise ValueError(
            "Upload: bytes() not yet wired (Python phase 3 follow-up); "
            "use path() for now"
        )

    provider = Provider(
        name=b.client.provider.name,
        api_key=b.client.provider.api_key,
    )
    if b.client.provider.base_url:
        provider.base_url = b.client.provider.base_url

    kwargs: dict = {}
    if b._middleware:
        kwargs["middleware"] = list(b._middleware)

    return await asyncio.to_thread(
        legacy_upload_file, provider, b._path, **kwargs
    )
