"""llmkit — unified LLM client. One API, many providers, zero dependencies.

Public surface:

    from llmkit.builders import new_client
    c = new_client("anthropic", api_key)
    resp = await c.text.system("...").temperature(0.7).prompt("hello")

Plan-018 D3.x absorbed the legacy free-function layer (``prompt``,
``prompt_stream``, ``generate_image``, ``upload_file``, batch trio,
``Agent``) into typed-builder terminals; the public surface here is
types + error classes + the ``Providers`` enum + the ``BatchHandle``
class returned by ``c.text.submit_batch(...)``.
"""

from __future__ import annotations

from .batch import BatchHandle
from .errors import APIError, MiddlewareVetoError, ValidationError
from .image import (
    ImageData,
    ImageRequest,
    ImageResponse,
    MediaRef,
    Part,
)
from .providers.generated.middleware import (
    Event,
    MiddlewareFn,
    MiddlewareOp,
    MiddlewarePhase,
    Usage,
)
from .providers.generated.providers import PROVIDERS, ProviderConfig, ProviderName
from .types import File, InputImage, Message, Options, Provider, Request, Response, Tool

__all__ = [
    "APIError",
    "BatchHandle",
    "Event",
    "File",
    "ImageData",
    "ImageRequest",
    "ImageResponse",
    "InputImage",
    "MediaRef",
    "Part",
    "Message",
    "MiddlewareFn",
    "MiddlewareOp",
    "MiddlewarePhase",
    "MiddlewareVetoError",
    "Options",
    "PROVIDERS",
    "Provider",
    "ProviderConfig",
    "ProviderName",
    "Request",
    "Response",
    "Tool",
    "Usage",
    "ValidationError",
]
