"""llmkit — unified LLM client. One API, many providers, zero dependencies.

Public surface:

    import llmkit
    resp = llmkit.prompt(
        provider=llmkit.Provider(name="anthropic", api_key=key),
        request=llmkit.Request(system="...", user="hello"),
        temperature=0.7,
    )
    print(resp.text)
"""

from __future__ import annotations

from .agent import Agent
from .batch import BatchHandle, prompt_batch, submit_batch, wait_batch
from .client import StreamCallback, prompt, prompt_stream, upload_file
from .errors import APIError, MiddlewareVetoError, ValidationError
from .providers.generated.middleware import (
    Event,
    MiddlewareFn,
    MiddlewareOp,
    MiddlewarePhase,
    Usage,
)
from .providers.generated.providers import PROVIDERS, ProviderConfig, ProviderName
from .types import File, Image, Message, Options, Provider, Request, Response, Tool

__all__ = [
    "APIError",
    "Agent",
    "BatchHandle",
    "Event",
    "File",
    "Image",
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
    "StreamCallback",
    "Tool",
    "Usage",
    "ValidationError",
    "prompt",
    "prompt_batch",
    "prompt_stream",
    "submit_batch",
    "upload_file",
    "wait_batch",
]
