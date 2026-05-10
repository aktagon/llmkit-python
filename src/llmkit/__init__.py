"""llmkit — unified LLM client. One API, many providers, zero dependencies.

Quick start::

    import llmkit
    c = llmkit.anthropic(api_key)
    resp = await c.text().system("...").temperature(0.7).prompt("hello")

Or via the explicit subpackage::

    from llmkit.builders import new_client, anthropic, openai, google

The typed builder is the only public surface as of v1.0.0. Imports at
the top level bring in the per-provider factories, the `Client` type,
and the four builder classes (`Text`, `Image`, `Agent`, `Upload`) so
`import llmkit` is enough for the common case.
"""

from __future__ import annotations

from .batch import BatchHandle
from .builders import (
    Agent,
    Client,
    Image,
    Text,
    Upload,
    ai21,
    anthropic,
    azure,
    bedrock,
    cerebras,
    cohere,
    deepseek,
    doubao,
    ernie,
    fireworks,
    google,
    grok,
    groq,
    lmstudio,
    minimax,
    mistral,
    moonshot,
    new_client,
    ollama,
    openai,
    openrouter,
    perplexity,
    qwen,
    sambanova,
    together,
    vllm,
    yi,
    zhipu,
)
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
from .types import (
    File,
    InputImage,
    Message,
    Options,
    Provider,
    Request,
    Response,
    Tool,
)

__all__ = [
    # Typed builder factories (the v1.0.0 entry points).
    "new_client",
    "ai21",
    "anthropic",
    "azure",
    "bedrock",
    "cerebras",
    "cohere",
    "deepseek",
    "doubao",
    "ernie",
    "fireworks",
    "google",
    "grok",
    "groq",
    "lmstudio",
    "minimax",
    "mistral",
    "moonshot",
    "ollama",
    "openai",
    "openrouter",
    "perplexity",
    "qwen",
    "sambanova",
    "together",
    "vllm",
    "yi",
    "zhipu",
    # Builder + result types.
    "Client",
    "Text",
    "Image",
    "Agent",
    "Upload",
    "BatchHandle",
    # Conversation / response types.
    "File",
    "Message",
    "Response",
    "Tool",
    "ImageData",
    "ImageRequest",
    "ImageResponse",
    "MediaRef",
    "Part",
    # Middleware.
    "Event",
    "MiddlewareFn",
    "MiddlewareOp",
    "MiddlewarePhase",
    "Usage",
    # Errors.
    "APIError",
    "MiddlewareVetoError",
    "ValidationError",
    # Provider enum + registry (for use with `new_client`).
    "ProviderName",
    "ProviderConfig",
    "PROVIDERS",
    # Codegen-runtime types — kept exposed for the contract-level test
    # surface and for callers writing custom transports. Typical v1.0.0
    # callers should not need these; use the typed builder.
    "InputImage",
    "Options",
    "Provider",
    "Request",
]
