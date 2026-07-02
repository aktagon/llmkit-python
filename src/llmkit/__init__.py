"""llmkit — unified LLM client. One API, many providers, zero dependencies.

Quick start::

    import llmkit
    c = llmkit.anthropic(api_key)
    resp = await c.text.system("...").temperature(0.7).prompt("hello")

Or via the explicit subpackage::

    from llmkit.builders import new_client, anthropic, openai, google

The typed builder is the only public surface as of v1.0.0. Imports at
the top level bring in the per-provider factories, the `Client` type,
and the four builder classes (`Text`, `Image`, `Agent`, `Upload`) so
`import llmkit` is enough for the common case.
"""

from __future__ import annotations

from .batch import BatchHandle
from .client import Responses
from .builders import (
    Agent,
    Client,
    Image,
    Music,
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
    MediaRef,
    Part,
    audio,
    audio_bytes,
)
from .music import MusicRequest
from .structs import (
    AudioData,
    ImageResponse,
    MusicResponse,
    ToolCall,
    ToolResult,
    TranscriptionResponse,
    TranscriptSegment,
)
from .wire import (
    MissingWireVersionError,
    UnknownWireKeyError,
    UnsupportedWireVersionError,
    load_history,
    save_history,
)
from .wire_version import WIRE_SCHEMA_VERSION
from .providers.generated.middleware import (
    Event,
    MiddlewareFn,
    MiddlewareOp,
    MiddlewarePhase,
    Usage,
)
from .providers.generated.providers import ProviderName
from .telemetry import Telemetry, with_telemetry
from .types import (
    Capability,
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
    "Music",
    "Agent",
    "Upload",
    "BatchHandle",
    # Conversation / response types.
    "File",
    "Message",
    "Response",
    "Tool",
    "ToolCall",
    "ToolResult",
    # Chat protocol opt-in token (ADR-055).
    "Responses",
    # Wire format (ADR-023).
    "save_history",
    "load_history",
    "WIRE_SCHEMA_VERSION",
    "MissingWireVersionError",
    "UnknownWireKeyError",
    "UnsupportedWireVersionError",
    "ImageData",
    "ImageRequest",
    "ImageResponse",
    "AudioData",
    "MusicRequest",
    "MusicResponse",
    "MediaRef",
    "Part",
    # Audio Part constructors + transcription containers (ADR-048).
    "audio",
    "audio_bytes",
    "TranscriptionResponse",
    "TranscriptSegment",
    # Middleware.
    "Event",
    "MiddlewareFn",
    "MiddlewareOp",
    "MiddlewarePhase",
    "Usage",
    # Telemetry (ADR-054, opt-in).
    "Telemetry",
    "with_telemetry",
    # Errors.
    "APIError",
    "MiddlewareVetoError",
    "ValidationError",
    # Provider identity (for use with `new_client`). The internal 37-field
    # wire/transform spec (ProviderSpec / PROVIDERS) is NOT public surface
    # (ADR-038 PMD-004); read provider metadata via the `providers` namespace —
    # `from llmkit import providers; providers.info(name)` / `providers.list()`.
    "ProviderName",
    # Capability vocabulary (ADR-019 catalogue filter + ADR-030
    # Client.supports query).
    "Capability",
    # Codegen-runtime types — kept exposed for the contract-level test
    # surface and for callers writing custom transports. Typical v1.0.0
    # callers should not need these; use the typed builder.
    "InputImage",
    "Options",
    "Provider",
    "Request",
]
