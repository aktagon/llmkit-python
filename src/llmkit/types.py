"""Core public types: Provider, Request, Response, Message, File, InputImage, Tool, Options."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .providers.generated.middleware import MiddlewareFn
from .structs import File, Message, Response


@dataclass
class Provider:
    name: str
    api_key: str
    model: str = ""
    base_url: str = ""
    # Custom HTTP headers added via Client.add_header (ADR-052). Merged into
    # every request before the provider auth header and the static required
    # header, so a gateway header (e.g. cf-aig-authorization) rides alongside
    # the provider key without clobbering it.
    headers: dict[str, str] = field(default_factory=dict)


class Capability(str):
    """Capability identifier mirroring llm:Capability instances.

    Ontology-derived per ADR-019. ModelInfo.capabilities is ``list[Capability]``;
    subclasses ``str`` so existing string-comparison sites keep working.
    """

    CHAT_COMPLETION = "chat_completion"
    IMAGE_GENERATION = "image_generation"
    TOOL_CALLING = "tool_calling"
    FILE_UPLOAD = "file_upload"
    BATCHING = "batching"
    CACHING = "caching"
    REASONING = "reasoning"
    CATALOGUE = "catalogue"


@dataclass
class InputImage:
    """Image attached to a text-generation request (vision input).

    Distinct from llmkit.Image() — that's the Part constructor used for
    image-generation calls. The two concepts target different capabilities;
    aligning text generation onto Part-based vocabulary is tracked
    separately (ADR-008 OQ-2).
    """

    url: str
    mime_type: str = ""
    detail: str = ""


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    run: Callable[[dict[str, Any]], str]


@dataclass
class SafetySetting:
    """Per-category content safety filter for Gemini providers."""
    category: str
    threshold: str


# Harm category constants for SafetySetting.category
HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"
HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"
HARM_CATEGORY_CIVIC_INTEGRITY = "HARM_CATEGORY_CIVIC_INTEGRITY"

# Harm block threshold constants for SafetySetting.threshold
HARM_BLOCK_THRESHOLD_NONE = "BLOCK_NONE"
HARM_BLOCK_THRESHOLD_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"
HARM_BLOCK_THRESHOLD_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
HARM_BLOCK_THRESHOLD_HIGH_ONLY = "BLOCK_ONLY_HIGH"

# Vertex Imagen safety filter threshold constants
IMAGE_SAFETY_FILTER_BLOCK_FEW = "block_few"
IMAGE_SAFETY_FILTER_BLOCK_SOME = "block_some"
IMAGE_SAFETY_FILTER_BLOCK_MOST = "block_most"
IMAGE_SAFETY_FILTER_BLOCK_ONLY_HIGH = "block_only_high"


@dataclass
class Request:
    system: str = ""
    user: str = ""
    messages: list[Message] = field(default_factory=list)
    schema: str = ""
    files: list[File] = field(default_factory=list)
    images: list[InputImage] = field(default_factory=list)


# Response and File are generated from the ontology (ADR-018, API-PDS-002)
# and re-exported above so existing `from llmkit.types import Response, File`
# imports keep working without touching every call site.


@dataclass
class Options:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] = field(default_factory=list)
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    thinking_budget: int | None = None
    reasoning_effort: str = ""
    max_tool_iterations: int = 10
    caching: bool = False
    cache_ttl: float = 0.0
    middleware: list[MiddlewareFn] = field(default_factory=list)
    request_timeout: float = 600.0
    safety_settings: list["SafetySetting"] = field(default_factory=list)
    # Opt-in: populate Response.raw with the parsed provider response body
    # (ADR-014). Plumbed by the typed-builder's .raw() chain method.
    raw: bool = False
