"""Core public types: Provider, Request, Response, Message, File, InputImage, Tool, Options."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .providers.generated.middleware import MiddlewareFn, Usage


@dataclass
class Provider:
    name: str
    api_key: str
    model: str = ""
    base_url: str = ""


@dataclass
class Message:
    role: str
    content: str


@dataclass
class File:
    id: str = ""
    uri: str = ""
    mime_type: str = ""
    name: str = ""


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
class Request:
    system: str = ""
    user: str = ""
    messages: list[Message] = field(default_factory=list)
    schema: str = ""
    files: list[File] = field(default_factory=list)
    images: list[InputImage] = field(default_factory=list)


@dataclass
class Response:
    text: str = ""
    tokens: Usage = field(default_factory=Usage)


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
