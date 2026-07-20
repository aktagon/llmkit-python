""""""

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
    #
    #
    #
    #
    headers: dict[str, str] = field(default_factory=dict)


class Capability(str):
    """



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
    """






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
    """"""
    category: str
    threshold: str


#
HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"
HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"
HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"
HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"
HARM_CATEGORY_CIVIC_INTEGRITY = "HARM_CATEGORY_CIVIC_INTEGRITY"

#
HARM_BLOCK_THRESHOLD_NONE = "BLOCK_NONE"
HARM_BLOCK_THRESHOLD_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"
HARM_BLOCK_THRESHOLD_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"
HARM_BLOCK_THRESHOLD_HIGH_ONLY = "BLOCK_ONLY_HIGH"

#
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


#
#
#


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
    #
    #
    raw: bool = False
