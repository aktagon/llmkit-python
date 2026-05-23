#

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_write: int = 0
    cache_read: int = 0
    reasoning: int = 0


class MiddlewarePhase(str, Enum):
    PRE = "pre"
    POST = "post"


class MiddlewareOp(str, Enum):
    LLM_REQUEST = "llm_request"
    TOOL_CALL = "tool_call"
    CACHE_CREATE = "cache_create"
    UPLOAD = "upload"
    BATCH_SUBMIT = "batch_submit"
    IMAGE_GENERATION = "image_generation"
    MODELS_LIST = "models_list"


@dataclass
class Event:
    """"""
    #
    op: MiddlewareOp = MiddlewareOp.LLM_REQUEST
    #
    phase: MiddlewarePhase = MiddlewarePhase.PRE
    #
    provider: str = ""
    #
    model: str = ""
    #
    tool: str = ""
    #
    args: dict[str, Any] = field(default_factory=dict)
    #
    result: str = ""
    #
    usage: Usage | None = None
    #
    err: str | None = None
    #
    duration: float = 0.0


#
#
MiddlewareFn = Callable[[Event], Exception | None]
