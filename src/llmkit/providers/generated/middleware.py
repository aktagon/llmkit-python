# Code generated — DO NOT EDIT.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


@dataclass
class Usage:
    """





"""

    input: int | None = None
    output: int | None = None
    cache_write: int | None = None
    cache_read: int | None = None
    reasoning: int | None = None
    #
    cost: float | None = None


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
    MUSIC_GENERATION = "music_generation"
    VIDEO_GENERATION = "video_generation"
    MODELS_LIST = "models_list"
    SPEECH_GENERATION = "speech_generation"
    TRANSCRIPTION = "transcription"


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
    err_type: str = ""
    #
    duration: float = 0.0


#
#
MiddlewareFn = Callable[[Event], Exception | None]
