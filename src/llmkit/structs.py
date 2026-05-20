# Code generated — DO NOT EDIT.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .providers.generated.middleware import Usage

if TYPE_CHECKING:
    from .image import ImageData
    from .types import Provider


@dataclass(kw_only=True)
class BatchHandle:
    """"""
    #
    id: str = ""

    #
    provider: Provider

    #
    raw: bool = False


@dataclass
class ImageResponse:
    """"""
    #
    images: list[ImageData] = field(default_factory=list)

    #
    text: str = ""

    #
    usage: Usage = field(default_factory=Usage)

    #
    finish_reason: str = ""

    #
    finish_message: str = ""

    #
    raw: Any | None = None


@dataclass
class Response:
    """"""
    #
    text: str = ""

    #
    usage: Usage = field(default_factory=Usage)

    #
    finish_reason: str = ""

    #
    finish_message: str = ""

    #
    raw: Any | None = None
