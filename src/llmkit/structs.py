#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .providers.generated.middleware import Usage

if TYPE_CHECKING:
    from .types import Capability, Provider


@dataclass
class AudioData:
    """"""
    #
    mime_type: str = ""

    #
    bytes: bytes = b''


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
class File:
    """"""
    #
    id: str = ""

    #
    uri: str = ""

    #
    mime_type: str = ""

    #
    name: str = ""


@dataclass
class ImageData:
    """"""
    #
    mime_type: str = ""

    #
    bytes: bytes = b''


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
class LiveResult:
    """"""
    #
    models: list[ModelInfo] = field(default_factory=list)

    #
    errors: dict[str, ProviderError] = field(default_factory=dict)


@dataclass
class MediaRef:
    """"""
    #
    mime_type: str = ""

    #
    bytes: bytes = b''


@dataclass
class Message:
    """"""
    #
    role: str = ""

    #
    content: str = ""

    #
    tool_calls: list[ToolCall] = field(default_factory=list)

    #
    tool_result: ToolResult | None = None


@dataclass(kw_only=True)
class ModelInfo:
    """"""
    #
    id: str = ""

    #
    provider: Provider

    #
    capabilities: list[Capability] = field(default_factory=list)

    #
    display_name: str = ""

    #
    description: str = ""

    #
    context_window: int = 0

    #
    max_output: int = 0

    #
    created: int = 0

    #
    raw: Any | None = None


@dataclass
class MusicResponse:
    """"""
    #
    audio: list[AudioData] = field(default_factory=list)

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
class ProviderError:
    """"""
    #
    kind: str = ""

    #
    message: str = ""


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


@dataclass
class SpeechResponse:
    """"""
    #
    audio: AudioData = field(default_factory=AudioData)

    #
    usage: Usage = field(default_factory=Usage)

    #
    finish_reason: str = ""


@dataclass
class ToolCall:
    """"""
    #
    id: str = ""

    #
    name: str = ""

    #
    input: Any | None = None


@dataclass
class ToolResult:
    """"""
    #
    tool_use_id: str = ""

    #
    content: str = ""


@dataclass
class TranscriptSegment:
    """"""
    #
    text: str = ""

    #
    start: int = 0

    #
    end: int = 0

    #
    speaker: str = ""


@dataclass(kw_only=True)
class TranscriptionHandle:
    """"""
    #
    id: str = ""

    #
    provider: Provider


@dataclass
class TranscriptionResponse:
    """"""
    #
    text: str = ""

    #
    segments: list[TranscriptSegment] = field(default_factory=list)

    #
    usage: Usage = field(default_factory=Usage)


@dataclass
class VideoData:
    """"""
    #
    mime_type: str = ""

    #
    url: str = ""

    #
    bytes: bytes = b''

    #
    duration_seconds: int = 0


@dataclass(kw_only=True)
class VideoHandle:
    """"""
    #
    id: str = ""

    #
    provider: Provider

    #
    raw: bool = False

    #
    model: str = ""


@dataclass
class VideoResponse:
    """"""
    #
    videos: list[VideoData] = field(default_factory=list)

    #
    usage: Usage = field(default_factory=Usage)

    #
    finish_reason: str = ""

    #
    finish_message: str = ""

    #
    raw: Any | None = None
