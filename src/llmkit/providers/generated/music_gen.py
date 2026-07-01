#

from __future__ import annotations

from dataclasses import dataclass, field

from .providers import ProviderName


#
#


@dataclass(frozen=True)
class MusicModelDef:
    model_id: str
    label: str
    supports_lyrics: bool = False
    max_duration_seconds: int = 0
    output_mime: str = ""
    #
    #
    sample_rate_hz: int = 0
    available_output_formats: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MusicGenDef:
    #
    wire_shape: str
    #
    gen_endpoint: str = ""
    models: tuple[MusicModelDef, ...] = field(default_factory=tuple)


_MUSIC_GEN: dict[ProviderName, MusicGenDef] = {
    ProviderName.GOOGLE: MusicGenDef(
        wire_shape="MusicGenerateContent",
        gen_endpoint="",
        models=(
            MusicModelDef(
                model_id="lyria-3-clip-preview",
                label="Lyria 3 Clip",
                supports_lyrics=True,
                max_duration_seconds=30,
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                available_output_formats=("audio/mpeg",),
            ),
            MusicModelDef(
                model_id="lyria-3-pro-preview",
                label="Lyria 3 Pro",
                supports_lyrics=True,
                max_duration_seconds=120,
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                available_output_formats=("audio/mpeg",),
            ),
        ),
    ),
    ProviderName.MINIMAX: MusicGenDef(
        wire_shape="MusicMinimax",
        gen_endpoint="https://api.minimax.io/v1/music_generation",
        models=(
            MusicModelDef(
                model_id="music-2.6",
                label="MiniMax Music 2.6",
                supports_lyrics=True,
                max_duration_seconds=0,
                output_mime="audio/mpeg",
                sample_rate_hz=44100,
                available_output_formats=("audio/mpeg", "audio/wav"),
            ),
        ),
    ),
    ProviderName.VERTEX: MusicGenDef(
        wire_shape="MusicPredict",
        gen_endpoint="",
        models=(
            MusicModelDef(
                model_id="lyria-002",
                label="Lyria 2",
                supports_lyrics=False,
                max_duration_seconds=30,
                output_mime="audio/wav",
                sample_rate_hz=48000,
                available_output_formats=("audio/wav",),
            ),
        ),
    ),
}


def music_gen_config(provider: ProviderName) -> MusicGenDef | None:
    return _MUSIC_GEN.get(provider)
