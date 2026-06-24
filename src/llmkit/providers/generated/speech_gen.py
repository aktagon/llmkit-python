# Code generated — DO NOT EDIT.

from __future__ import annotations

from dataclasses import dataclass, field

from .providers import ProviderName


#
#


@dataclass(frozen=True)
class SpeechModelDef:
    model_id: str
    label: str
    output_mime: str = ""
    #
    sample_rate_hz: int = 0


@dataclass(frozen=True)
class SpeechGenDef:
    #
    wire_shape: str
    #
    gen_endpoint: str = ""
    #
    voices: tuple[str, ...] = field(default_factory=tuple)
    models: tuple[SpeechModelDef, ...] = field(default_factory=tuple)


_SPEECH_GEN: dict[ProviderName, SpeechGenDef] = {
    ProviderName.INWORLD: SpeechGenDef(
        wire_shape="SpeechInworld",
        gen_endpoint="/tts/v1/voice",
        voices=("Alex", "Ashley", "Dennis"),
        models=(
            SpeechModelDef(
                model_id="inworld-tts-1.5-max",
                label="Inworld TTS 1.5 Max",
                output_mime="audio/wav",
                sample_rate_hz=0,
            ),
            SpeechModelDef(
                model_id="inworld-tts-1.5-mini",
                label="Inworld TTS 1.5 Mini",
                output_mime="audio/wav",
                sample_rate_hz=0,
            ),
            SpeechModelDef(
                model_id="inworld-tts-2",
                label="Inworld TTS 2",
                output_mime="audio/wav",
                sample_rate_hz=0,
            ),
        ),
    ),
}


def speech_gen_config(provider: ProviderName) -> SpeechGenDef | None:
    return _SPEECH_GEN.get(provider)
