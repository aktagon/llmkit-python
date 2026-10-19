# Code generated — DO NOT EDIT.

from __future__ import annotations

from dataclasses import dataclass, field

from .providers import ProviderName


# Speech wire-shape discriminators. Each fully determines the request
# body and the response audio path.


@dataclass(frozen=True)
class SpeechModelDef:
    model_id: str
    label: str
    output_mime: str = ""
    # ADR-037 advisory fact (queryable, never gated); 0 when unverified.
    sample_rate_hz: int = 0
    # Speakers the model takes (1 or 2); None = unknown, which sends (decision 009).
    max_speakers: int | None = None
    # Whether the provider accepts a style (style, add_styled_line); None =
    # unknown, only False blocks (decision 009).
    supports_speech_style: bool | None = None


@dataclass(frozen=True)
class SpeechGenDef:
    # wire_shape is SpeechInworld | SpeechOpenAI | SpeechGemini | SpeechMiniMax | SpeechXAI.
    wire_shape: str
    # audio_response_encoding selects the response decoder: "base64Envelope" | "rawBody" | "base64InteractionSteps" | "hexEnvelope" (ADR-051).
    audio_response_encoding: str = "base64Envelope"
    # requested_output_format is the format token the body sends (mp3, LINEAR16); "" = the body sends no format (HANDOFF-098).
    requested_output_format: str = ""
    # gen_endpoint is an override; empty = use provider main endpoint.
    gen_endpoint: str = ""
    models: tuple[SpeechModelDef, ...] = field(default_factory=tuple)
    # base_url is the speech API base when it differs from the chat base; "" = use chat base.
    base_url: str = ""
    # llmkit's body for this config can carry a style; False rejects every style (HANDOFF-099).
    carries_speech_style: bool = False
    # The config has a multi-speaker body; False rejects every speaker request (HANDOFF-099).
    multi_speaker: bool = False


_SPEECH_GEN: dict[ProviderName, SpeechGenDef] = {
    ProviderName.GOOGLE: SpeechGenDef(
        wire_shape="SpeechGemini",
        audio_response_encoding="base64InteractionSteps",
        requested_output_format="",
        gen_endpoint="/v1beta/interactions",
        models=(
            SpeechModelDef(
                model_id="gemini-3.1-flash-tts-preview",
                label="Gemini 3.1 Flash TTS Preview",
                output_mime="audio/l16",
                sample_rate_hz=0,
                max_speakers=1,
                supports_speech_style=False,
            ),
            SpeechModelDef(
                model_id="gemini-3.8-flash-lite-tts",
                label="Gemini 3.8 Flash-Lite TTS",
                output_mime="audio/wav",
                sample_rate_hz=24000,
                max_speakers=2,
                supports_speech_style=True,
            ),
            SpeechModelDef(
                model_id="gemini-3.8-flash-tts",
                label="Gemini 3.8 Flash TTS",
                output_mime="audio/wav",
                sample_rate_hz=24000,
                max_speakers=2,
                supports_speech_style=True,
            ),
        ),
        base_url="",
        carries_speech_style=True,
        multi_speaker=True,
    ),
    ProviderName.GROK: SpeechGenDef(
        wire_shape="SpeechXAI",
        audio_response_encoding="rawBody",
        requested_output_format="mp3",
        gen_endpoint="/v1/tts",
        models=(
            SpeechModelDef(
                model_id="grok-tts",
                label="Grok TTS",
                output_mime="audio/mpeg",
                sample_rate_hz=24000,
                max_speakers=None,
                supports_speech_style=True,
            ),
        ),
        base_url="",
        carries_speech_style=False,
        multi_speaker=False,
    ),
    ProviderName.INWORLD: SpeechGenDef(
        wire_shape="SpeechInworld",
        audio_response_encoding="base64Envelope",
        requested_output_format="LINEAR16",
        gen_endpoint="/tts/v1/voice",
        models=(
            SpeechModelDef(
                model_id="inworld-tts-1.5-max",
                label="Inworld TTS 1.5 Max",
                output_mime="audio/wav",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=False,
            ),
            SpeechModelDef(
                model_id="inworld-tts-1.5-mini",
                label="Inworld TTS 1.5 Mini",
                output_mime="audio/wav",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=False,
            ),
            SpeechModelDef(
                model_id="inworld-tts-2",
                label="Inworld TTS 2",
                output_mime="audio/wav",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=True,
            ),
        ),
        base_url="",
        carries_speech_style=False,
        multi_speaker=False,
    ),
    ProviderName.MINIMAX: SpeechGenDef(
        wire_shape="SpeechMiniMax",
        audio_response_encoding="hexEnvelope",
        requested_output_format="",
        gen_endpoint="/v1/t2a_v2",
        models=(
            SpeechModelDef(
                model_id="speech-2.6-hd",
                label="MiniMax Speech 2.6 HD",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=True,
            ),
            SpeechModelDef(
                model_id="speech-2.6-turbo",
                label="MiniMax Speech 2.6 Turbo",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=True,
            ),
            SpeechModelDef(
                model_id="speech-2.8-hd",
                label="MiniMax Speech 2.8 HD",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=True,
            ),
            SpeechModelDef(
                model_id="speech-2.8-turbo",
                label="MiniMax Speech 2.8 Turbo",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=True,
            ),
        ),
        base_url="https://api.minimax.io",
        carries_speech_style=False,
        multi_speaker=False,
    ),
    ProviderName.OPENAI: SpeechGenDef(
        wire_shape="SpeechOpenAI",
        audio_response_encoding="rawBody",
        requested_output_format="mp3",
        gen_endpoint="/v1/audio/speech",
        models=(
            SpeechModelDef(
                model_id="gpt-4o-mini-tts",
                label="GPT-4o mini TTS",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=True,
            ),
            SpeechModelDef(
                model_id="tts-1",
                label="TTS 1",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=False,
            ),
            SpeechModelDef(
                model_id="tts-1-hd",
                label="TTS 1 HD",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=False,
            ),
        ),
        base_url="",
        carries_speech_style=True,
        multi_speaker=False,
    ),
    ProviderName.OPENROUTER: SpeechGenDef(
        wire_shape="SpeechOpenAI",
        audio_response_encoding="rawBody",
        requested_output_format="mp3",
        gen_endpoint="/v1/audio/speech",
        models=(
            SpeechModelDef(
                model_id="elevenlabs/eleven-v3",
                label="Eleven v3",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=1,
                supports_speech_style=False,
            ),
            SpeechModelDef(
                model_id="elevenlabs/eleven-v4",
                label="Eleven v4",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=1,
                supports_speech_style=False,
            ),
            SpeechModelDef(
                model_id="elevenlabs/eleven-v4-turbo",
                label="Eleven v4 Turbo",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=1,
                supports_speech_style=False,
            ),
            SpeechModelDef(
                model_id="fish-audio/drama-3",
                label="Fish Audio Drama 3",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=True,
            ),
            SpeechModelDef(
                model_id="fish-audio/s2.1-pro",
                label="Fish Audio S2.1 Pro",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=True,
            ),
            SpeechModelDef(
                model_id="google/gemini-3.1-flash-tts-preview",
                label="Gemini 3.1 Flash TTS Preview",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=True,
            ),
            SpeechModelDef(
                model_id="google/gemini-3.8-flash-lite-tts",
                label="Gemini 3.8 Flash Lite TTS",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=True,
            ),
            SpeechModelDef(
                model_id="google/gemini-3.8-flash-tts",
                label="Gemini 3.8 Flash TTS",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=None,
                supports_speech_style=True,
            ),
            SpeechModelDef(
                model_id="hexgrad/kokoro-82m",
                label="Kokoro 82M",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=1,
                supports_speech_style=False,
            ),
            SpeechModelDef(
                model_id="x-ai/grok-voice-tts-1.0",
                label="Grok Voice TTS 1.0",
                output_mime="audio/mpeg",
                sample_rate_hz=0,
                max_speakers=1,
                supports_speech_style=False,
            ),
        ),
        base_url="",
        carries_speech_style=False,
        multi_speaker=False,
    ),
}


def speech_gen_config(provider: ProviderName) -> SpeechGenDef | None:
    return _SPEECH_GEN.get(provider)
