#

from __future__ import annotations

from dataclasses import dataclass

from .providers import ProviderName


#
#


@dataclass(frozen=True)
class TranscriptionDef:
    #
    wire_shape: str
    submit_endpoint: str
    interaction: str = "async"  # "sync" | "async"
    request_encoding: str = "json"  # "json" | "multipart"
    poll_endpoint: str = ""  # template with {id}; async only
    submit_handle_field: str = ""  # dotted path to the handle id; async only
    status_path: str = ""  # dotted path to the poll status string; async only
    done_status: str = ""  # status value marking terminal success; async only
    error_status: str = ""  # status value marking terminal failure; async only
    upload_endpoint: str = ""  # local-bytes upload hop; "" = url-only / inline-bytes


_TRANSCRIPTION_GEN: dict[ProviderName, TranscriptionDef] = {
    ProviderName.ASSEMBLYAI: TranscriptionDef(
        wire_shape="TranscriptionAssemblyAI",
        submit_endpoint="/v2/transcript",
        interaction="async",
        request_encoding="json",
        poll_endpoint="/v2/transcript/{id}",
        submit_handle_field="id",
        status_path="status",
        done_status="completed",
        error_status="error",
        upload_endpoint="/v2/upload",
    ),
    ProviderName.OPENAI: TranscriptionDef(
        wire_shape="TranscriptionOpenAI",
        submit_endpoint="/v1/audio/transcriptions",
        interaction="sync",
        request_encoding="multipart",
        poll_endpoint="",
        submit_handle_field="",
        status_path="",
        done_status="",
        error_status="",
        upload_endpoint="",
    ),
}


def transcription_config(provider: ProviderName) -> TranscriptionDef | None:
    return _TRANSCRIPTION_GEN.get(provider)
