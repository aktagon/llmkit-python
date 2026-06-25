# Code generated — DO NOT EDIT.

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
    poll_endpoint: str  # template with {id}
    submit_handle_field: str  # dotted path to the handle id
    status_path: str  # dotted path to the poll status string
    done_status: str  # status value marking terminal success
    error_status: str  # status value marking terminal failure
    upload_endpoint: str = ""  # local-bytes upload hop; "" = url-only


_TRANSCRIPTION_GEN: dict[ProviderName, TranscriptionDef] = {
    ProviderName.ASSEMBLYAI: TranscriptionDef(
        wire_shape="TranscriptionAssemblyAI",
        submit_endpoint="/v2/transcript",
        poll_endpoint="/v2/transcript/{id}",
        submit_handle_field="id",
        status_path="status",
        done_status="completed",
        error_status="error",
        upload_endpoint="/v2/upload",
    ),
}


def transcription_config(provider: ProviderName) -> TranscriptionDef | None:
    return _TRANSCRIPTION_GEN.get(provider)
