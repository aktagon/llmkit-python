# Code generated — DO NOT EDIT.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .caching import ResourceLifecycleDef
from .providers import ProviderName


# Batch contract constants shared by every SDK (ADR-091).
# Prefix + the request index is the id sent with each batch request.
BATCH_REQUEST_ID_PREFIX = "req-"
# finish_reason of a batch slot whose request has no result line.
BATCH_SLOT_MISSING = "missing"
# finish_reason of a failed batch slot when the provider gives no reason.
BATCH_SLOT_ERROR = "error"


class BatchInputMode(str, Enum):
    INLINE_REQUESTS = "InlineRequests"
    FILE_REFERENCE_INPUT = "FileReferenceInput"


@dataclass(frozen=True)
class BatchDef:
    input_mode: BatchInputMode
    input_field: str = ""
    file_purpose: str = ""
    request_wrapper: str = ""
    completion_window: str = ""
    endpoint_path: str = ""
    item_body_field: str = ""
    result_body_path: str = ""
    result_key_path: str = ""
    result_status_path: str = ""
    result_success_values: tuple[str, ...] = ()
    result_reason_paths: tuple[str, ...] = ()
    result_message_paths: tuple[str, ...] = ()
    request_count_paths: tuple[str, ...] = ()
    lifecycle: ResourceLifecycleDef | None = None


_BATCH: dict[ProviderName, BatchDef] = {
    ProviderName.ANTHROPIC: BatchDef(
        input_mode=BatchInputMode.INLINE_REQUESTS,
        input_field="",
        file_purpose="",
        request_wrapper="requests",
        completion_window="",
        endpoint_path="",
        item_body_field="params",
        result_body_path="result.message",
        result_key_path="custom_id",
        result_status_path="result.type",
        result_success_values=("succeeded",),
        result_reason_paths=("result.type",),
        result_message_paths=("result.error.error.message",),
        request_count_paths=("request_counts.processing", "request_counts.succeeded", "request_counts.errored", "request_counts.canceled", "request_counts.expired"),
        lifecycle=(
            ResourceLifecycleDef(
                create_endpoint="/v1/messages/batches",
                response_id_path="id",
                reference_field="",
                polling_endpoint="",
                polling_status_path="processing_status",
                polling_done_value="ended",
                polling_error_values=(),
                result_endpoint="/v1/messages/batches/{id}/results",
                result_response_path="",
                result_file_id_path="",
                error_file_id_path="",
                file_content_endpoint="",
            )
        ),
    ),
    ProviderName.GOOGLE: BatchDef(
        input_mode=BatchInputMode.INLINE_REQUESTS,
        input_field="",
        file_purpose="",
        request_wrapper="requests",
        completion_window="",
        endpoint_path="",
        item_body_field="",
        result_body_path="",
        result_key_path="",
        result_status_path="",
        result_success_values=(),
        result_reason_paths=(),
        result_message_paths=(),
        request_count_paths=(),
        lifecycle=None,
    ),
    ProviderName.OPENAI: BatchDef(
        input_mode=BatchInputMode.FILE_REFERENCE_INPUT,
        input_field="input_file_id",
        file_purpose="batch",
        request_wrapper="",
        completion_window="24h",
        endpoint_path="/v1/chat/completions",
        item_body_field="",
        result_body_path="response.body",
        result_key_path="custom_id",
        result_status_path="response.status_code",
        result_success_values=("200",),
        result_reason_paths=("error.code", "response.body.error.code"),
        result_message_paths=("error.message", "response.body.error.message"),
        request_count_paths=("request_counts.total",),
        lifecycle=(
            ResourceLifecycleDef(
                create_endpoint="/v1/batches",
                response_id_path="id",
                reference_field="",
                polling_endpoint="",
                polling_status_path="status",
                polling_done_value="completed",
                polling_error_values=("failed", "expired", "cancelled"),
                result_endpoint="",
                result_response_path="",
                result_file_id_path="output_file_id",
                error_file_id_path="error_file_id",
                file_content_endpoint="/v1/files/{id}/content",
            )
        ),
    ),
}


def batch_config(provider: ProviderName) -> BatchDef | None:
    return _BATCH.get(provider)
