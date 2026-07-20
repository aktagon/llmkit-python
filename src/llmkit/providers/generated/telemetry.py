#


from .middleware import MiddlewareOp

#
#
#

TELEMETRY_SEMCONV_VERSION = "1.29.0"
TELEMETRY_TRACES_PATH = "/v1/traces"
TELEMETRY_ENDPOINT_REQUIRED = True
TELEMETRY_CAPTURE_CONTENT_DEFAULT = False

#
OTEL_ATTR_OP = "gen_ai.operation.name"  # Event.op
OTEL_ATTR_PROVIDER = "gen_ai.system"  # Event.provider
OTEL_ATTR_MODEL = "gen_ai.request.model"  # Event.model
OTEL_ATTR_ERR_TYPE = "error.type"  # Event.err_type

#
OTEL_USAGE_INPUT = "gen_ai.usage.input_tokens"
OTEL_USAGE_OUTPUT = "gen_ai.usage.output_tokens"

#
#
TELEMETRY_OPERATION_NAME: dict[MiddlewareOp, str] = {
    MiddlewareOp.LLM_REQUEST: "chat",
    MiddlewareOp.TOOL_CALL: "execute_tool",
}
