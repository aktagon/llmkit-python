"""Multi-turn Agent with tool-calling loop. Mirrors go/agent.go."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .errors import APIError, ValidationError, parse_error
from .http import do_post, do_sigv4_post
from .middleware import fire_post, fire_pre, resolve_model
from .paths import extract_float_path, extract_int_path, extract_path
from .providers.generated.middleware import Event, MiddlewareOp, Usage
from .providers.generated.providers import PROVIDERS, ProviderName
from .providers.generated.request import AuthScheme, SystemPlacement, auth_scheme, system_placement, tool_call_config
from .transforms import (
    map_role,
    select_message_transform,
    select_tool_call_extractor,
    select_tool_call_transform,
    select_tool_def_transform,
    select_tool_result_transform,
)
from .types import Message, Options, Provider, Request, Response, Tool


@dataclass
class _InternalMessage:
    role: str = ""
    content: str = ""
    tool_calls: list = field(default_factory=list)   # list[ToolCall]
    tool_result: Any = None                           # ToolResult | None


class Agent:
    """Multi-turn conversation manager with optional tool calling."""

    def __init__(
        self,
        provider: Provider,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        thinking_budget: int | None = None,
        reasoning_effort: str = "",
        caching: bool = False,
        max_tool_iterations: int = 10,
        middleware: list | None = None,
        safety_settings: list | None = None,
        request_timeout: float = 600.0,
        raw: bool = False,
    ) -> None:
        self.provider = provider
        self.opts = Options(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop_sequences=list(stop_sequences or []),
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            thinking_budget=thinking_budget,
            reasoning_effort=reasoning_effort,
            caching=caching,
            max_tool_iterations=max_tool_iterations,
            middleware=list(middleware or []),
            safety_settings=list(safety_settings or []),
            request_timeout=request_timeout,
            raw=raw,
        )
        self.tools: list[Tool] = []
        self.history: list[_InternalMessage] = []
        self.system = ""

    def set_system(self, system: str) -> None:
        self.system = system

    def add_tool(self, tool: Tool) -> None:
        self.tools.append(tool)

    def reset(self) -> None:
        self.history = []
        self.tools = []

    def chat(self, msg: str) -> Response:
        """Send a message, execute any tool calls the LLM requests, and return the final response."""
        self.history.append(_InternalMessage(role="user", content=msg))
        return self._run_tool_loop()

    def _run_tool_loop(self) -> Response:
        from .client import _build_url  # noqa: F401

        cfg = PROVIDERS.get(self.provider.name)
        if cfg is None:
            raise ValidationError(field="provider", message=f"unknown: {self.provider.name}")

        tc_cfg = tool_call_config(ProviderName(self.provider.name))
        tc_call_transform = select_tool_call_transform(cfg)
        tc_result_transform = select_tool_result_transform(cfg)
        tc_extractor = select_tool_call_extractor(cfg)

        total_usage = Usage()

        for _ in range(self.opts.max_tool_iterations):
            body, headers = self._build_agent_request(cfg)

            # Caching is a shared request-construction step (ADR-026): applied
            # on every send path by construction, like Text/batch. Before this,
            # a .caching() agent silently paid full input price (BUG-004).
            if self.opts.caching:
                from .caching import apply_caching

                apply_caching(body, self.provider, self.opts, cfg)

            llm_event = Event(
                op=MiddlewareOp.LLM_REQUEST,
                provider=self.provider.name,
                model=resolve_model(self.provider.model, cfg),
            )
            llm_start = time.monotonic()
            fire_pre(self.opts.middleware, llm_event)

            json_body = json.dumps(body).encode("utf-8")
            url = _build_url(self.provider, cfg)

            try:
                if auth_scheme(ProviderName(self.provider.name)) == AuthScheme.SIG_V4:
                    region = os.environ.get(cfg.region_env_var, "")
                    secret_key = os.environ.get(cfg.secret_key_env_var, "")
                    session_token = os.environ.get(cfg.session_token_env_var, "")
                    resp_body = do_sigv4_post(
                        url,
                        json_body,
                        self.provider.api_key,
                        secret_key,
                        session_token,
                        region,
                        cfg.service_name,
                        timeout=self.opts.request_timeout,
                    )
                else:
                    resp_body = do_post(url, json_body, headers, timeout=self.opts.request_timeout)
            except APIError as raw_api_err:
                err = parse_error(self.provider.name, raw_api_err.status_code, raw_api_err.message.encode("utf-8"), None)
                _fire_post_err(self.opts.middleware, llm_event, err, llm_start)
                raise err from raw_api_err
            except Exception as exc:
                _fire_post_err(self.opts.middleware, llm_event, exc, llm_start)
                raise

            try:
                raw = json.loads(resp_body)
            except ValueError as exc:
                _fire_post_err(self.opts.middleware, llm_event, exc, llm_start)
                raise

            input_path = cfg.usage_input_path
            output_path = cfg.usage_output_path
            turn_input = extract_int_path(raw, input_path)
            turn_output = extract_int_path(raw, output_path)
            turn_cost = extract_float_path(raw, cfg.usage_cost_path) * cfg.usage_cost_scale if cfg.usage_cost_path else 0.0
            total_usage.input += turn_input
            total_usage.output += turn_output
            total_usage.cost += turn_cost

            post_ev = Event(
                op=MiddlewareOp.LLM_REQUEST,
                provider=self.provider.name,
                model=resolve_model(self.provider.model, cfg),
                usage=Usage(input=turn_input, output=turn_output),
                duration=time.monotonic() - llm_start,
            )
            fire_post(self.opts.middleware, post_ev)

            calls = tc_extractor(raw, tc_cfg)

            if not calls:
                text = extract_path(raw, cfg.response_text_path)
                self.history.append(_InternalMessage(role="assistant", content=text))
                finish_reason = extract_path(raw, cfg.finish_reason_path) if cfg.finish_reason_path else ""
                finish_message = extract_path(raw, cfg.finish_message_path) if cfg.finish_message_path else ""
                return Response(
                    text=text,
                    usage=total_usage,
                    finish_reason=finish_reason,
                    finish_message=finish_message,
                    raw=raw if self.opts.raw else None,
                )

            self.history.append(_InternalMessage(role="assistant", tool_calls=list(calls)))

            from .structs import ToolResult

            for tc in calls:
                # ADR-020 widened ToolCall.input to Any | None. Tool authors'
                # run() callback still receives the dict shape they registered,
                # so coerce non-dicts (None, primitives, lists) to {} here.
                tc_args = tc.input if isinstance(tc.input, dict) else {}
                tool = self._find_tool(tc.name)
                if tool is None:
                    result = f"error: unknown tool {tc.name!r}"
                    self.history.append(
                        _InternalMessage(
                            role="tool_result",
                            tool_result=ToolResult(tool_use_id=tc.id, content=result),
                        )
                    )
                    continue

                tool_ev = Event(
                    op=MiddlewareOp.TOOL_CALL,
                    provider=self.provider.name,
                    model=resolve_model(self.provider.model, cfg),
                    tool=tc.name,
                    args=dict(tc_args),
                )
                tool_start = time.monotonic()
                fire_pre(self.opts.middleware, tool_ev)

                run_err: BaseException | None = None
                try:
                    output = tool.run(tc_args)
                except BaseException as exc:
                    run_err = exc
                    output = f"error: {exc}"

                post_ev = Event(
                    op=MiddlewareOp.TOOL_CALL,
                    provider=self.provider.name,
                    model=resolve_model(self.provider.model, cfg),
                    tool=tc.name,
                    args=dict(tc_args),
                    result=output,
                    err=(str(run_err) if run_err else ""),
                    duration=time.monotonic() - tool_start,
                )
                fire_post(self.opts.middleware, post_ev)

                self.history.append(
                    _InternalMessage(
                        role="tool_result",
                        tool_result=ToolResult(tool_use_id=tc.id, content=output),
                    )
                )

        raise APIError(
            provider=self.provider.name,
            message=f"max tool iterations ({self.opts.max_tool_iterations}) reached",
            status_code=0,
        )

    def _build_agent_request(self, cfg) -> tuple[dict[str, Any], dict[str, str]]:
        from .client import _add_options, _build_url, _resolve_option_key  # noqa: F401
        from .providers.generated.options import OptionKey, supported_options

        body: dict[str, Any] = {}
        headers: dict[str, str] = {}

        model = self.provider.model or cfg.default_model
        if cfg.model_in_body:
            body["model"] = model

        max_tokens = cfg.default_max_tokens
        if self.opts.max_tokens is not None:
            max_tokens = self.opts.max_tokens

        supported = {o.key: o for o in supported_options(ProviderName(self.provider.name))}

        placement = system_placement(ProviderName(self.provider.name))
        if placement == SystemPlacement.TOP_LEVEL_FIELD and self.system:
            body["system"] = self.system
        elif placement == SystemPlacement.SIBLING_OBJECT and self.system:
            body["system_instruction"] = {"parts": [{"text": self.system}]}

        msg_transform = select_message_transform(cfg)
        tc_call_transform = select_tool_call_transform(cfg)
        tc_result_transform = select_tool_result_transform(cfg)

        self._build_history_messages(body, cfg, msg_transform, tc_call_transform, tc_result_transform)

        if self.tools:
            tool_def_transform = select_tool_def_transform(cfg)
            tool_def_transform(body, self.tools)

        from .paths import set_nested_field

        pname = ProviderName(self.provider.name)
        max_json_key = _resolve_option_key(pname, model, OptionKey.MAX_TOKENS, supported)
        if cfg.wraps_options_in:
            opt_body: dict[str, Any] = {}
            _add_options(opt_body, self.opts, self.provider.name, model)
            if max_json_key is not None:
                set_nested_field(opt_body, max_json_key, max_tokens)
            if opt_body:
                body[cfg.wraps_options_in] = opt_body
        else:
            if max_json_key is not None:
                set_nested_field(body, max_json_key, max_tokens)
            _add_options(body, self.opts, self.provider.name, model)

        scheme = auth_scheme(ProviderName(self.provider.name))
        if scheme == AuthScheme.BEARER_TOKEN:
            headers[cfg.auth_header] = cfg.auth_prefix + " " + self.provider.api_key
        elif scheme == AuthScheme.HEADER_API_KEY:
            headers[cfg.auth_header] = self.provider.api_key
        if cfg.required_header:
            headers[cfg.required_header] = cfg.required_header_value

        return body, headers

    def _build_history_messages(
        self,
        body: dict[str, Any],
        cfg,
        msg_transform,
        tc_call_transform,
        tc_result_transform,
    ) -> None:
        has_tool_messages = any(
            m.tool_result is not None or m.tool_calls for m in self.history
        )

        if not has_tool_messages:
            req = Request(system=self.system)
            req.messages = [Message(role=m.role, content=m.content) for m in self.history]
            msg_transform(body, req, cfg)
            return

        placement = system_placement(ProviderName(self.provider.name))
        if placement == SystemPlacement.SIBLING_OBJECT:
            contents: list[dict[str, Any]] = []
            for m in self.history:
                if m.tool_result is not None:
                    contents.append(tc_result_transform(m.tool_result, cfg.role_mappings))
                elif m.tool_calls:
                    contents.append(tc_call_transform(m.tool_calls, cfg.role_mappings))
                else:
                    contents.append(
                        {
                            "role": map_role(m.role, cfg.role_mappings),
                            "parts": [{"text": m.content}],
                        }
                    )
            body["contents"] = contents
            return

        msgs: list[dict[str, Any]] = []
        if placement == SystemPlacement.MESSAGE_IN_ARRAY and self.system:
            msgs.append(
                {
                    "role": map_role("system", cfg.role_mappings),
                    "content": self.system,
                }
            )
        for m in self.history:
            if m.tool_result is not None:
                msgs.append(tc_result_transform(m.tool_result, cfg.role_mappings))
            elif m.tool_calls:
                msgs.append(tc_call_transform(m.tool_calls, cfg.role_mappings))
            else:
                msgs.append(
                    {
                        "role": map_role(m.role, cfg.role_mappings),
                        "content": m.content,
                    }
                )
        body["messages"] = msgs

    def _find_tool(self, name: str) -> Tool | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


def _fire_post_err(mws: list, base_event: Event, exc: BaseException, start: float) -> None:
    import dataclasses

    ev = dataclasses.replace(base_event, err=str(exc), duration=time.monotonic() - start)
    fire_post(mws, ev)
