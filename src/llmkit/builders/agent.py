"""









"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..agent import Agent as LegacyAgent
from ..structs import Message, ToolCall, ToolResult
from ..types import Provider, Response

if TYPE_CHECKING:
    from . import Agent


class AgentState:
    def __init__(self, agent: LegacyAgent) -> None:
        self.agent = agent


def _init_agent(b: "Agent") -> AgentState:
    provider = Provider(
        name=b.client.provider.name,
        api_key=b.client.provider.api_key,
    )
    if b._model:
        provider.model = b._model
    if b.client.provider.base_url:
        provider.base_url = b.client.provider.base_url

    kwargs: dict = {}
    if b._max_tokens is not None:
        kwargs["max_tokens"] = b._max_tokens
    if b._temperature is not None:
        kwargs["temperature"] = b._temperature
    if b._top_p is not None:
        kwargs["top_p"] = b._top_p
    if b._top_k is not None:
        kwargs["top_k"] = b._top_k
    if b._frequency_penalty is not None:
        kwargs["frequency_penalty"] = b._frequency_penalty
    if b._presence_penalty is not None:
        kwargs["presence_penalty"] = b._presence_penalty
    if b._seed is not None:
        kwargs["seed"] = b._seed
    if b._stop_sequences:
        kwargs["stop_sequences"] = list(b._stop_sequences)
    if b._thinking_budget is not None:
        kwargs["thinking_budget"] = b._thinking_budget
    if b._reasoning_effort:
        kwargs["reasoning_effort"] = b._reasoning_effort
    if b._max_tool_iterations is not None:
        kwargs["max_tool_iterations"] = b._max_tool_iterations
    if b._caching:
        kwargs["caching"] = True
    if b._middleware:
        kwargs["middleware"] = list(b._middleware)
    if b._safety_settings:
        kwargs["safety_settings"] = list(b._safety_settings)
    if b._raw:
        kwargs["raw"] = True

    agent = LegacyAgent(provider, **kwargs)
    if b._system:
        agent.set_system(b._system)
    for t in b._tools:
        agent.add_tool(t)
    #
    #
    #
    #
    if b._history:
        from ..agent import _InternalMessage
        seeded: list[_InternalMessage] = []
        for m in b._history:
            internal_role = "tool_result" if m.role == "tool" else m.role
            seeded.append(
                _InternalMessage(
                    role=internal_role,
                    content=m.content or "",
                    tool_calls=list(m.tool_calls),
                    tool_result=m.tool_result,
                )
            )
        agent.history = seeded
    return AgentState(agent)


async def agent_prompt(b: "Agent", msg: str) -> Response:
    if b._state is None:
        b._state = _init_agent(b)
    return await asyncio.to_thread(b._state.agent.chat, msg)


def agent_reset(b: "Agent") -> None:
    """


"""
    b._state = None


def _agent_messages(legacy_agent: LegacyAgent) -> tuple[Message, ...]:
    """











"""
    out: list[Message] = []
    for m in legacy_agent.history:
        role = m.role
        if role == "tool_result":
            role = "tool"
        public_tool_calls: list[ToolCall] = []
        for tc in m.tool_calls:
            public_tool_calls.append(
                ToolCall(id=tc.id, name=tc.name, input=tc.input)
            )
        tool_result: ToolResult | None = None
        if m.tool_result is not None:
            tool_result = ToolResult(
                tool_use_id=m.tool_result.tool_use_id,
                content=m.tool_result.content,
            )
        out.append(
            Message(
                role=role,
                content=m.content or "",
                tool_calls=public_tool_calls,
                tool_result=tool_result,
            )
        )
    return tuple(out)
