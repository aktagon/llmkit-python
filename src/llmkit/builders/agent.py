"""Phase 3 slice 2c — wires Agent.prompt + Agent.reset against the
legacy ``Agent`` class.

Stateful builder pattern (mirror of Go/TS slice 2c). The typed-builder
``Agent`` carries a private ``_state: AgentState | None`` that wraps a
live legacy ``Agent`` instance. First ``.prompt()`` lazily constructs
it from chained config; subsequent calls reuse it so history
accumulates. Forking via a chain method (e.g., ``bot.system("new")``)
produces a clone with ``_state = None`` thanks to the codegen
post-mutation hook (PYTHON_BUILDER_POST_MUTATION["Agent"]).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..agent import Agent as LegacyAgent
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
    return AgentState(agent)


async def agent_prompt(b: "Agent", msg: str) -> Response:
    if b._state is None:
        b._state = _init_agent(b)
    return await asyncio.to_thread(b._state.agent.chat, msg)


def agent_reset(b: "Agent") -> None:
    """Clears state. Chain config is preserved on the typed builder;
    next .prompt() re-runs ``_init_agent``. Deliberately doesn't call
    ``LegacyAgent.reset()``, which clears tools too — the typed
    builder's own ``_tools`` slice re-supplies them on re-init."""
    b._state = None
