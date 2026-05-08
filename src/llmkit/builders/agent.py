"""









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
    if b._middleware:
        kwargs["middleware"] = list(b._middleware)

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
    """


"""
    b._state = None
