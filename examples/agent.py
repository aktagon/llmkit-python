"""Agent tool loop.

Run: ANTHROPIC_API_KEY=sk-... python examples/agent.py

Note `c.agent` is a stateful field — repeated `bot.prompt(...)` calls
on the same builder accumulate conversation history. Chain methods
(`.system(...)`, `.tool(...)`) clone and reset state, so a forked
builder gets a fresh conversation. `bot.reset()` clears history
without dropping chained config.
"""
import asyncio
import os
from typing import Any

from llmkit import Tool
from llmkit.builders import anthropic


def _add(args: dict[str, Any]) -> str:
    return str(args["a"] + args["b"])


add_tool = Tool(
    name="add",
    description="Add two numbers",
    schema={
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
    },
    run=_add,
)


async def main() -> None:
    c = anthropic(os.environ.get("ANTHROPIC_API_KEY", "sk-test"))
    bot = (
        c.agent
        .system("You are a calculator. Use the add tool.")
        .tool(add_tool)
        .max_tool_iterations(5)
    )
    resp = await bot.prompt("What is 2 + 3?")
    print(resp.text)


if __name__ == "__main__":
    asyncio.run(main())
