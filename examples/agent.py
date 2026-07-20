"""








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
        .add_tool(add_tool)
        .max_tool_iterations(5)
    )
    resp = await bot.prompt("What is 2 + 3?")
    print(resp.text)


if __name__ == "__main__":
    asyncio.run(main())
