"""Reasoning-effort prompting against OpenAI.

Run: OPENAI_API_KEY=sk-... python examples/reasoning.py

`.reasoning_effort("high")` raises the internal reasoning budget. Reasoning
tokens are only reported by o-series / thinking models (OpenAI o1/o3/o4,
Gemini 2.5+); other models leave usage.reasoning at zero.
"""
import asyncio
import os

from llmkit.builders import openai


async def main() -> None:
    c = openai(os.environ.get("OPENAI_API_KEY", "sk-test"))
    resp = await (
        c.text
        .model("o4-mini")
        .reasoning_effort("high")
        .prompt("A farmer has 17 sheep; all but 9 run away. How many remain?")
    )
    print(f"reasoning tokens: {resp.usage.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
