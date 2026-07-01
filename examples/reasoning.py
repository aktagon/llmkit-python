"""






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
