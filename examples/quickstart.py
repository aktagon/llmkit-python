"""






"""
import asyncio
import os

from llmkit.builders import anthropic


async def main() -> None:
    c = anthropic(os.environ.get("ANTHROPIC_API_KEY", "sk-test"))
    resp = await (
        c.text
        .system("Be concise.")
        .temperature(0.3)
        .max_tokens(50)
        .prompt("Say hi")
    )
    print(resp.text)
    print(resp.usage.input, "input tokens,", resp.usage.output, "output tokens")


if __name__ == "__main__":
    asyncio.run(main())
