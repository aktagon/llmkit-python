"""






"""
import asyncio
import os

from llmkit.builders import anthropic


async def main() -> None:
    c = anthropic(os.environ.get("ANTHROPIC_API_KEY", "sk-test"))
    stream = c.text.system("Be brief").stream("Tell me a one-line joke")
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()
    final = stream.response
    if final is not None:
        print(
            f"input={final.usage.input} output={final.usage.output} "
            f"finish_reason={final.finish_reason}"
        )


if __name__ == "__main__":
    asyncio.run(main())
