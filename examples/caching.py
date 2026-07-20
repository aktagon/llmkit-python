"""







"""
import asyncio
import os

from llmkit.builders import anthropic

LONG_SYS_PROMPT = (
    "You are a meticulous technical editor for a long-running engineering "
    "handbook. Follow these rules on every turn:\n"
    "1. Preserve the author's voice; never rewrite for style alone.\n"
    "2. Correct factual errors and flag claims you cannot verify.\n"
    "3. Keep code samples runnable and idiomatic for their language.\n"
    "4. Prefer stdlib over third-party dependencies in all suggestions.\n"
    "5. When a passage is ambiguous, ask one clarifying question rather "
    "than guessing.\n"
    "6. Use concrete domain values in examples, never placeholders.\n"
    "7. Keep paragraphs short and scannable; one idea per paragraph.\n"
    "This instruction block is reused verbatim across many calls, which is "
    "exactly the workload prompt caching is designed to make cheap.\n"
) * 4


async def main() -> None:
    c = anthropic(os.environ.get("ANTHROPIC_API_KEY", "sk-test"))
    resp = await (
        c.text
        .model("claude-sonnet-4-6")
        .system(LONG_SYS_PROMPT)
        .caching()
        .prompt("Edit this sentence: 'The API are fast.'")
    )
    print(f"cache_read={resp.usage.cache_read} cache_write={resp.usage.cache_write}")


if __name__ == "__main__":
    asyncio.run(main())
