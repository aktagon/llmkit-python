"""Batch prompting against Anthropic.

Run: ANTHROPIC_API_KEY=sk-... python examples/batch.py

`.batch(...)` submits all prompts as one batch job and waits for every
result. Each prompt becomes an independent request sharing the chained
system prompt and sampling options.
"""
import asyncio
import os

from llmkit.builders import anthropic


async def main() -> None:
    c = anthropic(os.environ.get("ANTHROPIC_API_KEY", "sk-test"))
    # #region batch
    results = await (
        c.text
        .model("claude-sonnet-4-6")
        .system("Be brief")
        .batch(
            "Translate hello to French",
            "Translate hello to Spanish",
            "Translate hello to German",
        )
    )
    for r in results:
        print(r.text)
    # #endregion


if __name__ == "__main__":
    asyncio.run(main())
