"""Batch prompting against Anthropic.

Run: ANTHROPIC_API_KEY=sk-... python examples/batch.py

`c.text.<config>.batch(...)` queues all prompts as one batch job and
returns a handle; `await handle.wait()` blocks until every result is
ready. Each prompt becomes an independent request sharing the chained
system prompt and sampling options.
"""
import asyncio
import os

from llmkit.builders import anthropic


async def main() -> None:
    c = anthropic(os.environ.get("ANTHROPIC_API_KEY", "sk-test"))
    # #region batch
    handle = await (
        c.text
        .model("claude-sonnet-4-6")
        .system("Be brief")
        .batch(
            "Translate hello to French",
            "Translate hello to Spanish",
            "Translate hello to German",
        )
    )
    results = await handle.wait()
    for r in results:
        print(r.text)
    # #endregion


if __name__ == "__main__":
    asyncio.run(main())
