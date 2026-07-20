"""













"""
import asyncio
import os

from llmkit import Provider, providers
from llmkit.builders import anthropic
from llmkit.types import Capability


async def main(c=None) -> None:
    c = c or anthropic(os.environ.get("ANTHROPIC_API_KEY", "sk-test"))

    #
    all_models = c.models.list()
    print(f"compiled-in non-empty: {len(all_models) > 0}")

    info = c.models.get("claude-opus-4-7")
    print(f"claude-opus-4-7 context > 0: {info is not None and info.context_window > 0}")

    chat = c.models.with_capability(Capability.CHAT_COMPLETION).list()
    print(f"chat-capable non-empty: {len(chat) > 0}")

    #
    configured = [p.slug for p in c.providers.list()]
    print(f"configured: {configured}")
    print(f"supported >= 1: {len(providers.list()) > 0}")

    #
    p = Provider(
        name="anthropic",
        api_key=os.environ.get("ANTHROPIC_API_KEY", "sk-test"),
    )
    live = await c.models.live()
    print(f"live models: {len(live.models)}")

    scoped = await c.models.provider(p).list()
    print(f"scoped list: {len(scoped)}")

    raw_scoped = await c.models.provider(p).raw().list()
    raw_populated = bool(raw_scoped) and raw_scoped[0].raw is not None
    print(f"raw populated: {raw_populated}")


if __name__ == "__main__":
    asyncio.run(main())
