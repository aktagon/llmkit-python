"""Model catalogue + provider lookup.

Demonstrates the c.models and c.providers surface (ADR-019). Three modes:

1. Compiled-in catalogue — synchronous, no HTTP. List, filter by
   capability, get by id. Backed by ontology data baked into the SDK.
2. Providers namespace — configured (have credentials + a /v1/models
   endpoint) and supported (every provider the SDK was built with).
3. Live + scoped HTTP — opt into provider /v1/models endpoints for
   the freshest catalogue. live() fans out across configured providers;
   provider(p).list() hits one. raw() additionally populates
   ModelInfo.raw with the provider-native record.

Run: ANTHROPIC_API_KEY=sk-... python examples/catalogue.py
"""
import asyncio
import os

from llmkit import Provider, providers
from llmkit.builders import anthropic
from llmkit.types import Capability


async def main(c=None) -> None:
    c = c or anthropic(os.environ.get("ANTHROPIC_API_KEY", "sk-test"))

    # 1. Compiled-in catalogue.
    all_models = c.models.list()
    print(f"compiled-in non-empty: {len(all_models) > 0}")

    info = c.models.get("claude-opus-4-7")
    print(f"claude-opus-4-7 context > 0: {info is not None and info.context_window > 0}")

    chat = c.models.with_capability(Capability.CHAT_COMPLETION).list()
    print(f"chat-capable non-empty: {len(chat) > 0}")

    # 2. Providers namespace.
    configured = [p.slug for p in c.providers.list()]
    print(f"configured: {configured}")
    print(f"supported >= 1: {len(providers.list()) > 0}")

    # 3. Live + scoped HTTP.
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
