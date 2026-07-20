"""







"""

from __future__ import annotations

import asyncio

import pytest

from llmkit import Capability, ValidationError, builders, new_client
from llmkit.providers.generated.batch import batch_config
from llmkit.providers.generated.caching import caching_config
from llmkit.providers.generated.image_gen import image_gen_config
from llmkit.providers.generated.providers import ALL_PROVIDER_NAMES
from llmkit.providers.generated.request import file_upload_config


def test_supports_caching_answers_from_gate_table() -> None:
    assert builders.anthropic("k").supports(Capability.CACHING) is True
    assert builders.ollama("").supports(Capability.CACHING) is False


def test_supports_ungated_capabilities_true() -> None:
    c = builders.ollama("")
    assert c.supports(Capability.CHAT_COMPLETION) is True
    assert c.supports(Capability.TOOL_CALLING) is True
    assert c.supports(Capability.REASONING) is True
    assert c.supports(Capability.CATALOGUE) is True


def test_supports_unknown_provider_false_for_gated() -> None:
    #
    c = new_client("nonexistent", "k")
    assert c.supports(Capability.CACHING) is False
    assert c.supports(Capability.BATCHING) is False


def test_supports_matches_strict_gate_lookups_for_every_provider() -> None:
    #
    #
    for pn in ALL_PROVIDER_NAMES:
        c = new_client(pn.value, "k")
        assert c.supports(Capability.CACHING) is (caching_config(pn) is not None)
        assert c.supports(Capability.BATCHING) is (batch_config(pn) is not None)
        assert c.supports(Capability.FILE_UPLOAD) is (
            file_upload_config(pn) is not None
        )
        assert c.supports(Capability.IMAGE_GENERATION) is (
            image_gen_config(pn) is not None
        )


def test_bug008_repro_caching_still_raises_and_supports_gates_it() -> None:
    #
    bot = builders.ollama("").agent.model("gemma4:latest").caching()
    with pytest.raises(ValidationError):
        asyncio.run(bot.prompt("hi"))

    #
    #
    c = builders.ollama("")
    assert c.supports(Capability.CACHING) is False
    gated = c.agent.model("gemma4:latest")
    if c.supports(Capability.CACHING):  # False -> chain stays ungated
        gated = gated.caching()
    assert gated._caching is False  # the .caching() fork was never taken
