"""ADR-031: honest no-default contract.

resolve_model is the single predicate every resolution point dispatches
on (middleware events, request body, URL templates). Local daemons
declare no registry default — what a daemon serves is runtime inventory
— so a missing model choice raises immediately with an instruction to
pick one, instead of guessing a model the daemon may not have pulled
(the BUG-009 guess-then-404). Cloud defaults are curated protocol facts
and stay registry-resolved.
"""

from __future__ import annotations

import pytest

# PROVIDERS (the 37-field wire/transform spec registry) is crate-internal as of
# ADR-038 (no longer re-exported from the llmkit root); this test drives the
# internal resolve_model against that spec, so it reads it from the generated
# module directly.
from llmkit.providers.generated.providers import PROVIDERS
from llmkit.errors import ValidationError
from llmkit.middleware import resolve_model
from llmkit.types import Provider

LOCAL_DAEMONS = ["ollama", "vllm", "llamacpp", "lmstudio", "jan"]


def test_no_model_on_local_daemon_raises_naming_provider() -> None:
    p = Provider(name="ollama", api_key="")
    with pytest.raises(ValidationError) as exc_info:
        resolve_model(p, PROVIDERS["ollama"])
    assert exc_info.value.field == "model"
    assert '"ollama" declares no default' in exc_info.value.message
    assert "models.live()" in exc_info.value.message


def test_explicit_model_passes_verbatim() -> None:
    p = Provider(name="ollama", api_key="", model="gemma4:latest")
    assert resolve_model(p, PROVIDERS["ollama"]) == "gemma4:latest"


def test_cloud_default_unchanged() -> None:
    p = Provider(name="anthropic", api_key="")
    assert resolve_model(p, PROVIDERS["anthropic"]) == PROVIDERS["anthropic"].default_model
    assert PROVIDERS["anthropic"].default_model != ""


def test_local_daemons_declare_no_default() -> None:
    for name in LOCAL_DAEMONS:
        assert PROVIDERS[name].default_model == "", name


def test_cloud_providers_declare_defaults() -> None:
    for name, cfg in PROVIDERS.items():
        if name in LOCAL_DAEMONS:
            continue
        assert cfg.default_model != "", name
