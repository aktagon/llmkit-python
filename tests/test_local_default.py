"""ADR-031 / BUG-009(c): local-daemon default-model resolution.

resolve_model is the single predicate every resolution point dispatches
on (middleware events, request body, URL templates). For cfg.local
providers a missing model choice resolves from the daemon's live
listing; cloud providers stay registry-resolved.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llmkit import PROVIDERS
from llmkit.middleware import resolve_model
from llmkit.models import _local_default_cache
from llmkit.types import Provider


class _ModelsHandler(BaseHTTPRequestHandler):
    """Serves an OpenAI-cohort /v1/models listing; counts requests."""

    models: list[str] = []
    requests_served = 0

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        type(self).requests_served += 1
        body = json.dumps(
            {"object": "list", "data": [{"id": m, "object": "model"} for m in self.models]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # silence test output
        pass


@pytest.fixture()
def mock_daemon():
    server = HTTPServer(("127.0.0.1", 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _ModelsHandler.requests_served = 0
    _local_default_cache.clear()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    _local_default_cache.clear()


def test_local_default_falls_back_to_first_installed(mock_daemon: str) -> None:
    # The BUG-009 machine: gemma4 pulled, registry default llama3.2 absent.
    _ModelsHandler.models = ["gemma4:latest"]
    p = Provider(name="ollama", api_key="", base_url=mock_daemon)
    assert resolve_model(p, PROVIDERS["ollama"]) == "gemma4:latest"


def test_local_default_keeps_registry_default_when_installed(mock_daemon: str) -> None:
    _ModelsHandler.models = ["gemma4:latest", "llama3.2"]
    p = Provider(name="ollama", api_key="", base_url=mock_daemon)
    assert resolve_model(p, PROVIDERS["ollama"]) == "llama3.2"


def test_local_default_registry_fallback_when_daemon_unreachable() -> None:
    _local_default_cache.clear()
    # Nothing listens on this port — connection refused, instant.
    p = Provider(name="ollama", api_key="", base_url="http://127.0.0.1:9")
    assert resolve_model(p, PROVIDERS["ollama"]) == "llama3.2"


def test_explicit_model_short_circuits_daemon(mock_daemon: str) -> None:
    _ModelsHandler.models = ["gemma4:latest"]
    p = Provider(name="ollama", api_key="", model="qwen3:8b", base_url=mock_daemon)
    assert resolve_model(p, PROVIDERS["ollama"]) == "qwen3:8b"
    assert _ModelsHandler.requests_served == 0


def test_cloud_default_stays_registry_resolved(mock_daemon: str) -> None:
    _ModelsHandler.models = ["gemma4:latest"]
    p = Provider(name="anthropic", api_key="k", base_url=mock_daemon)
    got = resolve_model(p, PROVIDERS["anthropic"])
    assert got == PROVIDERS["anthropic"].default_model
    assert _ModelsHandler.requests_served == 0


def test_successful_resolution_is_cached(mock_daemon: str) -> None:
    _ModelsHandler.models = ["gemma4:latest"]
    p = Provider(name="ollama", api_key="", base_url=mock_daemon)
    assert resolve_model(p, PROVIDERS["ollama"]) == "gemma4:latest"
    assert resolve_model(p, PROVIDERS["ollama"]) == "gemma4:latest"
    assert _ModelsHandler.requests_served == 1


def test_local_flag_set_for_all_five_daemons() -> None:
    # The generated registry carries the ADR-031 fact for every local.
    for name in ("ollama", "vllm", "llamacpp", "lmstudio", "jan"):
        assert PROVIDERS[name].local is True, name
    assert PROVIDERS["anthropic"].local is False
    assert PROVIDERS["openai"].local is False
