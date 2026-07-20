"""











"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from llmkit.builders.batch import BatchHandle
from llmkit.job import JobStatus
from llmkit.types import Provider

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = REPO_ROOT / "codegen" / "testdata" / "wire" / "lifecycle" / "v1"
ARTIFACT_ROOT = REPO_ROOT / "target" / "wire" / "lifecycle"

#
#
_RESULT_LINE = json.dumps(
    {
        "custom_id": "req-0",
        "response": {
            "body": {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        },
    }
)


class _LifecycleMockServer:
    """

"""

    def __init__(self, status: str, output_file_id: str) -> None:
        self.status = status
        self.output_file_id = output_file_id
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self, payload: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                path = urlparse(self.path).path
                if path.startswith("/v1/batches/"):
                    body = {"id": "batch_1", "status": outer.status}
                    if outer.output_file_id:
                        body["output_file_id"] = outer.output_file_id
                    return self._send(json.dumps(body).encode("utf-8"))
                if path.startswith("/v1/files/"):
                    return self._send((_RESULT_LINE + "\n").encode("utf-8"))
                self.send_response(404)
                self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_LifecycleMockServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def _openai_batch_handle(base_url: str) -> BatchHandle:
    return BatchHandle(
        id="batch_1",
        provider=Provider(name="openai", api_key="test-key", base_url=base_url),
    )


def _artifact_from(st: JobStatus) -> dict:
    """


"""
    cause = None
    if st.cause is not None:
        cause = {"status": st.cause.status, "timedOut": st.cause.timed_out}
    return {
        "state": st.state.value,
        "hasResult": st.result is not None,
        "rawStatus": st.raw_status,
        "cause": cause,
    }


def _run_fixture(fixture: str, status: str, output_file_id: str) -> None:
    with _LifecycleMockServer(status, output_file_id) as server:
        handle = _openai_batch_handle(server.url)
        st = asyncio.run(handle.poll())

    artifact = _artifact_from(st)
    out_dir = ARTIFACT_ROOT / fixture
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "python.json").write_text(json.dumps(artifact, indent=2))

    golden = json.loads((GOLDEN_DIR / f"{fixture}.json").read_text())
    assert artifact == golden


def test_lifecycle_batch_succeeded() -> None:
    _run_fixture("batch-succeeded", "completed", "file-out-1")


def test_lifecycle_batch_failed() -> None:
    _run_fixture("batch-failed", "failed", "")
