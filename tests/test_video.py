"""Video generation tests (ADR-034) — mock HTTP server, no live API calls.

Mirror of go/video_test.go. Video generation is asynchronous: submit
returns a handle, then handle.wait() polls until terminal. Slice 1 wires
the Grok (xAI) wire shape only: {model, prompt} submit, url delivery.

The mock server returns `pending` for the first N polls, then the supplied
done body. Each wait() call passes a small poll_interval (mirroring
test_batch.py) so tests run fast.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pytest

from llmkit import ValidationError
from llmkit.builders import new_client
from llmkit.builders.video import VideoHandle, VideoRequest, _submit_video
from llmkit.errors import APIError
from llmkit.types import Provider

GROK_VIDEO_MODEL = "grok-imagine-video"


# Tests pass poll_interval=0.01 to handle.wait() (mirroring test_batch.py) so
# pending->done loops are instant; request_timeout default is fine.
_FAST = {"poll_interval": 0.01}


class _GrokVideoServer:
    """Serves the Grok submit + poll endpoints. The poll returns `pending`
    for the first ``pending_polls`` GET calls, then the supplied done body."""

    def __init__(self, pending_polls: int, done_body: dict[str, Any]) -> None:
        self.pending_polls = pending_polls
        self.done_body = done_body
        self.polls = 0
        self.submit_body: dict[str, Any] | None = None
        self.submit_headers: dict[str, str] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):  # silence noise
                pass

            def _send(self, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                outer.submit_body = json.loads(raw.decode("utf-8"))
                outer.submit_headers = dict(self.headers.items())
                if path.endswith("/v1/videos/generations"):
                    return self._send({"request_id": "vid-123"})
                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                path = urlparse(self.path).path
                if "/v1/videos/vid-123" in path:
                    outer.polls += 1
                    if outer.polls <= outer.pending_polls:
                        return self._send({"status": "pending"})
                    return self._send(outer.done_body)
                self.send_response(404)
                self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_GrokVideoServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def _done_body(url: str, duration: int | None = None) -> dict[str, Any]:
    video: dict[str, Any] = {"url": url}
    if duration is not None:
        video["duration"] = duration
    return {"status": "done", "video": video, "model": GROK_VIDEO_MODEL}


# ===== submit + wait (pending -> done) =====


def test_video_submit_and_wait_grok() -> None:
    done = _done_body("https://vidgen.x.ai/abc/video.mp4", duration=8)
    with _GrokVideoServer(pending_polls=2, done_body=done) as server:
        c = new_client("grok", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(
            c.video.model(GROK_VIDEO_MODEL).submit("a drone shot over the alps, 8s")
        )
        assert isinstance(h, VideoHandle)
        assert h.id == "vid-123"

        resp = asyncio.run(h.wait(**_FAST))

    assert server.submit_headers.get("Authorization") == "Bearer test-token"
    assert server.submit_body == {
        "model": GROK_VIDEO_MODEL,
        "prompt": "a drone shot over the alps, 8s",
    }
    assert len(resp.videos) == 1
    assert resp.videos[0].url == "https://vidgen.x.ai/abc/video.mp4"
    assert resp.videos[0].mime_type == "video/mp4"
    assert resp.videos[0].duration_seconds == 8
    assert resp.videos[0].bytes == b""  # url delivery must not download bytes


# The fixed 1x1 PNG seed frame (single brick-red pixel), shared with the
# image-edit wire fixture; the bytes the image-to-video path inlines.
_GROK_SEED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGM4YWQEAALyAS2saifrAAAAAElFTkSuQmCC"
)


def test_video_grok_image_to_video_submit_body() -> None:
    # BUG-010: .image(...) on Video appends a seed Part; submit inlines it as a
    # data URL at image.url. The round-trip reaches a done video.
    seed = base64.b64decode(_GROK_SEED_PNG_B64)
    done = _done_body("https://vidgen.x.ai/i2v/out.mp4", duration=6)
    with _GrokVideoServer(pending_polls=1, done_body=done) as server:
        c = new_client("grok", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(
            c.video.model(GROK_VIDEO_MODEL)
            .image("image/png", seed)
            .submit("animate the still: slow push-in")
        )
        resp = asyncio.run(h.wait(**_FAST))

    assert server.submit_body == {
        "model": GROK_VIDEO_MODEL,
        "prompt": "animate the still: slow push-in",
        "image": {"url": f"data:image/png;base64,{_GROK_SEED_PNG_B64}"},
    }
    assert resp.videos[0].url == "https://vidgen.x.ai/i2v/out.mp4"


def test_video_image_part_on_text_only_model_rejects() -> None:
    # BUG-010 gate: a model without supports_image_to_video (every model but
    # grok-imagine-video this slice) rejects an image part pre-flight.
    seed = base64.b64decode(_GROK_SEED_PNG_B64)
    c = new_client("zhipu", "test-token")
    with pytest.raises(ValidationError, match="text-to-video-only"):
        asyncio.run(
            c.video.model("cogvideox-3")
            .image("image/png", seed)
            .submit("animate this")
        )


def test_video_rejects_multiple_seed_frames() -> None:
    # Grok Imagine animates one seed frame; a second image part is an error.
    seed = base64.b64decode(_GROK_SEED_PNG_B64)
    c = new_client("grok", "test-token")
    with pytest.raises(ValidationError, match="single seed frame"):
        asyncio.run(
            c.video.model(GROK_VIDEO_MODEL)
            .image("image/png", seed)
            .image("image/png", seed)
            .submit("animate this")
        )


ZHIPU_VIDEO_MODEL = "cogvideox-3"


class _ZhipuVideoServer:
    """Serves the Zhipu CogVideoX submit + async-result endpoints. Submit
    returns the poll handle as the top-level ``id`` (Zhipu's own ``request_id``
    is present but is NOT the poll key); the async-result poll returns
    ``task_status: PROCESSING`` for the first ``pending_polls`` GET calls,
    then the supplied done body."""

    def __init__(self, pending_polls: int, done_body: dict[str, Any]) -> None:
        self.pending_polls = pending_polls
        self.done_body = done_body
        self.polls = 0
        self.submit_body: dict[str, Any] | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                outer.submit_body = json.loads(raw.decode("utf-8"))
                if path.endswith("/v4/videos/generations"):
                    return self._send(
                        {"id": "zhipu-vid-1", "request_id": "rq-xyz", "task_status": "PROCESSING"}
                    )
                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                path = urlparse(self.path).path
                if "/v4/async-result/zhipu-vid-1" in path:
                    outer.polls += 1
                    if outer.polls <= outer.pending_polls:
                        return self._send({"task_status": "PROCESSING"})
                    return self._send(outer.done_body)
                self.send_response(404)
                self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_ZhipuVideoServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def test_video_submit_and_wait_zhipu() -> None:
    done = {
        "task_status": "SUCCESS",
        "video_result": [
            {
                "url": "https://cogvideo.bigmodel.cn/abc/v.mp4",
                "cover_image_url": "https://cogvideo.bigmodel.cn/abc/c.jpg",
            }
        ],
        "model": ZHIPU_VIDEO_MODEL,
    }
    with _ZhipuVideoServer(pending_polls=2, done_body=done) as server:
        c = new_client("zhipu", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(
            c.video.model(ZHIPU_VIDEO_MODEL).submit("a drone shot over the alps")
        )
        assert isinstance(h, VideoHandle)
        assert h.id == "zhipu-vid-1"  # the top-level id, not request_id

        resp = asyncio.run(h.wait(**_FAST))

    assert server.submit_body == {
        "model": ZHIPU_VIDEO_MODEL,
        "prompt": "a drone shot over the alps",
    }
    assert len(resp.videos) == 1
    assert resp.videos[0].url == "https://cogvideo.bigmodel.cn/abc/v.mp4"
    assert resp.videos[0].mime_type == "video/mp4"
    assert resp.videos[0].bytes == b""  # url delivery must not download bytes


def test_video_wait_failed_zhipu_raises() -> None:
    with _ZhipuVideoServer(pending_polls=0, done_body={"task_status": "FAIL"}) as server:
        c = new_client("zhipu", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(c.video.model(ZHIPU_VIDEO_MODEL).submit("blocked prompt"))
        with pytest.raises(APIError):
            asyncio.run(h.wait(**_FAST))


VIDU_VIDEO_MODEL = "viduq3-pro"


class _ViduVideoServer:
    """Serves the Vidu (Shengshu) submit + task-creations poll endpoints.
    Submit POSTs ``/ent/v2/text2video`` and returns the poll handle as the
    top-level ``task_id``; the poll GET ``/ent/v2/tasks/{id}/creations``
    returns ``state: processing`` for the first ``pending_polls`` GET calls,
    then the supplied done body. Vidu authenticates with the ``Token`` scheme
    (Authorization: Token <key>), not Bearer."""

    def __init__(self, pending_polls: int, done_body: dict[str, Any]) -> None:
        self.pending_polls = pending_polls
        self.done_body = done_body
        self.polls = 0
        self.submit_body: dict[str, Any] | None = None
        self.submit_auth: str | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                outer.submit_body = json.loads(raw.decode("utf-8"))
                outer.submit_auth = self.headers.get("Authorization")
                if path.endswith("/ent/v2/text2video"):
                    return self._send({"task_id": "vidu-task-1", "state": "created"})
                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/ent/v2/tasks/vidu-task-1/creations":
                    outer.polls += 1
                    if outer.polls <= outer.pending_polls:
                        return self._send({"state": "processing"})
                    return self._send(outer.done_body)
                self.send_response(404)
                self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_ViduVideoServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def test_video_submit_and_wait_vidu() -> None:
    done = {
        "state": "success",
        "creations": [{"url": "https://api.vidu.com/creations/abc/v.mp4"}],
    }
    with _ViduVideoServer(pending_polls=2, done_body=done) as server:
        c = new_client("vidu", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(
            c.video.model(VIDU_VIDEO_MODEL).submit("a drone shot over the alps")
        )
        assert isinstance(h, VideoHandle)
        assert h.id == "vidu-task-1"

        resp = asyncio.run(h.wait(**_FAST))

    assert server.submit_auth == "Token test-token"
    assert server.submit_body == {
        "model": VIDU_VIDEO_MODEL,
        "prompt": "a drone shot over the alps",
    }
    assert len(resp.videos) == 1
    assert resp.videos[0].url == "https://api.vidu.com/creations/abc/v.mp4"
    assert resp.videos[0].mime_type == "video/mp4"
    assert resp.videos[0].bytes == b""  # url delivery must not download bytes


def test_video_wait_failed_vidu_raises() -> None:
    with _ViduVideoServer(
        pending_polls=0, done_body={"state": "failed", "err_code": "content_moderation"}
    ) as server:
        c = new_client("vidu", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(c.video.model(VIDU_VIDEO_MODEL).submit("blocked prompt"))
        with pytest.raises(APIError, match="content_moderation"):
            asyncio.run(h.wait(**_FAST))


def test_video_image_part_on_text_only_vidu_rejects() -> None:
    # BUG-010 gate: Vidu models set supports_image_to_video=False, so an image
    # part is rejected pre-flight.
    seed = base64.b64decode(_GROK_SEED_PNG_B64)
    c = new_client("vidu", "test-token")
    with pytest.raises(ValidationError, match="text-to-video-only"):
        asyncio.run(
            c.video.model(VIDU_VIDEO_MODEL)
            .image("image/png", seed)
            .submit("animate this")
        )


TOGETHER_VIDEO_MODEL = "minimax/video-01-director"


class _TogetherVideoServer:
    """Serves the Together submit + poll endpoints. Submit returns the poll
    handle as the top-level ``id`` with status=queued; the poll returns
    ``status: in_progress`` for the first ``pending_polls`` GET calls, then the
    supplied done body."""

    def __init__(self, pending_polls: int, done_body: dict[str, Any]) -> None:
        self.pending_polls = pending_polls
        self.done_body = done_body
        self.polls = 0
        self.submit_body: dict[str, Any] | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                outer.submit_body = json.loads(raw.decode("utf-8"))
                if path.endswith("/v2/videos"):
                    return self._send({"id": "together-vid-1", "status": "queued"})
                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                path = urlparse(self.path).path
                if "/v2/videos/together-vid-1" in path:
                    outer.polls += 1
                    if outer.polls <= outer.pending_polls:
                        return self._send({"status": "in_progress"})
                    return self._send(outer.done_body)
                self.send_response(404)
                self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_TogetherVideoServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def test_video_submit_and_wait_together() -> None:
    done = {
        "status": "completed",
        "outputs": {"video_url": "https://api.together.xyz/files/v.mp4"},
        "model": TOGETHER_VIDEO_MODEL,
    }
    with _TogetherVideoServer(pending_polls=2, done_body=done) as server:
        c = new_client("together", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(
            c.video.model(TOGETHER_VIDEO_MODEL).submit("a drone shot over the alps")
        )
        assert isinstance(h, VideoHandle)
        assert h.id == "together-vid-1"  # the top-level id

        resp = asyncio.run(h.wait(**_FAST))

    assert server.submit_body == {
        "model": TOGETHER_VIDEO_MODEL,
        "prompt": "a drone shot over the alps",
    }
    assert len(resp.videos) == 1
    assert resp.videos[0].url == "https://api.together.xyz/files/v.mp4"
    assert resp.videos[0].mime_type == "video/mp4"
    assert resp.videos[0].bytes == b""  # url delivery must not download bytes


def test_video_wait_cancelled_together_raises() -> None:
    with _TogetherVideoServer(pending_polls=0, done_body={"status": "cancelled"}) as server:
        c = new_client("together", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(c.video.model(TOGETHER_VIDEO_MODEL).submit("blocked prompt"))
        with pytest.raises(APIError):
            asyncio.run(h.wait(**_FAST))


QWEN_VIDEO_MODEL = "wan2.2-t2v-plus"


class _QwenVideoServer:
    """Serves the DashScope (Qwen) submit + poll endpoints. Submit returns the
    poll handle as ``output.task_id`` (the dotted-path handle) with
    ``output.task_status: PENDING``; the poll returns ``output.task_status:
    RUNNING`` for the first ``pending_polls`` GET calls, then the supplied done
    body. Captures the submit body plus the X-DashScope-Async header value."""

    def __init__(self, pending_polls: int, done_body: dict[str, Any]) -> None:
        self.pending_polls = pending_polls
        self.done_body = done_body
        self.polls = 0
        self.submit_body: dict[str, Any] | None = None
        self.async_header: str | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                outer.submit_body = json.loads(raw.decode("utf-8"))
                outer.async_header = self.headers.get("X-DashScope-Async")
                if path.endswith("/video-synthesis"):
                    return self._send(
                        {
                            "output": {
                                "task_id": "qwen-vid-1",
                                "task_status": "PENDING",
                            },
                            "request_id": "req-1",
                        }
                    )
                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                path = urlparse(self.path).path
                if "/api/v1/tasks/qwen-vid-1" in path:
                    outer.polls += 1
                    if outer.polls <= outer.pending_polls:
                        return self._send({"output": {"task_status": "RUNNING"}})
                    return self._send(outer.done_body)
                self.send_response(404)
                self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_QwenVideoServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def test_video_submit_and_wait_qwen() -> None:
    done = {
        "output": {
            "task_status": "SUCCEEDED",
            "video_url": "https://dashscope-result.oss-cn.aliyuncs.com/v.mp4",
        }
    }
    with _QwenVideoServer(pending_polls=2, done_body=done) as server:
        c = new_client("qwen", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(
            c.video.model(QWEN_VIDEO_MODEL).submit("a drone shot over the alps")
        )
        assert isinstance(h, VideoHandle)
        assert h.id == "qwen-vid-1"  # the output.task_id dotted-path handle

        resp = asyncio.run(h.wait(**_FAST))

    # Nested submit body: prompt under input, no top-level prompt; async header.
    assert server.submit_body == {
        "model": QWEN_VIDEO_MODEL,
        "input": {"prompt": "a drone shot over the alps"},
    }
    assert server.async_header == "enable"
    assert len(resp.videos) == 1
    assert resp.videos[0].url == "https://dashscope-result.oss-cn.aliyuncs.com/v.mp4"
    assert resp.videos[0].mime_type == "video/mp4"
    assert resp.videos[0].bytes == b""  # url delivery must not download bytes


def test_video_wait_failed_qwen_raises() -> None:
    done = {"output": {"task_status": "FAILED"}}
    with _QwenVideoServer(pending_polls=0, done_body=done) as server:
        c = new_client("qwen", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(c.video.model(QWEN_VIDEO_MODEL).submit("blocked prompt"))
        with pytest.raises(APIError):
            asyncio.run(h.wait(**_FAST))


MINIMAX_VIDEO_MODEL = "MiniMax-Hailuo-2.3"


class _MinimaxVideoServer:
    """Serves the MiniMax two-hop flow: submit -> {task_id}; query poll returns
    ``status: Processing`` for the first ``pending_polls`` GET calls, then
    ``{status: Success, file_id}``; the file-retrieve hop returns the download
    URL. file_id is served as a JSON number (minimax encodes it as an integer).
    When ``fail`` is set the poll returns ``status: Fail``."""

    def __init__(self, pending_polls: int, download_url: str, fail: bool = False) -> None:
        self.pending_polls = pending_polls
        self.download_url = download_url
        self.fail = fail
        self.polls = 0
        self.submit_body: dict[str, Any] | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                outer.submit_body = json.loads(raw.decode("utf-8"))
                if path.endswith("/v1/video_generation"):
                    return self._send(
                        {"task_id": "mmtask-1", "base_resp": {"status_code": 0}}
                    )
                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                path = urlparse(self.path).path
                if "/v1/query/video_generation" in path:
                    if outer.fail:
                        return self._send({"status": "Fail"})
                    outer.polls += 1
                    if outer.polls <= outer.pending_polls:
                        return self._send({"status": "Processing"})
                    return self._send({"status": "Success", "file_id": 99887766})
                if "/v1/files/retrieve" in path:
                    return self._send({"file": {"download_url": outer.download_url}})
                self.send_response(404)
                self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_MinimaxVideoServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def test_video_submit_and_wait_minimax_two_hop() -> None:
    with _MinimaxVideoServer(
        pending_polls=2, download_url="https://files.minimax.io/abc/v.mp4"
    ) as server:
        c = new_client("minimax", "test-token")
        c.provider.base_url = server.url  # override wins (Option D)
        h = asyncio.run(
            c.video.model(MINIMAX_VIDEO_MODEL).submit("a drone shot over the alps")
        )
        assert isinstance(h, VideoHandle)
        assert h.id == "mmtask-1"

        resp = asyncio.run(h.wait(**_FAST))

    assert server.submit_body == {
        "model": MINIMAX_VIDEO_MODEL,
        "prompt": "a drone shot over the alps",
    }
    assert len(resp.videos) == 1
    # The URL came from the second (file-retrieve) hop, not the poll body.
    assert resp.videos[0].url == "https://files.minimax.io/abc/v.mp4"
    assert resp.videos[0].bytes == b""  # url delivery must not download bytes


def test_video_wait_failed_minimax_raises() -> None:
    with _MinimaxVideoServer(pending_polls=0, download_url="", fail=True) as server:
        c = new_client("minimax", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(c.video.model(MINIMAX_VIDEO_MODEL).submit("blocked prompt"))
        with pytest.raises(APIError):
            asyncio.run(h.wait(**_FAST))


VEO_VIDEO_MODEL = "veo-3.1-generate-preview"


class _VeoVideoServer:
    """Serves the Google Veo LRO flow: submit ->
    {name:"models/.../operations/op-1"}; operation poll returns {done:false}
    for the first ``pending_polls`` GET calls, then a done op whose response
    carries the Files-API video.uri (download delivery). The download hop GETs
    that uri and returns raw mp4 bytes. Every hop must carry the ?key= query-
    param auth (Google is the first video provider that is NOT bearer-header).
    The download uri is served with a pre-existing ?alt=media query so the test
    also witnesses the ?->& auth-append branch. When ``fail`` is set the done op
    carries an error."""

    def __init__(self, pending_polls: int, video_bytes: bytes, fail: bool = False) -> None:
        self.pending_polls = pending_polls
        self.video_bytes = video_bytes
        self.fail = fail
        self.polls = 0
        self.submit_body: dict[str, Any] | None = None
        self.seen_keys: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send_json(self, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _record_key(self) -> dict[str, str]:
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                outer.seen_keys.append(query.get("key", [""])[0])
                return query

            def do_POST(self):
                self._record_key()
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                outer.submit_body = json.loads(raw.decode("utf-8"))
                if path.endswith(":predictLongRunning"):
                    return self._send_json(
                        {"name": "models/veo-3.1-generate-preview/operations/op-1"}
                    )
                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                query = self._record_key()
                path = urlparse(self.path).path
                if path.endswith("/operations/op-1"):
                    if outer.fail:
                        return self._send_json(
                            {
                                "done": True,
                                "error": {
                                    "code": 3,
                                    "message": "prompt blocked by safety filter",
                                },
                            }
                        )
                    outer.polls += 1
                    if outer.polls <= outer.pending_polls:
                        return self._send_json({"done": False})
                    return self._send_json(
                        {
                            "done": True,
                            "response": {
                                "generateVideoResponse": {
                                    "generatedSamples": [
                                        {
                                            "video": {
                                                "uri": outer.url
                                                + "/v1beta/files/vid-file:download?alt=media"
                                            }
                                        }
                                    ]
                                }
                            },
                        }
                    )
                if path.endswith("/files/vid-file:download"):
                    if query.get("alt", [""])[0] != "media":
                        self.send_response(400)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(len(outer.video_bytes)))
                    self.end_headers()
                    self.wfile.write(outer.video_bytes)
                    return
                self.send_response(404)
                self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_VeoVideoServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def test_video_submit_and_wait_veo_download_delivery() -> None:
    want_bytes = b"\x00\x00\x00\x18ftypmp42 fake mp4 payload"
    with _VeoVideoServer(pending_polls=2, video_bytes=want_bytes) as server:
        c = new_client("google", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(
            c.video.model(VEO_VIDEO_MODEL).submit(
                "a drone shot over the alps at sunrise"
            )
        )
        assert isinstance(h, VideoHandle)
        assert h.id == "models/veo-3.1-generate-preview/operations/op-1"

        resp = asyncio.run(h.wait(**_FAST))

    # Veo submit body has instances[0].prompt and NO model field.
    assert server.submit_body == {
        "instances": [{"prompt": "a drone shot over the alps at sunrise"}]
    }
    assert len(resp.videos) == 1
    # Download delivery filled bytes and cleared url (source-XOR, VID-004).
    assert resp.videos[0].bytes == want_bytes
    assert resp.videos[0].url == ""
    assert resp.videos[0].mime_type == "video/mp4"
    # ?key=test-token on submit, every poll, and the download hop.
    assert server.seen_keys
    assert all(k == "test-token" for k in server.seen_keys)


def test_video_wait_failed_veo_raises() -> None:
    with _VeoVideoServer(pending_polls=0, video_bytes=b"", fail=True) as server:
        c = new_client("google", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(c.video.model(VEO_VIDEO_MODEL).submit("blocked prompt"))
        with pytest.raises(APIError) as exc_info:
            asyncio.run(h.wait(**_FAST))
        assert "prompt blocked by safety filter" in str(exc_info.value)


VERTEX_VEO_MODEL = "veo-3.1-generate-preview"


class _VertexVeoVideoServer:
    """Serves the Vertex AI Veo fetchPredictOperation flow: submit ->
    {name:"projects/.../operations/op-7"}; the operation poll is a POST to
    {model}:fetchPredictOperation carrying {"operationName": <id>} and returns
    {done:false} for the first ``pending_polls`` calls, then a done op whose
    response.videos[0].bytesBase64Encoded carries the inline mp4 (download
    delivery with NO fetch hop — bytes arrive in the poll body). Vertex uses
    bearer auth (no ?key= query param). When ``fail`` is set the done op carries
    an error; when ``empty`` is set the done op carries no decodable bytes."""

    def __init__(
        self,
        pending_polls: int,
        video_bytes: bytes,
        fail: bool = False,
        empty: bool = False,
    ) -> None:
        self.pending_polls = pending_polls
        self.video_bytes = video_bytes
        self.fail = fail
        self.empty = empty
        self.polls = 0
        self.submit_body: dict[str, Any] | None = None
        self.poll_bodies: list[dict[str, Any]] = []
        self.submit_auth: str | None = None
        self.poll_auth: str | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send_json(self, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                parsed = json.loads(raw.decode("utf-8"))
                if path.endswith(":predictLongRunning"):
                    outer.submit_body = parsed
                    outer.submit_auth = self.headers.get("Authorization")
                    return self._send_json(
                        {"name": "projects/p-1/locations/us-central1/operations/op-7"}
                    )
                if path.endswith(":fetchPredictOperation"):
                    outer.poll_bodies.append(parsed)
                    outer.poll_auth = self.headers.get("Authorization")
                    if outer.fail:
                        return self._send_json(
                            {
                                "done": True,
                                "error": {
                                    "code": 3,
                                    "message": "prompt blocked by safety filter",
                                },
                            }
                        )
                    outer.polls += 1
                    if outer.polls <= outer.pending_polls:
                        return self._send_json({"done": False})
                    if outer.empty:
                        return self._send_json({"done": True, "response": {"videos": []}})
                    return self._send_json(
                        {
                            "done": True,
                            "response": {
                                "videos": [
                                    {
                                        "bytesBase64Encoded": base64.b64encode(
                                            outer.video_bytes
                                        ).decode("ascii"),
                                        "mimeType": "video/mp4",
                                    }
                                ]
                            },
                        }
                    )
                self.send_response(404)
                self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_VertexVeoVideoServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def test_video_submit_and_wait_vertex_veo_inline_bytes() -> None:
    want_bytes = b"\x00\x00\x00\x18ftypmp42 vertex fake mp4 payload"
    with _VertexVeoVideoServer(pending_polls=2, video_bytes=want_bytes) as server:
        c = new_client("vertex", "ya29.bearer-token")
        c.provider.base_url = server.url
        h = asyncio.run(
            c.video.model(VERTEX_VEO_MODEL).submit(
                "a drone shot over the alps at sunrise"
            )
        )
        assert isinstance(h, VideoHandle)
        assert h.id == "projects/p-1/locations/us-central1/operations/op-7"
        # The handle carries the model so the POST poll can template
        # {model}:fetchPredictOperation.
        assert h.model == VERTEX_VEO_MODEL

        resp = asyncio.run(h.wait(**_FAST))

    # Vertex Veo submit body has instances[0].prompt and NO model field.
    assert server.submit_body == {
        "instances": [{"prompt": "a drone shot over the alps at sunrise"}]
    }
    # The poll is a POST carrying the operation name in the body, not the URL.
    assert server.poll_bodies
    assert all(
        b == {"operationName": "projects/p-1/locations/us-central1/operations/op-7"}
        for b in server.poll_bodies
    )
    assert len(resp.videos) == 1
    # Inline base64 decoded straight into bytes; url stays empty (source-XOR,
    # VID-004) — download delivery with no fetch hop.
    assert resp.videos[0].bytes == want_bytes
    assert resp.videos[0].url == ""
    assert resp.videos[0].mime_type == "video/mp4"
    # Bearer auth on submit and poll (Vertex is NOT a ?key= query-param provider).
    assert server.submit_auth == "Bearer ya29.bearer-token"
    assert server.poll_auth == "Bearer ya29.bearer-token"


def test_video_wait_failed_vertex_veo_raises() -> None:
    with _VertexVeoVideoServer(pending_polls=0, video_bytes=b"", fail=True) as server:
        c = new_client("vertex", "ya29.bearer-token")
        c.provider.base_url = server.url
        h = asyncio.run(c.video.model(VERTEX_VEO_MODEL).submit("blocked prompt"))
        with pytest.raises(APIError) as exc_info:
            asyncio.run(h.wait(**_FAST))
        assert "prompt blocked by safety filter" in str(exc_info.value)


def test_video_vertex_veo_done_no_bytes_raises() -> None:
    # A done operation that carries no decodable bytes must error, not return a
    # silent empty success (mirrors the Veo done+no-uri guard).
    with _VertexVeoVideoServer(pending_polls=0, video_bytes=b"", empty=True) as server:
        c = new_client("vertex", "ya29.bearer-token")
        c.provider.base_url = server.url
        h = asyncio.run(c.video.model(VERTEX_VEO_MODEL).submit("a quiet forest"))
        with pytest.raises(APIError) as exc_info:
            asyncio.run(h.wait(**_FAST))
    assert "no video bytes" in exc_info.value.message


NOVA_REEL_MODEL = "amazon.nova-reel-v1:0"
NOVA_REEL_ARN = "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456"
NOVA_REEL_OUTPUT_URI = "s3://my-bucket/out/"


class _BedrockVideoServer:
    """Serves the Nova Reel start-async-invoke + get-async-invoke endpoints.
    Bedrock is the FIRST SigV4-signed video provider (every other is a bearer
    header) and the FIRST output-uri delivery (the provider writes the mp4 to
    the caller's S3 bucket; the SDK never downloads). Submit returns the poll
    handle as the top-level ``invocationArn``; the poll returns
    ``status: InProgress`` for the first ``pending_polls`` GET calls, then the
    supplied done body. When ``fail_msg`` is non-empty the poll returns a Failed
    status carrying it. Captures the submit body, the Authorization header, and
    the round-tripped poll path."""

    def __init__(
        self,
        pending_polls: int,
        done_body: dict[str, Any],
        fail_msg: str = "",
    ) -> None:
        self.pending_polls = pending_polls
        self.done_body = done_body
        self.fail_msg = fail_msg
        self.polls = 0
        self.submit_body: dict[str, Any] | None = None
        self.submit_auth: str | None = None
        self.poll_auth: str | None = None
        self.poll_path: str | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):
                pass

            def _send(self, body: dict[str, Any]) -> None:
                payload = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                path = urlparse(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                outer.submit_body = json.loads(raw.decode("utf-8"))
                outer.submit_auth = self.headers.get("Authorization")
                if path.endswith("/async-invoke"):
                    return self._send({"invocationArn": NOVA_REEL_ARN})
                self.send_response(404)
                self.end_headers()

            def do_GET(self):
                # The ARN is percent-encoded as one path segment on the wire; the
                # server's decoded Path restores the ':' and '/'. Witness that the
                # full ARN round-trips so the encoding is not lossy.
                outer.poll_path = urlparse(self.path).path
                outer.poll_auth = self.headers.get("Authorization")
                if "/async-invoke/" in outer.poll_path:
                    if outer.fail_msg:
                        return self._send(
                            {"status": "Failed", "failureMessage": outer.fail_msg}
                        )
                    outer.polls += 1
                    if outer.polls <= outer.pending_polls:
                        return self._send({"status": "InProgress"})
                    return self._send(outer.done_body)
                self.send_response(404)
                self.end_headers()

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_BedrockVideoServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


def test_video_submit_and_wait_bedrock_output_uri() -> None:
    done = {
        "status": "Completed",
        "outputDataConfig": {
            "s3OutputDataConfig": {"s3Uri": NOVA_REEL_OUTPUT_URI},
        },
    }
    with _BedrockVideoServer(pending_polls=2, done_body=done) as server:
        c = new_client("bedrock", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(
            c.video.model(NOVA_REEL_MODEL)
            .output_uri(NOVA_REEL_OUTPUT_URI)
            .submit("a drone shot over the alps, 6s")
        )
        assert isinstance(h, VideoHandle)
        assert h.id == NOVA_REEL_ARN  # the invocationArn

        resp = asyncio.run(h.wait(**_FAST))

    # SigV4 auth, not bearer, on both the submit and the poll.
    assert server.submit_auth is not None
    assert server.submit_auth.startswith("AWS4-HMAC-SHA256")
    assert server.poll_auth is not None
    assert server.poll_auth.startswith("AWS4-HMAC-SHA256")
    # The full ARN round-trips in the poll path: the ':' is signed literally and
    # the '/' is percent-encoded as one path segment (%2F), so unquoting the wire
    # path restores the exact ARN (Go's r.URL.Path auto-decodes; Python's
    # BaseHTTPRequestHandler leaves %2F raw, so unquote here).
    assert server.poll_path is not None
    assert NOVA_REEL_ARN in unquote(server.poll_path)
    # Nova Reel carries the model in the body, prompt under modelInput, and the
    # caller S3 URI under outputDataConfig.
    assert server.submit_body == {
        "modelId": NOVA_REEL_MODEL,
        "modelInput": {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": {"text": "a drone shot over the alps, 6s"},
        },
        "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": NOVA_REEL_OUTPUT_URI}},
    }
    assert len(resp.videos) == 1
    # Output-uri delivery: the caller S3 URI in url, no bytes (the provider wrote
    # to the caller's bucket; the SDK never downloads).
    assert resp.videos[0].url == NOVA_REEL_OUTPUT_URI
    assert resp.videos[0].bytes == b""
    assert resp.videos[0].mime_type == "video/mp4"


def test_video_bedrock_requires_output_uri() -> None:
    # VID-005: an output-uri provider must reject a submit that omits the caller
    # S3 URI before any HTTP call. No server: validation fails pre-flight.
    c = new_client("bedrock", "test-token")
    c.provider.base_url = "http://unused"
    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(c.video.model(NOVA_REEL_MODEL).submit("a drone shot over the alps"))
    assert exc_info.value.field == "output_uri"


def test_video_wait_failed_bedrock_raises() -> None:
    with _BedrockVideoServer(
        pending_polls=0,
        done_body={},
        fail_msg="S3 bucket not writable by the service role",
    ) as server:
        c = new_client("bedrock", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(
            c.video.model(NOVA_REEL_MODEL)
            .output_uri(NOVA_REEL_OUTPUT_URI)
            .submit("a drone shot over the alps")
        )
        with pytest.raises(APIError) as exc_info:
            asyncio.run(h.wait(**_FAST))
    assert "S3 bucket not writable by the service role" in exc_info.value.message


def test_video_bedrock_completed_no_uri_raises() -> None:
    # A Completed invocation that echoes no output s3 uri must error, not return
    # a silent empty success (mirrors the Veo done+no-uri guard).
    with _BedrockVideoServer(pending_polls=0, done_body={"status": "Completed"}) as server:
        c = new_client("bedrock", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(
            c.video.model(NOVA_REEL_MODEL)
            .output_uri(NOVA_REEL_OUTPUT_URI)
            .submit("a drone shot")
        )
        with pytest.raises(APIError) as exc_info:
            asyncio.run(h.wait(**_FAST))
    assert "no output s3 uri" in exc_info.value.message


def test_video_text_chain_method() -> None:
    done = _done_body("https://vidgen.x.ai/t.mp4")
    with _GrokVideoServer(pending_polls=0, done_body=done) as server:
        c = new_client("grok", "test-token")
        c.provider.base_url = server.url
        # Exercises the Video.text accumulator with an empty submit msg.
        h = asyncio.run(
            c.video.model(GROK_VIDEO_MODEL).text("a calm lake at dawn").submit("")
        )
        resp = asyncio.run(h.wait(**_FAST))

    assert server.submit_body == {
        "model": GROK_VIDEO_MODEL,
        "prompt": "a calm lake at dawn",
    }
    assert resp.videos[0].url == "https://vidgen.x.ai/t.mp4"


# ===== raw capture =====


def test_video_raw_captures_poll_body() -> None:
    done = _done_body("https://vidgen.x.ai/x.mp4")
    with _GrokVideoServer(pending_polls=0, done_body=done) as server:
        c = new_client("grok", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(c.video.model(GROK_VIDEO_MODEL).raw().submit("a sunrise timelapse"))
        assert h.raw is True  # propagated from the chain
        resp = asyncio.run(h.wait(**_FAST))

    assert resp.raw is not None
    assert resp.raw["video"]["url"] == "https://vidgen.x.ai/x.mp4"


# ===== failed / expired job raises =====


def test_video_wait_failed_raises() -> None:
    done = {
        "status": "failed",
        "error": {"code": "invalid_argument", "message": "prompt blocked by moderation"},
    }
    with _GrokVideoServer(pending_polls=0, done_body=done) as server:
        c = new_client("grok", "test-token")
        c.provider.base_url = server.url
        h = asyncio.run(c.video.model(GROK_VIDEO_MODEL).submit("blocked prompt"))
        with pytest.raises(APIError) as exc_info:
            asyncio.run(h.wait(**_FAST))
    assert "prompt blocked by moderation" in exc_info.value.message


# ===== validation =====


def test_video_requires_model() -> None:
    with pytest.raises(ValidationError) as exc_info:
        c = new_client("grok", "test-token")
        asyncio.run(c.video.submit("no model set"))
    assert exc_info.value.field == "model"


def test_video_rejects_unknown_model() -> None:
    with pytest.raises(ValidationError) as exc_info:
        c = new_client("grok", "test-token")
        c.provider.base_url = "http://unused"
        asyncio.run(c.video.model("grok-imagine-nope").submit("x"))
    assert exc_info.value.field == "model"


def test_video_rejects_unsupported_provider() -> None:
    with pytest.raises(ValidationError) as exc_info:
        c = new_client("anthropic", "test-key")
        c.provider.base_url = "http://unused"
        asyncio.run(c.video.model(GROK_VIDEO_MODEL).submit("x"))
    assert exc_info.value.field == "provider"
    assert "does not support video generation" in exc_info.value.message


def test_video_rejects_lyrics_part() -> None:
    from llmkit import Part

    req = VideoRequest(model=GROK_VIDEO_MODEL, parts=[Part(lyrics="la la la")])
    provider = Provider(name="grok", api_key="test-token", base_url="http://unused")
    with pytest.raises(ValidationError) as exc_info:
        _submit_video(provider, req, [], False)
    assert exc_info.value.field == "parts[0]"
    assert "lyrics" in exc_info.value.message


def test_video_xor_neither_set() -> None:
    req = VideoRequest(model=GROK_VIDEO_MODEL)
    provider = Provider(name="grok", api_key="test-token", base_url="http://unused")
    with pytest.raises(ValidationError) as exc_info:
        _submit_video(provider, req, [], False)
    assert exc_info.value.field == "prompt"


def test_video_xor_both_set() -> None:
    from llmkit import Part

    req = VideoRequest(model=GROK_VIDEO_MODEL, prompt="x", parts=[Part(text="y")])
    provider = Provider(name="grok", api_key="test-token", base_url="http://unused")
    with pytest.raises(ValidationError) as exc_info:
        _submit_video(provider, req, [], False)
    assert exc_info.value.field == "parts"


# ===== middleware =====


def test_video_middleware_fires_pre_then_post() -> None:
    ops: list[str] = []
    phases: list[str] = []

    def mw(event):
        ops.append(event.op.value)
        phases.append(event.phase.value)
        return None

    done = _done_body("https://vidgen.x.ai/m.mp4")
    with _GrokVideoServer(pending_polls=0, done_body=done) as server:
        c = new_client("grok", "test-token")
        c.provider.base_url = server.url
        asyncio.run(
            c.video.model(GROK_VIDEO_MODEL).add_middleware(mw).submit("drone shot")
        )

    assert ops == ["video_generation", "video_generation"]
    assert phases == ["pre", "post"]
