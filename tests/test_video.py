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
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

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
