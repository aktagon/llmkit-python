"""




"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from llmkit.builders import anthropic, google, grok, groq, openai


class _SSEServer:
    """"""

    def __init__(self, events: list[str]) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a, **_k):  # silence noise
                pass

            def do_POST(self):
                #
                #
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b""
                outer.captured_body = json.loads(raw) if raw else {}
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for line in outer.events:
                    self.wfile.write((line + "\n").encode("utf-8"))
                    self.wfile.flush()

        self.events = events
        self.captured_body: dict = {}
        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_SSEServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"


async def _drain(stream) -> None:
    async for _ in stream:
        pass


def test_openai_stream_finish_reason() -> None:
    events = [
        'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
        "",
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}',
        "",
        "data: [DONE]",
        "",
    ]
    with _SSEServer(events) as server:
        c = openai("k")
        c.provider.base_url = server.url
        stream = c.text.stream("hi")
        asyncio.run(_drain(stream))
        assert stream.response is not None
        assert stream.response.finish_reason == "stop"


def test_anthropic_stream_finish_reason() -> None:
    events = [
        "event: content_block_delta",
        'data: {"delta":{"text":"Hi"}}',
        "",
        "event: message_delta",
        'data: {"usage":{"output_tokens":1}}',
        "",
        "event: message_stop",
        'data: {"type":"message_stop","stop_reason":"end_turn"}',
        "",
    ]
    with _SSEServer(events) as server:
        c = anthropic("k")
        c.provider.base_url = server.url
        stream = c.text.stream("hi")
        asyncio.run(_drain(stream))
        assert stream.response is not None
        assert stream.response.finish_reason == "end_turn"


def test_google_stream_finish_reason_filters_unspecified() -> None:
    #
    #
    events = [
        'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]},"finishReason":"FINISH_REASON_UNSPECIFIED"}]}',
        "",
        'data: {"candidates":[{"content":{"parts":[{"text":""}]},"finishReason":"STOP"}]}',
        "",
    ]
    with _SSEServer(events) as server:
        c = google("k")
        c.provider.base_url = server.url
        stream = c.text.stream("hi")
        asyncio.run(_drain(stream))
        assert stream.response is not None
        assert stream.response.finish_reason == "STOP"


def test_pathless_provider_stream_finish_reason_stays_empty() -> None:
    """
"""
    events = [
        'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}',
        "",
        "data: [DONE]",
        "",
    ]
    with _SSEServer(events) as server:
        c = groq("k")
        c.provider.base_url = server.url
        stream = c.text.stream("hi")
        asyncio.run(_drain(stream))
        assert stream.response is not None
        assert stream.response.finish_reason == ""


#
#
#
def test_openai_stream_sends_stream_options_include_usage() -> None:
    events = ["data: [DONE]", ""]
    with _SSEServer(events) as server:
        c = openai("k")
        c.provider.base_url = server.url
        stream = c.text.model("m").stream("hi")
        asyncio.run(_drain(stream))
        assert server.captured_body.get("stream_options") == {"include_usage": True}


def test_grok_stream_omits_stream_options() -> None:
    events = ["data: [DONE]", ""]
    with _SSEServer(events) as server:
        c = grok("k")
        c.provider.base_url = server.url
        stream = c.text.model("m").stream("hi")
        asyncio.run(_drain(stream))
        assert "stream_options" not in server.captured_body


def test_grok_stream_finish_reason() -> None:
    events = [
        'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}',
        "",
        'data: {"choices":[{"delta":{},"finish_reason":"length"}]}',
        "",
        "data: [DONE]",
        "",
    ]
    with _SSEServer(events) as server:
        c = grok("k")
        c.provider.base_url = server.url
        stream = c.text.stream("hi")
        asyncio.run(_drain(stream))
        assert stream.response is not None
        assert stream.response.finish_reason == "length"
