"""

"""

from __future__ import annotations

import threading
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, HTTPServer

from llmkit.http import do_multipart_post

HOSTILE_FILENAME = 'evil"name\\inject\r\nX-Fake: 1.mp3'
HOSTILE_FIELD = 'file"field\r\nX-Sneak: a'


def test_multipart_hostile_filename_escaped():
    captured: dict[str, bytes | str] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args, **_kwargs):  # silence noise
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            captured["body"] = self.rfile.read(length)
            captured["content_type"] = self.headers.get("Content-Type", "")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"id":"file_esc"}')

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _, status = do_multipart_post(
            f"http://127.0.0.1:{server.server_port}/upload",
            HOSTILE_FIELD,
            HOSTILE_FILENAME,
            b"audio-bytes",
            {"purpose": "batch"},
            {},
            mime_type="audio/mpeg",
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert status == 200
    raw = captured["body"]
    assert isinstance(raw, bytes)
    assert b"\nX-Fake" not in raw, raw
    assert b"\nX-Sneak" not in raw, raw
    assert b'filename="evil\\"name\\\\injectX-Fake: 1.mp3"' in raw, raw
    assert b'name="file\\"fieldX-Sneak: a"' in raw, raw

    #
    #
    #
    content_type = captured["content_type"]
    assert isinstance(content_type, str)
    message = BytesParser(policy=policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + raw
    )
    parts = list(message.iter_parts())
    assert parts[-1].get_filename() == 'evil"name\\injectX-Fake: 1.mp3'
