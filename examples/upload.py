"""File upload — Path and Bytes paths.

Run: OPENAI_API_KEY=sk-... python examples/upload.py

The `.path()` and `.bytes()` terminals are mutually exclusive on the
same Upload builder — pick one. `.bytes()` requires `.filename()` so
the multipart frame has a meaningful name; `.mime_type()` is
optional and defaults to `application/octet-stream`.
"""
import asyncio
import os

from llmkit.builders import openai


async def main() -> None:
    c = openai(os.environ.get("OPENAI_API_KEY", "sk-test"))

    # Path form
    by_path = await c.upload.path("./data.pdf").run()
    print("by_path:", by_path.id)

    # Bytes form
    payload = b"hello world"
    by_bytes = await (
        c.upload
        .bytes(payload)
        .filename("greeting.txt")
        .mime_type("text/plain")
        .run()
    )
    print("by_bytes:", by_bytes.id)


if __name__ == "__main__":
    asyncio.run(main())
