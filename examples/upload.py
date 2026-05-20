"""







"""
import asyncio
import os

from llmkit.builders import openai


async def main() -> None:
    c = openai(os.environ.get("OPENAI_API_KEY", "sk-test"))

    #
    by_path = await c.upload.path("./data.pdf").run()
    print("by_path:", by_path.id)

    #
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
