"""






"""
import asyncio
import os

from llmkit.builders import grok


async def main() -> None:
    c = grok(os.environ.get("XAI_API_KEY", "token"))
    #
    handle = await (
        c.video
        .model("grok-imagine-video")
        .submit("a slow cinematic drone shot flying over snow-capped alpine peaks at golden hour")
    )
    r = await handle.wait()
    v = r.videos[0]
    print(f"url={v.url} duration={v.duration_seconds}s mime={v.mime_type}")
    #


if __name__ == "__main__":
    asyncio.run(main())
