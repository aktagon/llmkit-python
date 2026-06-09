"""Text-to-video generation against xAI's Grok Imagine (ADR-034).

Run: XAI_API_KEY=... python examples/video.py

Video generation is asynchronous: submit returns a handle immediately;
wait() polls until the job completes and returns a temporary xAI-hosted URL
(url delivery — the SDK does not download the bytes for you).
"""
import asyncio
import os

from llmkit.builders import grok


async def main() -> None:
    c = grok(os.environ.get("XAI_API_KEY", "token"))
    # #region video
    handle = await (
        c.video
        .model("grok-imagine-video")
        .submit("a slow cinematic drone shot flying over snow-capped alpine peaks at golden hour")
    )
    r = await handle.wait()
    v = r.videos[0]
    print(f"url={v.url} duration={v.duration_seconds}s mime={v.mime_type}")
    # #endregion


if __name__ == "__main__":
    asyncio.run(main())
