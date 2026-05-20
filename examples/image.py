"""Text-to-image generation against Google's Nano Banana.

Run: GOOGLE_API_KEY=... python examples/image.py
"""
import asyncio
import os

from llmkit.builders import google


async def main() -> None:
    c = google(os.environ.get("GOOGLE_API_KEY", "k"))
    img = await (
        c.image
        .model("gemini-3.1-flash-image-preview")
        .aspect_ratio("16:9")
        .image_size("2K")
        .generate("A nano banana dish, studio lighting")
    )
    with open("out.png", "wb") as f:
        f.write(img.images[0].bytes)
    print(f"wrote out.png ({len(img.images[0].bytes)} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
