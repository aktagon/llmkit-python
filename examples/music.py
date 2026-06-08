"""Text-to-music generation against Vertex Lyria 2.

Run: GOOGLE_ACCESS_TOKEN=... python examples/music.py

lyria-002 is instrumental-only — no .lyrics() chain method.
"""
import asyncio
import os

from llmkit.builders import vertex


async def main() -> None:
    c = vertex(os.environ.get("GOOGLE_ACCESS_TOKEN", "token"))
    # #region music
    r = await (
        c.music
        .model("lyria-002")
        .generate("a calm instrumental, warm piano and soft strings")
    )
    with open("out.wav", "wb") as f:
        f.write(r.audio[0].bytes)
    # #endregion
    print(f"wrote out.wav ({len(r.audio[0].bytes)} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
