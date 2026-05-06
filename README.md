# llmkit (Python)

Unified LLM client library. One API, multiple providers, zero external dependencies.

The code is generated + hand-coded: a typed provider matrix is generated from a single source of truth, while request building, transport, streaming, caching, batching, and tool-loop behavior are hand-coded on top.

## Install

```bash
pip install llmkit
# or with uv:
uv add llmkit
```

Python 3.10 or later.

## Quick start

```python
import os
import llmkit

resp = llmkit.prompt(
    provider=llmkit.Provider(name="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"]),
    request=llmkit.Request(system="Be concise.", user="Say hi"),
    temperature=0.3,
)
print(resp.text)
print(resp.tokens.input, "input tokens")
```

## Streaming

```python
def on_chunk(text: str) -> None:
    print(text, end="", flush=True)

resp = llmkit.prompt_stream(
    provider=llmkit.Provider(name="openai", api_key=os.environ["OPENAI_API_KEY"]),
    request=llmkit.Request(user="Write a haiku about caching."),
    on_chunk=on_chunk,
)
```

## Tool-calling agent

```python
def weather(args):
    return f"It's sunny in {args['city']}."

agent = llmkit.Agent(
    llmkit.Provider(name="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"]),
)
agent.set_system("You can look up weather with the 'weather' tool.")
agent.add_tool(
    llmkit.Tool(
        name="weather",
        description="Get weather for a city",
        schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        run=weather,
    )
)
resp = agent.chat("What's the weather in Helsinki?")
print(resp.text)
```

## Image generation

Generate images from text, optionally conditioned on reference images for
editing or composition. Currently supports Google's Nano Banana 2
(`gemini-3.1-flash-image-preview`) and Pro (`gemini-3-pro-image-preview`).

```python
resp = llmkit.generate_image(
    provider=llmkit.Provider(name="google", api_key=os.environ["GOOGLE_API_KEY"]),
    request=llmkit.ImageRequest(
        prompt="A nano banana dish in a fancy restaurant",
        model="gemini-3.1-flash-image-preview",
    ),
    aspect_ratio="16:9",
    image_size="2K",
)
with open("out.png", "wb") as f:
    f.write(resp.images[0].data)
```

Pass reference images to edit or compose:

```python
edited = llmkit.generate_image(
    provider=provider,
    request=llmkit.ImageRequest(
        prompt="Add snow and frost; overcast sky.",
        model="gemini-3.1-flash-image-preview",
        reference_images=[
            llmkit.ImageInput(mime_type="image/png", data=png_bytes),
        ],
    ),
)
```

Aspect ratios and sizes are validated against a per-model whitelist before
the HTTP request — `image_size="512"` on Pro raises `ValidationError`
without paying for a 4xx round-trip.

| Model                 | Aspect ratios                                                               | Sizes           |
| --------------------- | --------------------------------------------------------------------------- | --------------- |
| Nano Banana 2 (Flash) | 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9, **1:4, 4:1, 1:8, 8:1** | 512, 1K, 2K, 4K |
| Nano Banana Pro       | 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9                         | 1K, 2K, 4K      |

Up to 14 reference images per request.

## Batching

```python
requests = [llmkit.Request(user=f"Summarize: {text}") for text in corpus]
responses = llmkit.prompt_batch(
    provider=llmkit.Provider(name="anthropic", api_key=key),
    requests=requests,
)
```

## Providers

OpenAI, Anthropic, Google, Grok, Bedrock, OpenRouter, Groq, DeepSeek, Cohere, Mistral, Together, Fireworks, Cerebras, Doubao, Ernie, Moonshot, Qwen, Perplexity, SambaNova, Yi, AI21, Zhipu, MiniMax, Azure, Ollama, LM Studio, vLLM.

Each provider has a default model, auth scheme, and feature matrix (caching, batching, streaming, tool calls, structured output, file upload) discoverable via `llmkit.PROVIDERS`.

## API surface

Entry points: `prompt`, `prompt_stream`, `upload_file`, `prompt_batch`, `submit_batch`, `wait_batch`, `Agent`.

Types: `Provider`, `Request`, `Response`, `Message`, `File`, `Image`, `Tool`, `Options`, `Usage`, `Event`, `MiddlewareFn`, `MiddlewareOp`, `MiddlewarePhase`, `ProviderName`, `ProviderConfig`, `PROVIDERS`.

Errors: `APIError`, `ValidationError`, `MiddlewareVetoError`.

## License

MIT.
