# llmkit (Python)

Unified async LLM client library. One API, multiple providers, zero external dependencies (stdlib only — no `httpx`, no `pydantic`).

The code is generated + hand-coded with the help of AI: a typed provider matrix and the typed-builder API surface are generated from a single source of truth, while request building, transport, streaming, caching, batching, and tool-loop behavior are hand-coded on top.

Shares a code-generation pipeline with the [Go](https://github.com/aktagon/llmkit-go), [TypeScript](https://github.com/aktagon/llmkit-ts), and [Rust](https://github.com/aktagon/llmkit-rust) SDKs.

## Install

```bash
pip install llmkit
# or with uv:
uv add llmkit
```

Python 3.10 or later.

## Quick Start

```python
import os
import asyncio
from llmkit.builders import anthropic

async def main():
    c = anthropic(os.environ["ANTHROPIC_API_KEY"])
    resp = await (
        c.text()
        .system("Be concise.")
        .temperature(0.3)
        .prompt("Say hi")
    )
    print(resp.text)
    print(resp.usage.input, "input tokens")

asyncio.run(main())
```

The typed builder is the only public surface as of v1.0.0. One mental model — `client.<capability>().<chain>.<terminal>` — across every capability.

## Providers

Per-provider factory functions:

```
ai21      anthropic  azure     bedrock   cerebras  cohere    deepseek
doubao    ernie      fireworks google    grok      groq      lmstudio
minimax   mistral    moonshot  ollama    openai    openrouter
perplexity qwen      sambanova together  vllm      yi        zhipu
```

Or use the generic `new_client(name, api_key)`. 27 providers, 4 API shapes (OpenAI-compatible, Anthropic Messages, Google Generative AI, AWS Bedrock Converse). Bedrock auth uses SigV4; other providers use API-key auth.

## API

### Text — one-shot prompt

```python
resp = await (
    c.text()
    .system("You are helpful")
    .temperature(0.7)
    .max_tokens(200)
    .prompt("What is 2+2?")
)

print(resp.text)              # "4"
print(resp.usage.input)       # prompt tokens
print(resp.usage.output)      # completion tokens
print(resp.usage.cache_read)  # tokens served from cache
print(resp.usage.cache_write) # tokens written to cache (Anthropic explicit)
print(resp.usage.reasoning)   # internal reasoning tokens (OpenAI o-series, Gemini 2.5+)
```

Capability-scoped fields (`cache_read`, `cache_write`, `reasoning`) are zero when the provider doesn't report them separately.

### Stream — async iteration with trailing handle

```python
stream = c.text().system("Be brief").stream("Tell me a joke")
async for chunk in stream:
    print(chunk, end="", flush=True)
print("\nUsage:", stream.response().usage)
```

`TextStream` implements `__aiter__`. After iteration completes, `stream.response()` returns the final `Response` (with token counts) and `stream.error()` returns any terminal error. Handles both Anthropic-style typed events and OpenAI-style data-only frames internally.

### Agent — tool loop

```python
from llmkit import Tool

def add(args):
    return str(args["a"] + args["b"])

add_tool = Tool(
    name="add",
    description="Add two numbers",
    schema={
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
    },
    run=add,
)

bot = (
    c.agent()
    .system("You are a calculator.")
    .tool(add_tool)
    .max_tool_iterations(5)
)
resp = await bot.prompt("What is 2+3?")
print(resp.text)
```

`*Agent` is **stateful** — repeated `bot.prompt(...)` calls accumulate history. Chain methods (`.system(...)`, `.tool(...)`) clone and reset state, so a forked builder gets a fresh conversation. `bot.reset()` clears state without dropping chained config.

Tool dispatch covers Anthropic `tool_use`, OpenAI `tool_calls`, Google `functionCall`, and Bedrock Converse `toolUse`.

### Image — text-to-image and edit

Supports Google's Nano Banana 2 (`gemini-3.1-flash-image-preview`) and Pro (`gemini-3-pro-image-preview`); OpenAI's `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1`, and `gpt-image-1-mini`; xAI's `grok-imagine-image-quality`; Google Cloud Vertex AI's Imagen 3 / Imagen 4 (`imagen-3.0-generate-002`, `imagen-3.0-fast-generate-001`, `imagen-4.0-generate-preview-06-06`).

```python
from llmkit.builders import google

c = google(os.environ["GOOGLE_API_KEY"])
img = await (
    c.image()
    .model("gemini-3.1-flash-image-preview")
    .aspect_ratio("16:9")
    .image_size("2K")
    .generate("A nano banana dish, studio lighting")
)

with open("out.png", "wb") as f:
    f.write(img.images[0].data)
```

For compositional editing, chain `.text(...)` and `.image(mime, bytes)` to interleave references with descriptions:

```python
await (
    c.image()
    .model("gemini-3.1-flash-image-preview")
    .text("Person:")
    .image("image/png", person_bytes)
    .text("Outfit:")
    .image("image/png", outfit_bytes)
    .generate("Generate the person wearing the outfit.")
)
```

Aspect ratios and sizes validate against a per-model whitelist before the HTTP request. Empty whitelists mean "no client-side check; pass through" — providers like OpenAI accept arbitrary sizes within documented bounds (max edge ≤3840, both edges multiples of 16, ratio ≤3:1, total pixels 655K–8.3M), so the SDK trusts the API boundary instead of carrying a stale list.

For OpenAI, the chain dispatches automatically — no image parts hits `/v1/images/generations` (JSON), one or more image parts hits `/v1/images/edits` (multipart/form-data with one `image[]` field per reference, in caller order).

Provider knobs are typed chain methods:

| Method               | Provider support            | Wire field       |
| -------------------- | --------------------------- | ---------------- |
| `.quality(s)`        | OpenAI gpt-image-\*         | `quality`        |
| `.output_format(s)`  | OpenAI gpt-image-\*         | `output_format`  |
| `.background(s)`     | OpenAI gpt-image-\*         | `background`     |
| `.count(n)`          | OpenAI + xAI Grok           | `n`              |
| `.mask(mime, bytes)` | OpenAI gpt-image-\* (edits) | multipart `mask` |

The chain validates per provider — calling `.quality(...)` on a Google or xAI builder raises `ValidationError` immediately, no HTTP round-trip. Knobs without typed methods (OpenAI: `output_compression`, `moderation`) remain reachable via `.extra_fields(...)`, which is unvalidated and freeform.

```python
from llmkit.builders import openai

c = openai(os.environ["OPENAI_API_KEY"])
resp = await (
    c.image()
    .model("gpt-image-2")
    .image_size("1024x1024")
    .quality("high")
    .count(4)
    .generate("A red circle on a white background")
)
```

OpenAI gpt-image-\* models require organization verification — see [platform.openai.com/docs/guides/your-data#organization-verification](https://platform.openai.com/docs/guides/your-data#organization-verification).

Up to 14 reference images per Google request, 16 per OpenAI request.

#### Vertex AI Imagen (Google Cloud)

Vertex Imagen uses the `:predict` endpoint family and OAuth bearer auth instead of API keys. The SDK takes a bearer token (string); caller manages OAuth refresh externally (e.g. `gcloud auth print-access-token`, service-account JSON, or workload identity).

```python
import os
from llmkit.builders import vertex

# Caller substitutes {project_id} and {location} before passing the URL.
base_url = (
    "https://us-central1-aiplatform.googleapis.com"
    "/v1/projects/my-gcp-project/locations/us-central1/publishers/google/models"
)

c = vertex(os.environ["VERTEX_BEARER_TOKEN"]).with_base_url(base_url)

resp = await (
    c.image()
    .model("imagen-3.0-generate-002")
    .aspect_ratio("16:9")
    .count(2)
    .generate("A red circle")
)
```

Edit-mode (single image into `instances[0].image`) and inpainting (`.mask(mime, bytes)` into `instances[0].mask.image`) work the same way. Imagen-specific knobs like `negativePrompt` and `safetySetting` are reachable through `.extra_fields(...)` — they spread into the request's `parameters` block. Vertex's `:predict` response does not carry token counts; `resp.tokens` stays zero.

### Safety Settings

Control content filtering for Gemini providers. `safety_settings` applies to text
generation, streaming, agents, and Gemini image generation. `safety_filter` applies
to Vertex Imagen only.

```python
from llmkit.builders import google, vertex
from llmkit.types import (
    SafetySetting,
    HARM_CATEGORY_DANGEROUS_CONTENT,
    HARM_CATEGORY_HARASSMENT,
    HARM_BLOCK_THRESHOLD_NONE,
    HARM_BLOCK_THRESHOLD_HIGH_ONLY,
    IMAGE_SAFETY_FILTER_BLOCK_FEW,
)

# Gemini text or agent
c = google(os.environ["GOOGLE_API_KEY"])
resp = await (
    c.text()
    .safety_settings([
        SafetySetting(category=HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HARM_BLOCK_THRESHOLD_NONE),
        SafetySetting(category=HARM_CATEGORY_HARASSMENT, threshold=HARM_BLOCK_THRESHOLD_HIGH_ONLY),
    ])
    .prompt("Write a story")
)

# Vertex Imagen
vc = vertex(os.environ["VERTEX_BEARER_TOKEN"])
img = await (
    vc.image()
    .model("imagen-3.0-generate-002")
    .safety_filter(IMAGE_SAFETY_FILTER_BLOCK_FEW)
    .generate("A landscape")
)
```

`safety_settings` on Vertex Imagen and `safety_filter` on non-Imagen providers raise
a `ValidationError`. The `HARM_CATEGORY_*`, `HARM_BLOCK_THRESHOLD_*`, and
`IMAGE_SAFETY_FILTER_*` constants cover all documented values; raw strings also work.

### Upload — Path or Bytes

```python
from llmkit.builders import openai

c = openai(os.environ["OPENAI_API_KEY"])

# from a path
file = await c.upload().path("./data.pdf").run()

# from bytes (filename required)
file2 = await (
    c.upload()
    .bytes(buf)
    .filename("report.pdf")
    .mime_type("application/pdf")
    .run()
)
```

### Batches

```python
results = await (
    c.text()
    .system("Be brief")
    .batch(["Translate hello to French", "Translate hello to Spanish"])
)
for r in results:
    print(r.text)
```

`.batch(prompts)` is `.submit_batch(prompts)` + `handle.wait()`. Use `.submit_batch(prompts)` to get a `BatchHandle` you can persist, then call `await handle.wait()` later. Both inline (Anthropic) and file-reference (OpenAI two-hop) flows are handled internally.

### Caching

```python
# Anthropic — explicit cache_control wrap of the system prompt:
await c.text().system(long_sys_prompt).caching().prompt("...")

# OpenAI — automatic server-side caching (caching() is a hint; reads
# surface in resp.usage.cache_read regardless):
await c.text().system(long_sys_prompt).caching().prompt("...")

# Google — pre-flight POST creates a cachedContents resource, then the
# main call references it. Google requires ~1k+ tokens of system prompt:
await c.text().system(big_sys_prompt).caching().prompt("...")
```

The mode is provider-specific and inferred from the provider config. The default TTL comes from `src/llmkit/providers/generated/caching.py` (Google: 3600s).

## Options

Across every `*Text` / `*Agent` builder:

| Concept           | Method                 |
| ----------------- | ---------------------- |
| System prompt     | `.system(s)`           |
| Model override    | `.model(name)`         |
| Sampling          | `.temperature(t)`      |
| Token cap         | `.max_tokens(n)`       |
| Caching           | `.caching()`           |
| Conversation hist | `.history(msgs)`       |
| Structured output | `.schema(json)`        |
| Middleware hooks  | `.middleware(fns)`     |
| Reasoning effort  | `.reasoning_effort(l)` |
| Thinking budget   | `.thinking_budget(n)`  |

Sampling hyperparameters (`.top_p`, `.top_k`, `.seed`, `.frequency_penalty`, `.presence_penalty`, `.stop_sequences`) are validated per provider; unsupported options raise `ValidationError` rather than silently dropping.

The Image builder has a narrower set: `.model`, `.aspect_ratio`, `.image_size`, `.include_text`, `.text`, `.image`, `.middleware`. Upload: `.path`, `.bytes`, `.filename`, `.mime_type`, `.middleware`.

## Middleware

```python
from llmkit import Event, MiddlewareFn

def log_usage(e):
    if e.op == "llm_request" and e.phase == "post":
        print(f"{e.provider}/{e.model}: {e.usage.input} in, {e.usage.output} out")
    return None

await c.text().middleware([log_usage]).prompt("...")
```

Pre-phase middleware can veto by returning a non-None error message; post-phase runs for observation only. Wired at six sites: text prompt, text stream, agent LLM call, agent tool execution, upload, batch submit, Google resource caching pre-flight.

## Self-hosted endpoints

```python
from llmkit.builders import openai

c = openai("anything").with_base_url("http://localhost:8080/v1")
```

Works for any OpenAI-compatible server (vLLM, LM Studio, Ollama, corporate gateways).

## Architecture

- **Generated** (`src/llmkit/providers/generated/*.py`, `src/llmkit/builders/__init__.py`) — per-provider config + the typed-builder API surface. Pure data and class skeletons, no business logic.
- **Hand-coded** (`src/llmkit/{client,types,errors,http,transforms,middleware,caching,batch,agent,sigv4,paths}.py`, plus `builders/{text,agent,image,stream,batch,upload}.py`) — HTTP, request shaping, SSE consumer, agent tool loop, SigV4 signing, caching, batch lifecycle, multipart upload, middleware fanout, builder terminals.

Transforms dispatch on config fields (`system_placement`, `wraps_options_in`, `auth_scheme`), not provider names.

## Mirror

This repo is a read-only mirror of a private monorepo. File issues here; code patches should target the private source via `christian@aktagon.com`.

## License

MIT
