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
