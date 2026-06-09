# Changelog

All notable changes to the Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.3.0] — 2026-06-09

### Added

- Video generation — `c.video.model(id).submit(prompt)` returns a `VideoHandle` immediately; `await handle.wait(poll_interval=..., request_timeout=...)` polls until the job finishes and returns `VideoResponse(videos: list[VideoData], usage, finish_reason, finish_message)`. Each `VideoData` carries `url`, `mime_type`, and `duration_seconds`. One provider so far: xAI Grok Imagine (`grok-imagine-video`), which delivers a temporary hosted URL — download it yourself.
- Music generation — `c.music.model(id).generate(prompt)` produces audio from a text prompt, with an optional `.lyrics(...)` chain method for models that support vocals. Returns `MusicResponse(audio: list[AudioData], text, usage)` with decoded audio bytes. Three providers: Vertex Lyria 2 (`lyria-002`, instrumental WAV), Google Lyria 3 (`lyria-3-pro-preview` / `lyria-3-clip-preview`, MP3 with lyrics), and MiniMax (`music-2.6`). Instrumental-only models reject lyrics before the request is sent.

## [2.2.0] — 2026-06-07

### Changed

- **Local providers no longer guess a model.** The 2.1.0 daemon-resolved
  default-model behavior is retracted: prompting a local provider (ollama,
  vLLM, llama.cpp, LM Studio, Jan) without choosing a model now raises an
  immediate `ValidationError` — `no model chosen and "<provider>" declares no
  default; pick one (models.live() lists what the daemon serves)` — instead of
  resolving one behind the scenes (or, before 2.1.0, sending a static default
  that could 404). What a local daemon serves is runtime inventory; query it
  with `await c.models.live()` and pass the choice via `.model(...)`.
  `PROVIDERS[name].default_model` is now the empty string for the five local
  providers. Cloud-provider defaults are unchanged.

### Removed

- The `local` field on `ProviderConfig` (added in 2.1.0; no longer needed once
  no behavior dispatches on it).

## [2.1.0] — 2026-06-06

### Added

- `Client.supports(cap)` — public capability query. Answers "will an explicit request for this capability hard-fail pre-flight on this provider?" for `Capability.CACHING` / `BATCHING` / `FILE_UPLOAD` / `IMAGE_GENERATION`; capabilities without a provider-level gate return `True`. Sync, no IO. `Capability` is now re-exported from the package root. Gate optional chain calls with `if c.supports(Capability.CACHING): bot = bot.caching()` instead of importing internals.
- Live model listing for local providers: ollama, vLLM, llama.cpp, LM Studio, and Jan now answer `await c.models.provider(p).list()` (and participate in `await c.models.live()`) via their OpenAI-compatible `/v1/models` endpoints. An unreachable daemon surfaces an explicit per-provider error instead of silently returning nothing.
- Local daemon default-model resolution: when no model is set and the registry default is not installed on the daemon, the first installed model is used; the static default remains the fallback when the daemon is unreachable. Cloud-provider defaults are unchanged.
- Anthropic adaptive thinking surface: `reasoning_effort()` on Anthropic models emits `output_config.effort` plus `thinking: {type: adaptive}` (required by `claude-opus-4-7`; accepted values `low`, `medium`, `high`, `xhigh`, `max`). `thinking_budget()` remains the budget-based control and is not converted.

### Fixed

- Google `reasoning_effort()` sent a field the API rejects; it now maps to `generationConfig.thinkingConfig.thinkingLevel` (accepted values `low`, `high`).

## [2.0.0] — 2026-06-05

### Breaking

- Builder chain methods renamed per the ADR-021 naming convention (bare-noun replacers, `add_*` appenders): `tool()` is now `add_tool()` (Agent builder) and `middleware()` is now `add_middleware()` (Text, Image, Agent, and Upload builders). No back-compat aliases; update call sites. Replacers (`system()`, `model()`, `temperature()`, ...) are unchanged.

### Added

- `Usage.cost` — provider-reported request cost in USD, default `0.0`. Populated only when the provider itself reports cost in its usage payload: OpenRouter (`usage.cost`; the request must opt into usage accounting by sending `usage: {include: true}` — llmkit does not add this automatically) and xAI Grok (`usage.cost_in_usd_ticks`, converted at 1 USD = 1e10 ticks). Providers that do not report cost (Anthropic, OpenAI, Google, and others) always return `0.0` — this passes through the provider's own figure; it is not a pricing table.
- Model catalogue (ADR-019): `c.models` and `c.providers` namespaces on `Client`. Static catalogue via `c.models.list()` / `c.models.get(id)` / `c.models.with_capability(cap)` (`ModelInfo` carries `context_window`, `max_output`, `capabilities`, `display_name`). Live per-provider listing via `await c.models.provider(p).list()` / `.get(id)` against the provider's models endpoint (`.raw()` for the unparsed payload; raises `ErrModelsNotSupported` for providers without a live endpoint). Cross-provider sweep via `await c.models.live()` returning `LiveResult` with typed per-provider errors. `c.providers.list()` / `c.providers.supported()` enumerate providers.
- Conversation history (ADR-020): public `Message` struct and `history(*msgs)` chain method on the Text and Agent builders for seeding multi-turn context.
- Stable message wire format (ADR-023): versioned serialization for `Message` history with typed decode errors (`UnsupportedWireVersionError`, `MissingWireVersionError`, `UnknownWireKeyError`) instead of silent misparses.
- `Response.finish_reason` and `Response.finish_message` — provider stop signal + free-text explanation passed through verbatim on `c.text.prompt()`, `c.agent.prompt()`, `c.text.batch()`, and `c.text.stream()` (the latter via the trailing `TextStream.response.finish_reason`). Examples: Anthropic `stop_reason`, OpenAI `choices[0].finish_reason`, Google `candidates[0].finishReason`. Default empty string; populated only when the provider response carries a signal. Streaming uses ADR-013's `event_name:json.path` locator — Anthropic captures from the `message_stop` event body; OpenAI/Grok/Google use last-non-empty-wins on the data frames; Google additionally filters `FINISH_REASON_UNSPECIFIED`. Bedrock Converse streaming is not yet wired.
- `ImageResponse.finish_reason` and `ImageResponse.finish_message` — same shape on `c.image.generate()`. Google populates both (including `IMAGE_OTHER` / `SAFETY` / `MAX_TOKENS` reasons that previously vanished into "no image returned"); Vertex Imagen surfaces `predictions[0].raiFilteredReason` as `finish_reason`; OpenAI Images API and xAI Grok have no equivalent fields and leave them empty. Callers can now render a useful message when `len(resp.images) == 0` instead of synthesizing one.

### Fixed

- `safety_settings()` chain method on the Text builder raised `TypeError` at the terminal — `prompt()` did not accept the keyword the chain method passed. Caught by the cross-SDK wire-conformance suite (ADR-028 M2).
- `schema()` structured-output chain method on the Text builder is now applied to the request body (previously silently dropped); Google's structured-output layout corrected.

## [1.0.0] — 2026-05-09

### Breaking

- Legacy free-function layer removed from the public API (plan-018 D3, ADR-010). `llmkit.prompt`, `llmkit.prompt_stream`, `llmkit.generate_image`, `llmkit.upload_file`, `llmkit.prompt_batch`, `llmkit.submit_batch`, `llmkit.wait_batch`, the `llmkit.Agent` class, and the `Text(s)` / `Image(m, b)` Part constructors are no longer in the public API. Use the typed builder:

  ```python
  from llmkit.builders import new_client
  c = new_client("anthropic", api_key)
  resp = await c.text.system("...").temperature(0.7).prompt("hello")
  ```

  - `c.text.<chain>.prompt(msg)` — replaces `llmkit.prompt`.
  - `c.text.<chain>.stream(msg)` — replaces `llmkit.prompt_stream`; returns an async iterator.
  - `c.image.model(id).<chain>.generate(msg)` — replaces `llmkit.generate_image`.
  - `c.upload.bytes(b).filename(n).run()` — replaces `llmkit.upload_file`.
  - `c.text.<chain>.batch(*prompts)` / `.submit_batch(*prompts).wait()` — replaces the batch trio.
  - `c.agent.<chain>.prompt(msg)` / `c.agent.reset()` — replaces the `Agent` class.

- All typed-builder terminals are async; legacy synchronous callers must wrap with `asyncio.run(...)` or use within an existing event loop.

### Added

- ADR-011 chain-field propagation lint integrated into `make check`.
- All eight sampling/decoding chain methods (`top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `seed`, `stop_sequences`, `thinking_budget`, `reasoning_effort`) now thread through to the wire body. They had been silently dropping since plan-016 phase 2b.
- `*Agent` typed builder now propagates `caching()` to the underlying agent (D3.0 wired text but missed agent).
- `Agent.max_tool_iterations(n)` chain method exposes the tool-loop depth cap (default 10) on the typed builder.
- `Upload.bytes()` is now wired end-to-end alongside `path()`. The internal `upload_file(provider, source, ...)` helper takes a single positional that can be a `str` / `os.PathLike` (read from disk) or `bytes` / `bytearray` (uploaded directly with the chained `filename()`). `mime_type()` overrides the filename-extension–based detection.
- `TextStream` trailing-handle class. Iterate via `async for chunk in stream` to consume chunks; `stream.response` (property) returns the accumulated `Response` (text + tokens) once iteration ends, and `stream.error` exposes any terminal exception. Implements `__aiter__` so existing `async for` loops keep working.

### Changed

- **Breaking**: `c.text.stream(msg)` now returns `TextStream` instead of being an async generator (`AsyncIterator[str]`). The class still implements `__aiter__` so existing iteration code is source-compatible; type hints that referenced `AsyncIterator[str]` for the return value should be updated to `TextStream`.

### Removed

- `caching()` chain method on the `Image` builder. The legacy `generate_image` runtime never accepted a caching option, so the chain method had been a silent no-op.
- `Text(s)` and `Image(m, b)` Part constructor functions in `image.py`. Construct `Part(text="...")` and `Part(image=MediaRef(mime_type=m, bytes=b))` directly if assembling Part lists manually; the typed-builder accumulators are the canonical path.

## [0.3.0] — 2026-05-08

### Breaking

- `ImageRequest.reference_images` (and the `ImageInput` type) is removed. Use `parts: list[Part]` instead, with the package-level `Text(...)` and `Image(...)` constructors. Migration: `ImageRequest(prompt="X", reference_images=[ImageInput(mime_type=m, data=b)])` becomes `ImageRequest(parts=[Text("X"), Image(m, b)])`. Pure text-to-image callers using only `prompt="X"` are unaffected.
- `ImageRequest` now requires exactly one of `prompt` or `parts` to be set (XOR). Both empty or both set raises `ValidationError`.
- `llmkit.Image` (the legacy text-generation vision-input dataclass on `Request.images`) is renamed to `llmkit.InputImage`. Frees the `Image()` Part constructor name. The `Image` symbol is now a function (`def Image(mime: str, data: bytes) -> Part`).
- Multi-reference compositional generation now works by ordering the parts list (e.g., `[Text("Person:"), Image(mime, ref_a), Text("Outfit:"), Image(mime, ref_b), Text("Generate ...")]`) — the wire shape preserves caller-controlled ordering. See ADR-008.

### Added

- `Part`, `MediaRef` dataclasses and `Text(str) -> Part` / `Image(str, bytes) -> Part` constructors at the package level. Universal multimodal atom shared across capabilities.

### Fixed

- `Options(thinking_budget=N)` now produces `{"thinking": {"budget_tokens": N, "type": "enabled"}}` for Anthropic, instead of a flat `"thinking.budget_tokens"` key the server silently ignored. **Behaviour change**: callers that already passed `thinking_budget` will now actually engage Anthropic extended thinking on supported models — expect higher latency and additional reasoning tokens in `Response.tokens.reasoning`. No code change required to opt in. To opt out, omit the option.
- Provider option overrides with dotted JSON keys (e.g. Google's `thinkingConfig.thinkingBudget`) are now correctly nested. Previously such options were dropped silently.

### Added

- `merge_into_parent` helper in `llmkit.paths` for attaching sibling fields to a dotted JSON path.

## [0.1.0] — initial release

See git history.
