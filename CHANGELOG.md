# Changelog

All notable changes to the Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
