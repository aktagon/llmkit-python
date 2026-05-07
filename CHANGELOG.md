# Changelog

All notable changes to the Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
