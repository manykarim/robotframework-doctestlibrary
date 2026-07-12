# Design: web-ai-review

## Decisions

- **D1 Enrichment point**: `llm_runtime_notes` in `compare_images` — when
  `capture_context` is present, append `Capture context: {json}` (includes the
  `dom_analysis` verdict that WebVisualTest stores into the context before
  comparing). One line; benefits any caller that passes `context=`.
- **D2 No web-specific LLM code path**: the web keywords inherit `llm`,
  `llm_override`, `llm_prompt` through kwargs passthrough — verified by test, not
  duplicated in code.
- **D3 Test isolation**: monkeypatch `_load_visual_llm_runtime` and
  `load_llm_settings` with fakes (existing pattern from test_llm_support);
  `pytest.importorskip("pydantic")` keeps the web CI job (no [ai] deps) green.

## Risks

- [LLM cost/latency in retry loop] → the LLM runs inside compare_images on each
  attempt; documented recommendation: use `llm=True` with `retry_timeout=0` or
  rely on its final-attempt behavior consciously.
