# Proposal: web-ai-review

## Why

The library already ships an optional LLM assist (`llm=True`, `llm_override=True`
on `Compare Images`) that can judge whether detected differences matter. For web
comparisons it works technically (kwargs pass through) but the model judges blind:
it does not know the browser, viewport, URL, or whether the semantic DOM changed —
exactly the signals that separate a false failure from a real one.

## What Changes

- **Context-enriched AI review**: when a comparison carries capture context (web
  keywords always do), it is appended to the LLM's textual summary — including the
  `dom_analysis` verdict when enabled. The model sees "chromium, 1280x720, DOM
  unchanged" next to the diff images.
- **Verified passthrough**: unit tests prove `llm=True llm_override=True` on
  `Compare Page To Baseline` reaches the LLM handler with the enriched notes and
  that an approving decision passes the keyword (fake runtime — no API calls, no
  [ai] dependency in the web CI job; tests skip without pydantic).
- Docs: an "optional AI review" section in the web guide (setup via [ai] extra +
  `DOCTEST_LLM_*`, when to use `llm_override`, cost/graceful-degradation notes).

## Capabilities

### Modified Capabilities

- `web-visual-testing`: AI-assisted review requirement (optional).

## Impact

`DocTest/VisualTest.py` (one enrichment hook in the LLM notes), unit tests,
docs/web-visual-testing.md. No new dependencies; [ai] extra stays optional.
