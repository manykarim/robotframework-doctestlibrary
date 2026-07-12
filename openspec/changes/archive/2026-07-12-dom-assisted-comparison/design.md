# Design: dom-assisted-comparison

## Decisions

- **D1 Walker**: one JS function constant (`DOM_WALKER_JS`, arrow function) in
  WebCapture; Browser runs it via `Evaluate JavaScript` (element handle when a
  locator is given), Selenium wraps it as `return (fn)(arguments[0])` with
  `Get WebElement`. Output: JSON string of the semantic tree (whitespace-normalized
  text; SCRIPT/STYLE/hidden elements skipped) — verified identical across engines.
- **D2 Storage**: `{baseline}.dom.json` beside the PNG. Written on baseline
  creation and REFERENCE_RUN, refreshed after every visual PASS (self-healing
  after dashboard accepts). Compared as canonical JSON (parse → re-dump sorted).
- **D3 Verdict flow**: snapshot captured once per keyword call (before the retry
  loop; retries recompare pixels, not DOM). Verdict recorded as
  `context["dom_analysis"] = {"verdict": ..., "changes": [...≤5 deepdiff paths]}`.
- **D4 accept_rendering_only**: wraps the comparison; on AssertionError with
  verdict `identical` → WARN `Rendering-only difference accepted (DOM unchanged)`
  and return. Sidecar keeps status FAIL → reviewable in the dashboard while the
  suite stays green. `missing-baseline`/`changed` verdicts never auto-accept.
- **D5 Scope**: element keyword snapshots the compared element only.

## Risks

- [Canvas/SVG-drawn changes invisible to the DOM] → verdict `identical` although
  content changed; `accept_rendering_only` is opt-in and documented for
  DOM-rendered UIs; pixel comparison alone still fails the test by default.
- [Value-only changes (input typing) count as semantic] → intended: user-visible.
