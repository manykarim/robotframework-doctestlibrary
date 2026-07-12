# Proposal: web-visual-testing

## Why

Using doctestlibrary for web-app VRT today means hand-rolling everything around the
engine: taking screenshots with Browser/SeleniumLibrary, inventing baseline path
conventions, pre-creating baselines manually, and living with capture-timing flake.
Competing tools (Playwright `toHaveScreenshot`, BackstopJS, Percy) auto-create
baselines, retry until stable, and name baselines for you. The engine, mask system
and review dashboard already beat those tools — the missing piece is the capture and
baseline workflow (docs/web-vrt-analysis.md).

## What Changes

- New RF library **`DocTest.WebVisualTest`** (extends `VisualTest` — every existing
  comparison option works) with keywords:
  - `Compare Page To Baseline    name    [**kwargs]`
  - `Compare Element To Baseline    locator    name    [**kwargs]`
- **Capture adapters** for Browser Library and SeleniumLibrary, auto-detected from
  the running suite via `BuiltIn.get_library_instance`, invoked purely through
  `BuiltIn.run_keyword` → no new dependencies, no imports of either library.
  Browser page captures default to `fullPage=True, scale=css` (DPR-independent,
  verified by experiment) with `disableAnimations=True`; Selenium captures
  viewport/element (full-page not natively supported — documented).
- **Baseline lifecycle**: baselines live at `{baseline_directory}/{name}.png`
  (import parameter, default `visual_baselines` under the suite's execution dir).
  Missing baseline → the capture becomes the baseline, keyword passes with a WARN
  (`baseline_created`); existing baseline → normal `Compare Images` comparison, so
  REFERENCE_RUN and dashboard accept promote web baselines exactly like any others.
- **Stabilization retry**: on comparison failure the page/element is recaptured and
  recompared until `retry_timeout` (default `3s`) expires — Playwright-style
  auto-retry that kills timing flake without hiding real diffs.
- Tests: unit tests with fake adapters (baseline creation, retry, path safety),
  acceptance suites running REAL Browser and SeleniumLibrary sessions against a
  local deterministic page; new `web` CI job (chrome + rfbrowser init).
- Docs: README section + `docs/web-visual-testing.md` quickstart for both libraries.

## Capabilities

### New Capabilities

- `web-visual-testing`: capture-agnostic page/element baseline comparison keywords
  with automatic baseline creation and stabilization retry.

## Impact

New `DocTest/WebVisualTest.py`, `DocTest/WebCapture.py`; `pyproject.toml`
dependency-group `web-test` (test-only); `.github/workflows` web job;
`atest/web/`, `utest/test_web_visual.py`; README + docs. No changes to existing
libraries or the dashboard.
