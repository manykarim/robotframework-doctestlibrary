# Proposal: doctest-dashboard

## Why

Reviewing visual comparison failures today happens inside `log.html`, with no way to accept a difference (promote candidate → baseline), reject it with collected bug data, or maintain masks visually. The existing tooling — the PyQt5 `TestResultEvaluator` prototype and the 172-line tkinter `utilities/mask_editor.py` — proves the workflow but lacks persistence, diff overlay modes, multi-user access, and mask awareness. A purpose-built local-first web dashboard closes the review loop (see `docs/doctest-dashboard-proposal.md` for full research).

## What Changes

- **New sidecar output in the core library**: `VisualTest`/`PdfTest` gain an opt-in `result_json` option that writes a machine-readable `comparison_result.json` per comparison (schema v1: status, reference/candidate paths, per-page SSIM and diff regions, resolved masks, settings, DPI) plus one parseable `DOCTEST_RESULT: <relpath>` log line. Verified: all required data (per-page `detected_differences` with `rectangles`, `score`, `threshold`, diff images) already exists in memory at comparison time (`VisualTest.py:406`, `get_diff_rectangles` at `VisualTest.py:2060`).
- **New `doctest-dashboard` package** (separate, uv-managed, PEP 621) providing:
  - **Ingest**: CLI + API ingestion of `output.xml` via `robot.api.ExecutionResult` (verified working against a real run; ingester must read keyword-level status because expect-error wrappers mask test-level status), preferring sidecar JSON when present, with HTML-`<img>`-scraping fallback. SQLite metadata store.
  - **Review**: run/test grid, diff viewer (side-by-side, overlay, blink, swipe), diff-region navigation, per-page accept/reject with baseline promotion, SHA-256 audit trail, bug-data ZIP export.
  - **Mask editor**: react-konva editor reading/writing the exact `IgnoreAreaManager` schema (verified: 4 mask implementations — `coordinates`, `area`, `pattern`/`line_pattern` shared, `word_pattern`; `page` + optional `name` fields; shorthand string normalization via `_parse_mask_string`).
  - **Embedded comparison engine**: the dashboard server imports the library directly (verified: `VisualTest` and `DocumentRepresentation` work fully outside a Robot run — instantiation, comparison, OCR/PDF mask resolution to pixel boxes) to provide live mask preview (`POST /api/mask-preview`) and **real-time re-comparison of past runs with adjusted masks** before committing a mask change.
- **Core library fixes discovered during verification** (small, included in the sidecar PR):
  - `reference_run` is currently a no-op: the flag is read and stored (`VisualTest.py:173-203`) but never used to save candidates as references, despite documented behavior. Restore it, since dashboard accept semantics are defined as its per-page counterpart.
  - `_convert_to_pixels` truncates values to `int` *before* unit conversion (`DocumentRepresentation.py:713`), so `25.4 mm` resolves to 196 px instead of 200 px at 200 DPI (verified experimentally). Fix to convert first, round last — required for mask-editor round-trip fidelity.
- **Deprecation**: `utilities/mask_editor.py` gets a README pointer to the dashboard editor (no removal in v1).
- **Deferred (not in this change)**: Listener v3 live mode, baseline history/rollback, CI gating, AI auto-triage.

## Capabilities

### New Capabilities

- `result-sidecar`: machine-readable per-comparison JSON sidecar emitted by the core library, including the `reference_run` repair and pre-conversion truncation fix that the sidecar's accuracy depends on.
- `dashboard-ingest`: post-execution ingestion of `output.xml` (+ sidecars) into the dashboard's SQLite store; asset serving for screenshots and renderings.
- `dashboard-review`: review workflow — grid, diff viewer modes, diff-region navigation, accept (baseline promotion with audit), reject (bug-data export).
- `mask-editor`: visual mask creation/editing with schema-exact `masks.json` I/O, unit/DPI handling, create-mask-from-diff-region.
- `live-recompare`: embedded comparison engine in the server for live mask preview and on-demand re-comparison of stored runs with adjusted masks.

### Modified Capabilities

(none — no existing specs in `openspec/specs/`)

## Impact

- **Core library** (`DocTest/VisualTest.py`, `DocTest/PdfTest.py`, `DocTest/DocumentRepresentation.py`): additive `result_json` option, `reference_run` implementation, truncation fix. Poetry remains the core package manager (pyproject is `[tool.poetry]`-style; verified `uv lock --check` fails — no `[project]` table; the stray root `uv.lock` is stale and should be removed).
- **New code** under `dashboard/` (FastAPI backend, React/Vite/TypeScript frontend, SQLite): uv-managed PEP 621 package `doctest-dashboard` depending on the core library; frontend built at release time and shipped as static files in the wheel.
- **Dependencies**: core gains none. Dashboard adds fastapi, uvicorn, sqlite (stdlib), and frontend toolchain (dev-time only).
- **Tests**: pytest for sidecar + backend services; full-user-journey end-to-end tests (ingest real `output.xml` → review → accept → verify baseline file changed; edit mask → preview → re-compare → save) using Playwright against the built frontend.
- **Environment facts** (verified): Python 3.13.11 venv, Robot Framework 7.4.2, screenshots logged today are *combined* images only (`_combined`, `_combined_with_diff`, `_absolute_diff`) — separate per-page ref/cand renderings exist only via the sidecar.
