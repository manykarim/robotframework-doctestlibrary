# Tasks: doctest-dashboard

## 1. Core library: fixes and sidecar (poetry-managed, `DocTest/`)

- [x] 1.1 Fix `_convert_to_pixels` to convert before rounding (`DocumentRepresentation.py:711-723`); add unit tests covering fractional mm/cm/pt at multiple DPIs (25.4 mm @200 DPI == 200 px regression test)
- [x] 1.2 Implement `reference_run` promotion in `compare_images`/`compare_pdf_documents` (save candidate as reference on missing/differing reference, pass keyword); unit + atest robot coverage for both scenarios
- [x] 1.3 Define sidecar schema v1 writer: collect per-page records from `detected_differences` (score, threshold, rectangles), settings, mask data (abstract + resolved pixel areas), timing; write `{OUTPUT_DIR}/doctest_results/{uuid}.json`
- [x] 1.4 Add `result_json` constructor option + `Set Result Json` keyword to `VisualTest` and `PdfTest`; log `DOCTEST_RESULT: <relpath>` INFO message once per comparison
- [x] 1.5 Save separate per-page reference/candidate/diff PNG renderings when `result_json` is enabled (independent of `screenshot_format`), referenced from page records
- [x] 1.6 Unit + atest coverage for sidecar: pass/fail, image/PDF, masked comparison, pabot index prefix; verify exactly one sidecar + log line per comparison and sidecar emitted for passing runs
- [x] 1.7 Update keyword docs; remove stale root `uv.lock`; run full `utest`/`atest` suites

## 2. Dashboard package scaffold (uv-managed, `dashboard/`)

- [x] 2.1 Create `dashboard/pyproject.toml` (PEP 621, uv-managed, package name `doctest-dashboard`, depends on core library) with `uv lock`; document the poetry-core/uv-dashboard split in `dashboard/README.md`
- [x] 2.2 Scaffold FastAPI app (`doctest-dashboard serve`, binds 127.0.0.1, `--host`/`--token` flags) and Vite/React/TS frontend with react-konva; wire static-file serving of the built frontend from the wheel
- [x] 2.3 Pydantic models for sidecar schema v1 + contract test that runs the core library (task 1.x build) and validates real sidecars against the models
- [x] 2.4 SQLite layer (WAL): `runs`, `tests`, `comparisons`, `pages`, `decisions`, `mask_files` tables with thin DAL and migration bootstrap
- [x] 2.5 Test fixtures: pytest helper that executes real robot runs from `atest/` data into temp dirs (passing, failing, masked, expect-error-wrapped, legacy no-sidecar) for use by all backend/e2e suites

## 3. Ingestion and assets

- [x] 3.1 `ResultVisitor` ingester: keyword-level status extraction, sidecar-first via `DOCTEST_RESULT:` message, path resolution relative to output.xml; idempotent re-ingest
- [x] 3.2 HTML `<img>` scraping fallback producing `degraded` records (combined images only)
- [x] 3.3 `doctest-dashboard ingest` CLI + `POST /api/ingest`; `GET /api/runs`, `GET /api/runs/{id}/tests`, `GET /api/tests/{id}`
- [x] 3.4 Asset service: opaque tokens, configured-roots confinement (symlink + `..` rejection tests), cache headers
- [x] 3.5 Backend tests: ingest all fixture variants, idempotency, expect-error keyword status, degraded flagging, traversal 403s

## 4. Review workflow

- [x] 4.1 Run list + test grid UI with status filters and diff thumbnails; unresolved state model
- [x] 4.2 Diff viewer: side-by-side with synced zoom/pan, overlay, blink, swipe; keyboard mode switching; degraded-record honest fallback
- [x] 4.3 Diff-region navigation (next/prev) from sidecar regions
- [x] 4.4 Accept endpoints (`POST /api/pages/{id}/accept`, `POST /api/tests/{id}/accept`): file promotion, SHA-256 before/after audit rows, PDF document-granularity redirect, roots confinement; parity test against `reference_run` output layout
- [x] 4.5 Reject endpoint with reason + bug-bundle ZIP export (ref, cand, diffs, sidecar, decision metadata)
- [x] 4.6 Stale-decision reset on newer-run ingest (changed pages → `unresolved`, history retained)
- [x] 4.7 Backend tests for accept/reject/reset; Playwright journey J1 (ingest→browse→view→accept→assert file changed + audit row) and J2 (reject→download bundle→assert contents)

## 5. Embedded engine: mask preview and recompare

- [x] 5.1 Worker-pool engine service (`ProcessPoolExecutor`, job queue, per-job timeout, bounded concurrency); startup `CapabilityCheck` + capabilities endpoint
- [x] 5.2 `POST /api/mask-preview`: resolve masks via `DocumentRepresentation`, return pixel areas + page DPI; (file, page, engine, pattern) caching
- [x] 5.3 `POST /api/recompare`: re-run stored comparison with adjusted masks/settings into a scratch area (originals untouched), return per-page results + diff image references; result caching on (artifact hashes, masks, settings)
- [x] 5.4 Batch recompare across stored comparisons sharing a masks.json ("which historical failures would this mask suppress")
- [x] 5.5 Backend tests: pattern preview boxes on `Beach_date.png`, recompare flips `birthday_1080` date-diff to PASS with date-pattern mask, cache hit on duplicate request, OCR-unavailable degradation

## 6. Mask editor

- [x] 6.1 Konva canvas: locked page layer, editable mask layer with Transformer, toggleable diff-region overlay, multi-page strip
- [x] 6.2 Mask CRUD: coordinates (drag/handles/synced numeric fields, unit selector with display-only conversion, sidecar DPI shown), area (location/percent panel with band preview), pattern types (regex input + live preview via 5.2)
- [x] 6.3 masks.json I/O: schema-exact load/save (stable key order, pretty-print), shorthand-string import normalization, atomic write + `.bak`, roots confinement; `GET/PUT /api/masks`
- [x] 6.4 Create-mask-from-diff-region action in the diff viewer (pre-seeded coordinates mask + padding, masks.json picker)
- [x] 6.5 Round-trip contract tests: library testdata masks load→save→`IgnoreAreaManager` parity; unit/DPI fidelity at multiple DPIs; property test parse→save→parse stability
- [x] 6.6 Playwright journeys J3 (diff region→editor→adjust→pattern live preview→recompare PASS→save→robot re-run with saved mask passes) and J4 (shorthand import→edit→export round trip)

## 7. Packaging, docs, release readiness

- [x] 7.1 Frontend production build shipped in wheel; `pipx install` smoke test; verify `doctest-dashboard serve` works with no Node present
- [x] 7.2 CI: core suite (poetry) + dashboard suite (`uv sync`, pytest, Playwright headless) as separate jobs
- [x] 7.3 README pointer deprecating `utilities/mask_editor.py`; dashboard user docs (file-mode + `result_json` prerequisites, degraded-mode explanation, team-mode token)
- [x] 7.4 End-to-end verification: full journey suite green against a fresh `uv sync` + `poetry install` environment; security pass on path confinement endpoints
