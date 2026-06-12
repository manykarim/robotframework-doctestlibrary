# Design: doctest-dashboard

## Context

The library produces everything a review workflow needs (diff screenshots, SSIM scores, diff rectangles, OCR text, masks) but exposes none of it in machine-readable form: failures raise `AssertionError("The compared images are different.")` while details land as free-text INFO messages (`VisualTest.py:888-890`, `_raise_comparison_failure` at `VisualTest.py:2014`). Verified environment: Python 3.13.11, Robot Framework 7.4.2, poetry-managed core (`[tool.poetry]` pyproject; uv cannot manage it — `uv lock --check` fails with "No `project` table").

Experiments performed before this design (all on real code/runs, see proposal):

1. **Engine runs outside Robot.** `VisualTest()` instantiates without a Robot context, `compare_images()` raises/passes normally, screenshots are written relative to CWD. `DocumentRepresentation(file, ignore_area=...)` resolves pattern masks via OCR to pixel boxes (`pages[0].pixel_ignore_areas`). This makes the embedded comparison engine and live mask preview feasible with no library changes.
2. **`output.xml` ingestion works.** `robot.api.ExecutionResult` + `ResultVisitor` extracts statuses and `<img src>` paths from HTML messages. Caveats found: image paths are relative to the output dir; only *combined* images are logged (`_combined`, `_combined_with_diff`, `_absolute_diff` suffixes) — no separate ref/cand renderings; tests wrapped in `Run Keyword And Expect Error` PASS at test level despite failing comparisons, so ingestion must inspect keyword-level results.
3. **Two core defects found.** `reference_run` is read/stored but never consumed (no save-as-reference logic exists anywhere in `DocTest/`); `_convert_to_pixels` (`DocumentRepresentation.py:711-723`) truncates to `int` *before* multiplying by the DPI constant, so `25.4 mm` @200 DPI → 196 px instead of 200 px.
4. **Mask schema ground truth.** Four implementations: `coordinates` (px/mm/cm/pt), `area` (location+percent, default 10), `pattern`/`line_pattern` (identical handling: line bboxes from PDF text, word bboxes from OCR), `word_pattern` (word bboxes, PDF path). No schema validation — invalid entries silently skip or raise `KeyError`. Shorthand `top:10;bottom:10` parses via `_parse_mask_string` into area masks.

## Goals / Non-Goals

**Goals:**
- A locally runnable web dashboard (single process, zero infrastructure) for reviewing, accepting/rejecting, and mask-editing.
- A structured sidecar (`comparison_result.json`) emitted by the core library as the durable data contract.
- Live mask preview and re-comparison with adjusted masks, server-side, using the library itself as the engine.
- Plain-file baselines: accepting a change produces a normal git diff in the user's test-data tree.
- Full-journey automated test coverage: backend (pytest, real robot-generated fixtures) and browser end-to-end (Playwright).

**Non-Goals:**
- Listener v3 live streaming, CI gating, baseline history/rollback, AI auto-triage (deferred to a future change).
- Cloud/SaaS, auth beyond a static token, replacing `log.html`.
- Reviewing `embed_screenshots=True` runs (documented prerequisite: file mode).

## Decisions

### D1 — Repository layout and tooling: monorepo subpackage, uv for the dashboard, poetry stays for core

The dashboard lives in `dashboard/` in this repo as its own distributable package `doctest-dashboard` with a PEP 621 `pyproject.toml` managed by **uv** (`dashboard/backend` Python + `dashboard/frontend` Vite/React/TS). The core library stays poetry — verified that uv cannot operate on the poetry-style pyproject, and converting the core's metadata is out of scope. The stale root `uv.lock` (resolves nothing usable; pyproject has no `[project]` table) is deleted. Alternative considered: separate repository — rejected for v1 because the sidecar schema, mask schema contract tests, and e2e fixtures all need the library source in lockstep.

### D2 — Data contract: sidecar JSON, schema v1, written by the library

Opt-in `result_json=True` on `VisualTest`/`PdfTest` (constructor + `set_result_json` keyword). Per comparison, write `{OUTPUT_DIR}/doctest_results/{uuid}.json` and log `DOCTEST_RESULT: <relpath>` as a plain INFO message. Schema v1 carries: `schema_version`, keyword, library, status, reference/candidate (path, page count, DPI), settings (move tolerance, check_text_content, watermarks, screenshot_format), resolved masks (the `abstract_ignore_areas` plus per-page `pixel_ignore_areas`), per-page records (status, SSIM `score`, `threshold`, `diff_regions` from `get_diff_rectangles`, image paths for reference/candidate/diff renderings saved separately), and timing. The in-memory `detected_differences` list already holds rectangles, scores, thresholds, and diff images (`VisualTest.py:406ff`), so this is serialization, not new computation. Pages that pass get a record too (status + score, no diff images) so the dashboard can show per-page state. Sidecar must also save **separate** per-page reference and candidate renderings (PNG regardless of `screenshot_format`, since JPEG-70 degrades review) — verified that today only combined images exist.

Pydantic models define the schema in the dashboard package; the core library writes plain dicts (core gains no dependency) and the contract is enforced by tests that run the core and validate output against the dashboard's models.

### D3 — Ingestion: `ResultVisitor` walking keyword bodies, sidecar-first

`doctest-dashboard ingest <output.xml>` (and `POST /api/ingest`). The visitor walks test bodies recursively to keyword level, keying on `DocTest.VisualTest`/`DocTest.PdfTest` keyword owners, and uses **keyword-level status** (verified necessity: expect-error wrappers). For each comparison keyword: prefer the `DOCTEST_RESULT:` message → load sidecar; fallback to scraping `<img src>` HTML messages (evaluator-style) yielding degraded records (combined images only, no per-page granularity, no diff regions — dashboard renders these read-only with an upgrade hint). Paths resolve relative to the `output.xml` location. Pabot-merged outputs work because each comparison carries its own sidecar.

### D4 — Storage: SQLite via `sqlite3`+thin DAL, WAL mode; baselines stay plain files

Tables: `runs`, `tests`, `comparisons`, `pages`, `decisions` (action, actor, reason, prev_sha256, new_sha256, created_at), `mask_files`. Write rate is tiny; SQLAlchemy/Postgres rejected as premature. All artifact references store paths, not blobs; the asset endpoint serves files through opaque tokens after validating the path is under a configured root (path traversal is the principal attack surface — accept writes into the test-data tree).

### D5 — Review semantics: per-page accept for images, document-level for PDFs

Accept(page) copies the candidate source file over the reference when the comparison is single-image; for multi-page PDFs the file-level operation is document-granular, so the UI offers accept-at-document and explains why, suggesting a mask for intended-partial changes. Every promotion records SHA-256 before/after in `decisions`. Reject stores a reason and offers a bug bundle (ZIP: ref + cand + diffs + sidecar + decision metadata — generalizes `generate_bug_data` from the RoboCon prototype). Re-ingesting a newer run resets affected pages to `unresolved`, history retained. As part of the core PR, `reference_run` is actually implemented (save candidate as reference when set) so dashboard accept and `REFERENCE_RUN` produce byte-identical layouts.

### D6 — Embedded comparison engine: import the library, run in a worker pool

The server imports `DocTest` directly (same venv) and exposes:
- `POST /api/mask-preview` — `DocumentRepresentation(file, ignore_area=...)` for the displayed page; returns resolved `pixel_ignore_areas` + page DPI. Debounced client-side, cached server-side per (file hash, page, engine, mask).
- `POST /api/recompare` — re-runs `compare_images` on a stored comparison's reference/candidate with user-adjusted masks, returning fresh per-page results/diff images to a scratch dir (never touching run artifacts). This enables "show past comparisons with adjusted masks in real time": tune a mask, immediately see which historical failures it would have suppressed, then save.

Comparisons are CPU-bound (OpenCV/OCR), so they run in a `ProcessPoolExecutor` with a small job queue and per-job timeout; results stream back over the existing REST polling (WebSocket deferred with the listener work). OCR-dependent previews require environment parity with the test environment; the existing `DocTest.CapabilityCheck` registry (verified present) runs at server startup and the UI surfaces missing capabilities (e.g., no tesseract → pattern preview disabled with explanation, coordinate/area preview unaffected).

### D7 — Mask editor: react-konva, schema-exact I/O, DPI sourced from sidecar

Locked image layer, editable `Rect` layer with `Transformer`, toggleable diff-region overlay. Coordinates masks: numeric fields synced with handles; unit selector converts for *display only* — the file stores the user's unit. The active DPI is displayed and always sourced from the sidecar (default 200). The core truncation fix (convert first, round last in `_convert_to_pixels`) lands before the editor ships, since round-trip fidelity depends on it. Area masks: side-panel location + percent slider with translucent band preview. Pattern masks: regex input with live preview via `/api/mask-preview`. Import accepts file JSON, inline list, and shorthand string (normalized through the same rules as `_parse_mask_string`); export always writes pretty-printed JSON with stable key order. Editor writes are atomic (temp file + rename) with a `.bak` of the previous version. "Add ignore mask" on a hovered diff region seeds a `coordinates` mask with configurable padding.

### D8 — Frontend stack and packaging

Vite + React + TypeScript, react-konva for the editor canvas. The wheel ships the built static frontend; end users `pipx install doctest-dashboard && doctest-dashboard serve` — Node is a dev-only dependency. Server binds 127.0.0.1 by default; `--host`/`--token` flags for team mode.

### D9 — Test strategy (full journeys, no mocked engine in e2e)

- **Core (poetry/pytest in `utest/`, robot suites in `atest/`)**: sidecar schema emission for image/PDF/pass/fail/masked comparisons; `reference_run` behavior; `_convert_to_pixels` precision regression.
- **Backend (uv/pytest in `dashboard/tests/`)**: ingestion against *generated* fixtures — the test suite executes real robot runs from `atest/` data into a temp dir and ingests the genuine `output.xml` (verified this takes seconds); accept/reject file mutations + audit rows; path-traversal rejections; mask round-trip property tests (parse → save → parse, byte-stable); recompare correctness (mask that covers the date region flips FAIL→PASS on the `birthday_1080` pair).
- **End-to-end (Playwright in `dashboard/e2e/`)**: full user journeys against `doctest-dashboard serve` with real ingested runs: (J1) ingest → browse grid → open diff viewer → switch modes → accept page → assert reference file content changed on disk + audit row; (J2) reject with reason → download bug ZIP → assert contents; (J3) open mask editor from diff region → adjust → live preview boxes appear → recompare shows PASS → save masks.json → assert file matches library schema and a robot re-run with that mask passes; (J4) shorthand-import → edit → export round trip.

## Risks / Trade-offs

- [DPI/unit drift between editor preview and execution] → DPI always read from sidecar and displayed; truncation fix in core; round-trip contract tests at multiple DPIs.
- [Recompare/preview latency on OCR paths] → process pool + per-(file,page,engine) caching + client debounce; UI shows engine + timing.
- [Server writes into user test-data trees (accept, masks.json)] → all writes confined to configured roots, atomic with backups, SHA-256 audit; refuse symlinks escaping roots.
- [Scraping fallback yields degraded records] → explicit "limited data — enable result_json" state in the UI rather than silently worse behavior.
- [Sidecar schema churn] → `schema_version` field from day one; dashboard supports v1 only and rejects unknown majors with a clear error.
- [Two package managers in one repo (poetry core, uv dashboard)] → CI matrix documents both; dashboard CI job uses `uv sync`; contributor docs in `dashboard/README.md`.
- [PDF page-level accept impossible at file level] → honest UI affordance offering document-accept or mask creation; never silently document-promote on a page action.

## Migration Plan

1. Core PR: sidecar + `reference_run` implementation + truncation fix (all opt-in/bugfix, backward compatible). Releasable independently.
2. Dashboard package lands in `dashboard/` and is published separately; consumes sidecars from any core ≥ that release, degrades to scraping for older outputs.
3. Rollback: sidecar is opt-in (`result_json` default False), dashboard is a standalone package — removing it touches nothing in core.

## Open Questions

- PyPI name: `doctest-dashboard` vs `robotframework-doctest-dashboard` (collision risk with stdlib-doctest tooling suggests the longer name).
- Should `result_json` default to on once stable (cheap, big ecosystem benefit)? Proposed: flip in the release after the dashboard ships.
- Actor identity for the audit trail in local mode (OS user?) vs token-derived name in team mode.
