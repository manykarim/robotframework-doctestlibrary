# Visual Review Dashboard for robotframework-doctestlibrary

## Solution Proposal & Research Report

**Date:** 2026-06-10
**Scope:** A companion web dashboard for [robotframework-doctestlibrary](https://github.com/manykarim/robotframework-doctestlibrary) that lets users review visual differences, accept or deny them (Applitools-style baseline management), and create/edit masks in a visual editor — plus the research needed to decide between a post-execution adapter and a Robot Framework Listener.

---

## 1. Executive Summary

robotframework-doctestlibrary already produces everything a review workflow needs — diff screenshots, reference/candidate renderings, OCR text, and a `reference_run` mode — but the review experience today lives inside `log.html`, and the only triage tooling is a PyQt5 desktop prototype (`TestResultEvaluator.py` from the RoboCon 2022 DE demo) and a minimal tkinter `utilities/mask_editor.py`. Both prove the concept; neither scales to team use, CI integration, or a pleasant reviewing experience.

The recommendation in one paragraph: build **`doctest-dashboard`**, a self-contained, locally-runnable web application (FastAPI backend + React frontend, SQLite metadata store, file-system baseline store) shipped as an optional extra of the library or as a sibling package. It ingests results through **two complementary channels**: a **post-execution ingester** that parses `output.xml` with `robot.api.ExecutionResult` (zero changes to test execution, works retroactively, works with pabot-merged outputs) and an **optional Listener v3** for live streaming during long runs. The critical enabler is a small change inside the library itself: emit a **machine-readable `comparison_result.json` sidecar per comparison**, so the dashboard never has to scrape HTML log messages or guess screenshot semantics from filename suffixes. The dashboard provides an Applitools-class diff viewer (side-by-side, overlay, blink, swipe, diff-region navigation), one-click accept (promote candidate → baseline) and reject (collect bug data, annotate), full audit history, and an integrated **mask editor** that reads and writes the exact JSON schema `IgnoreAreaManager` consumes today — including live preview of coordinate, area, and text-pattern masks, and a "create mask from this diff region" shortcut that closes the loop between reviewing and maintaining tests.

Estimated effort for a usable v1 (ingest + review + accept/reject + coordinate-mask editor): roughly 6–9 person-weeks. The pattern-mask live preview and listener live mode are v1.5/v2 items.

---

## 2. Goals and Non-Goals

**Goals.** Review visual comparison failures outside `log.html` with a purpose-built UI; accept differences by promoting candidates to new baselines with an audit trail; deny differences by packaging reproducible bug data; create and maintain masks visually with immediate preview; support both single-run review and run-over-run history; work locally with zero infrastructure, and optionally self-hosted for a team (Hetzner/Coolify-friendly: single container, single volume).

**Non-Goals (v1).** No cloud service, no SaaS billing, no AI auto-triage (though the library's existing LLM keywords are a natural v2 integration), no replacement of `log.html` (the dashboard complements it and deep-links into it), no attempt to manage the test code itself.

---

## 3. Prior Art: What "Applitools-like" Actually Means

To define the feature bar, it is worth being precise about what established tools do well, because the differentiating value is rarely the diff algorithm — doctestlibrary already has SSIM, move tolerance, OCR-based text checks, and watermark subtraction. The value is in the *review workflow*.

**Applitools Eyes** popularized the batch/test/step hierarchy, a baseline per (test, environment) tuple, accept/reject at step granularity with "accept all similar" batching, annotation regions that become ignore regions for future runs, and a full audit trail of who accepted what and when. Its single most-copied UX idea is that *creating an ignore region is a one-click action from the diff view*, not a separate tool.

**Percy (BrowserStack)** contributed the "build" review model borrowed from code review: a run is reviewed as a unit, reviewers approve or request changes, and approval state gates CI. Its diff UI defaults to a red-highlight overlay with a click-to-toggle original/candidate, which users consistently find faster than side-by-side for spotting small diffs.

**Open-source tools** (BackstopJS, reg-suit, Lost Pixel, Argos, Vizregress-style viewers) converge on the same minimum set: a grid of thumbnails filtered by status, a detail view with side-by-side / overlay / onion-skin / swipe modes, approve buttons that copy candidate files over reference files, and a JSON report format the viewer consumes. reg-suit's insight worth stealing: baselines live in plain directories so they remain git-versionable — exactly how doctestlibrary users already manage reference PDFs/images.

The synthesis for this project: **per-page (not just per-test) accept granularity** — because document tests are multi-page and page 3 may be an intended change while page 7 is a bug — plus **mask creation from the diff view**, plus **plain-file baselines** that stay compatible with existing repos.

---

## 4. Lessons from `TestResultEvaluator.py` (RoboCon 2022 DE)

The PyQt5 prototype ([source](https://github.com/manykarim/rbcn22de-visual-document-testing/blob/main/utilities/TestResultEvaluator.py)) is the conceptual seed of this proposal and validates the core workflow loop. What it gets right, and should be preserved:

**Parsing `output.xml` with `robot.api.ExecutionResult` and a `ResultVisitor`.** The evaluator opens an `output.xml`, runs a `TestResultParser` visitor over it, and extracts per-test records of suite, test name, error message, reference path, candidate path, and screenshot list. This proves the post-execution channel works and requires no test changes. The dashboard's ingester is a direct, hardened descendant of this approach.

**Accept and decline as first-class row actions.** The context menu offers "Accept Changes" / "Decline Changes" per selected rows, with a checkbox column tracking selection state. The dashboard generalizes this to per-page decisions with persisted state.

**Reference promotion via `candidate_` filename convention.** `generate_reference_data()` copies each accepted candidate into a reference folder, deriving the target filename by stripping the `candidate_` prefix. This convention is fragile (it warns when filenames don't contain `candidate_`) and is one of the things the structured sidecar JSON (Section 6) eliminates: the comparison result should *state* the reference path explicitly rather than have tooling reverse-engineer it.

**Bug data collection.** `generate_bug_data()` copies the entire directory of a declined test into a per-test-case bug folder — a genuinely useful "package this failure for a ticket" feature that few commercial tools offer. Keep it, and add the diff images and decision metadata to the package.

What the prototype lacks, and why a web app: no diff overlay modes (it stacks raw screenshots in a `QGraphicsView`); no persistence of decisions (state dies with the window); no multi-user/remote access; no run history; no mask awareness at all; desktop-only distribution with PyQt5 as a heavy dependency. Every one of these is structural, not incremental — which is the argument for a rebuild rather than an extension.

---

## 5. Current State of the Library: What the Dashboard Can Rely On Today

Findings from the `main` branch source (v0.33.x line), because the dashboard's data contract must be grounded in what the library actually emits.

**Screenshots.** `VisualTest.add_screenshot_to_log()` has two modes. With `embed_screenshots=True` it base64-inlines images into the log via `robot.api.logger` — invisible to any external tool except by parsing HTML out of `output.xml` messages. In file mode (default) it writes `{uuid1}{suffix}.{jpg|png}` into `{OUTPUT_DIR}/screenshots/` (configurable via `set_screenshot_dir`), prefixing `{PABOTQUEUEINDEX}-` under pabot, and logs an `<img>` HTML message with the relative path. Consequences: (1) the dashboard must require or strongly recommend file mode; (2) today, the *only* machine link between a test and its images is the HTML `<img src=...>` inside `output.xml` messages — parseable, but brittle; (3) the `suffix` (e.g. `_diff`, reference/candidate page markers) is the only semantic tag on an image. JPEG quality is hardcoded to 70, which visibly degrades diff inspection — the dashboard should recommend `screenshot_format=png` for review runs, or the sidecar should store lossless copies of diff images.

**Reference-run mode.** The library reads `${REFERENCE_RUN}` (and exposes `set_reference_run`); when true, candidates are saved as new references instead of compared. This is the existing "approve everything" mechanism — the dashboard's accept action is its surgical, per-page counterpart, and the two must write identical file layouts.

**Masks.** `IgnoreAreaManager` accepts a JSON file (`placeholder_file`), an inline dict/list/JSON string, or the shorthand `top:10;bottom:10` string. `DocumentRepresentation` implements five mask types — `coordinates` (x/y/width/height with `unit` of `px`, `mm`, `cm`, or `pt`, converted using the rendering DPI), `area` (top/bottom/left/right + percent), and the text-driven `pattern`, `line_pattern`, `word_pattern` (regex matched against OCR or PDF text, masking matched line/word bounding boxes). Each entry carries `page` (`"all"`, int, or numeric string) and an optional `name`. This is the schema the mask editor must read and write verbatim — no new format.

**Result semantics in the log.** Failures raise with a summary (a single top-level warning like "Comparison failed: N difference(s) found") while individual differences, SSIM scores, move-tolerance results, OCR text comparisons, and LLM verdicts are logged as INFO messages inside the keyword. Rich, but free-text — reinforcing the case for a structured sidecar.

**Existing mask editor.** `utilities/mask_editor.py` is a ~170-line tkinter tool: load an image, drag rectangles, right-click delete, export/load JSON. Pixel-unit coordinate masks only, no page concept, no area/pattern types, no preview of mask effect. It should be deprecated in favor of the dashboard's editor once that ships, with a pointer in the README.

---

## 6. The Key Architectural Decision: How Results Reach the Dashboard

Three channels were investigated. The recommendation is to implement the structured sidecar plus post-execution ingestion first, and the listener as an optional live mode — but the sidecar is the linchpin.

### 6.1 Channel A — Post-execution ingestion of `output.xml` (must have)

A CLI (`doctest-dashboard ingest output.xml` or auto-watch of a results directory) parses the result model:

```python
from robot.api import ExecutionResult, ResultVisitor

class ComparisonCollector(ResultVisitor):
    def visit_test(self, test):
        # walk test.body for keywords from DocTest.VisualTest / DocTest.PdfTest,
        # collect status, messages, <img> paths, and (preferred) the
        # comparison_result.json sidecar reference logged by the library
        ...

result = ExecutionResult("output.xml")
result.visit(ComparisonCollector(db))
```

Strengths: zero impact on execution, works on historical outputs, works on `rebot`-merged pabot results, works in CI where the dashboard isn't running. Weakness: without help from the library, semantics must be scraped from HTML messages (which the evaluator prototype already had to do). This mirrors the conclusion from the Allure-adapter analysis: post-execution adapters are robust exactly to the degree the producer emits structured data.

### 6.2 Channel B — Robot Framework Listener (optional, for live review)

Research summary of the listener interface (RF User Guide §4.3, current RF 7.x):

A listener is a class or module exposing hook methods; the API version is declared with `ROBOT_LISTENER_API_VERSION = 2` or `3`. **Listener v3 is the right choice** for this project: since RF 7.0 it is feature-complete (including keyword-level `start_keyword`/`end_keyword` and `start_library_keyword` etc.), and unlike v2's string/dict arguments it receives the *actual running and result model objects*, so `result.status`, `result.message`, timing, and tags are typed objects, and listeners may even modify results. Relevant hooks for the dashboard:

```python
from robot import result, running

class DashboardListener:
    ROBOT_LISTENER_API_VERSION = 3

    def __init__(self, url="http://127.0.0.1:8008", run_id=None):
        ...  # listeners accept init args: --listener Dashboard.py:arg1:arg2
             # or with named syntax --listener Dashboard.py;url=...;run_id=...

    def start_suite(self, data: running.TestSuite, result_: result.TestSuite): ...
    def end_test(self, data: running.TestCase, result_: result.TestCase):
        # POST test status + collected comparison sidecars to the dashboard API
        ...
    def log_message(self, message: result.Message):
        # capture INFO/WARN messages, including the <img> html messages,
        # and sidecar-path announcements, as they happen
        ...
    def output_file(self, path):   # final output.xml path → trigger full ingest
        ...
    def close(self): ...
```

Listeners are activated per run (`robot --listener DashboardListener.py tests/`) or registered *by the library itself* as a **library listener** (`self.ROBOT_LIBRARY_LISTENER = self` with `ROBOT_LIBRARY_SCOPE`), which is the elegant deployment: `Library DocTest.VisualTest dashboard_url=...` could self-report without any CLI flags. Listener calling order can be controlled with `ROBOT_LISTENER_PRIORITY` if interaction with other listeners ever matters. Two operational cautions: listener exceptions are reported but must never break test execution, so all network I/O in the listener must be fire-and-forget with short timeouts and local spooling when the dashboard is unreachable; and under pabot, N parallel processes each run their own listener instance, so events must carry the pabot index (the library already exposes `PABOTQUEUEINDEX`) and the dashboard must merge.

Verdict: the listener is *not required* for the core review workflow — `end_test` granularity adds "watch failures arrive live during a 2-hour regression run," which is valuable but secondary. Build it after ingestion works, as ~150 lines reusing the same API.

### 6.3 Channel C — Structured sidecar emitted by the library (small library change, biggest payoff)

Add to `VisualTest`/`PdfTest` an opt-in `result_json=${True}` (or always-on when cheap): for every comparison, write `{OUTPUT_DIR}/doctest_results/{uuid}.json` and log one parseable message `DOCTEST_RESULT: <relpath>`. Proposed schema:

```json
{
  "schema_version": 1,
  "keyword": "Compare Images",
  "library": "DocTest.VisualTest",
  "status": "FAIL",
  "reference": {"path": "testdata/Reference.pdf", "pages": 3, "dpi": 200},
  "candidate": {"path": "out/Candidate.pdf", "pages": 3, "dpi": 200},
  "settings": {"move_tolerance": 20, "check_text_content": false,
               "watermark_file": null, "screenshot_format": "png"},
  "masks": {"source": "masks.json", "resolved": [ /* IgnoreAreaManager output */ ]},
  "pages": [
    {"page": 1, "status": "PASS", "ssim": 0.9991,
     "images": {"reference": "screenshots/ab12_ref_p1.png",
                "candidate": "screenshots/ab12_cand_p1.png"}},
    {"page": 2, "status": "FAIL", "ssim": 0.9412,
     "diff_regions": [{"x": 410, "y": 36, "width": 150, "height": 42}],
     "images": {"reference": "...", "candidate": "...",
                "diff": "screenshots/ab12_diff_p2.png",
                "thresholded": "screenshots/ab12_thresh_p2.png"}}
  ],
  "llm": {"verdict": null},
  "timing": {"started": "2026-06-10T09:14:02", "elapsed_ms": 1840}
}
```

Two fields do most of the heavy lifting downstream. Explicit `reference.path` makes accept-promotion exact (no `candidate_` filename parsing). `diff_regions` — the bounding boxes the library already computes internally for contour/diff detection — enable diff-region navigation ("next difference" button), accept-region statistics, and crucially the **one-click "mask this region"** feature, since a diff region converts directly into a `coordinates` mask entry. This change is small (the data already exists in memory at comparison time), fully backward compatible, and benefits *any* future tooling, not just this dashboard — the same argument that favored the post-execution Allure adapter.

---

## 7. Recommended Architecture

```
┌────────────────────────────── doctest-dashboard ──────────────────────────────┐
│                                                                                │
│  React SPA (Vite, TypeScript)                                                  │
│   ├─ Run & test grid (status filters, thumbnails)                              │
│   ├─ Diff viewer (side-by-side / overlay / blink / swipe / diff-only)          │
│   ├─ Review actions (accept / reject / annotate, per page & batch)             │
│   └─ Mask editor (Konva canvas, schema-exact masks.json I/O)                   │
│                              │ REST + WebSocket                                │
│  FastAPI backend                                                               │
│   ├─ Ingest service: output.xml (ResultVisitor) + sidecar JSON                 │
│   ├─ Review service: decisions, baseline promotion, bug-data export            │
│   ├─ Mask service: load/save masks.json, server-side pattern preview           │
│   ├─ Asset service: serves screenshots & rendered pages (range/cache headers)  │
│   └─ Live API: receives Listener v3 events (optional)                          │
│                                                                                │
│  Storage                                                                       │
│   ├─ SQLite (runs, tests, pages, decisions, audit)  — single file              │
│   └─ Filesystem: baselines stay where they are (git-versionable);              │
│      run artifacts referenced in place under each OUTPUT_DIR                   │
└────────────────────────────────────────────────────────────────────────────────┘
```

Rationale for the stack: FastAPI + SQLite + React matches the existing ecosystem conventions in your other projects (rf-mcp tooling, the Polarion MCP recommendation) and keeps deployment to `pipx install doctest-dashboard && doctest-dashboard serve` locally, or one small container behind a Cloudflare tunnel for a team. SQLite is sufficient because the write rate is tiny (decisions and ingests) and concurrency is low; Postgres would be premature. Baselines deliberately remain **plain files in the repository** so accepting a change produces a reviewable git diff — the reg-suit lesson, and consistent with how doctestlibrary users already work.

Packaging options, in order of preference: (1) a separate package `robotframework-doctestlibrary-dashboard` that depends on the core library (keeps the core dependency-light — it already fights heavy native deps); (2) an extra `pip install robotframework-doctestlibrary[dashboard]`. The frontend is built at release time and shipped as static files inside the wheel, so end users never need Node.

## 8. Feature Set and Applitools-Parity Matrix

| Capability | Applitools/Percy | Dashboard v1 | Notes |
|---|---|---|---|
| Run (batch) overview with pass/fail/unresolved counts | ✓ | ✓ | per `output.xml` ingest; pabot runs merge by run id |
| Test grid with diff thumbnails & status filters | ✓ | ✓ | thumbnail = first failing page's diff |
| Side-by-side viewer with synced zoom/pan | ✓ | ✓ | |
| Overlay / blink / onion-skin / swipe modes | ✓ | ✓ | overlay default; keyboard `1–5` to switch |
| Diff-region navigation (next/prev difference) | ✓ | ✓ (needs sidecar) | from `diff_regions` |
| Accept → baseline promotion | ✓ | ✓ | per page, per test, per run ("accept all in suite") |
| Reject with reason / bug-data export | partial | ✓ | zip of ref+cand+diff+metadata, à la `generate_bug_data` |
| Audit trail (who/when/what, before/after hashes) | ✓ | ✓ | SQLite `decisions` table; baseline file hash recorded |
| Ignore-region creation from diff view | ✓ | ✓ | writes `coordinates` mask into chosen masks.json |
| Mask editor (coordinate, area, pattern) with preview | region-only | ✓ superset | pattern masks are a doctest-unique strength |
| Baseline history & rollback | ✓ | v1.5 | store promoted-over files in `.doctest_baseline_history/` or rely on git |
| Live results during a run | ✓ | v2 (listener) | WebSocket push to the run view |
| AI auto-triage of diffs | ✓ | v2 | reuse `DocTest.Ai` LLM verdicts already in the library |
| CI status gating (fail until reviewed) | ✓ | v2 | exit-code tool: `doctest-dashboard gate <run-id>` |

**Review semantics.** A failed comparison enters state `unresolved`. *Accept (page)* copies the candidate rendering or source file over the reference. For single-image comparisons this is a file copy; for PDFs, page-level accept is not possible at the file level, so the UI offers accept-at-document granularity when the artifact is a PDF and explains why — with the documented alternative of masking the intended change. *Accept (test)* promotes the whole candidate document, exactly equivalent to a targeted `REFERENCE_RUN`. Every promotion records previous and new SHA-256 in the audit table. *Reject* requires an optional reason, marks the test `rejected`, and offers the bug-data bundle. A re-ingested newer run resets pages whose images changed to `unresolved` while keeping history.

## 9. Mask Editor — Design Investigation

The editor's contract is simple and strict: read and write the exact `IgnoreAreaManager` / `DocumentRepresentation` schema, nothing proprietary. Everything else is UX.

**Canvas technology.** Three candidates were compared. Plain `<canvas>` + custom hit-testing: maximal control, highest effort. **Fabric.js**: object model with built-in selection/resize handles, but heavyweight and its serialization model fights a schema-exact JSON target. **Konva.js (react-konva)**: declarative React bindings, `Transformer` gives move/resize/rotate handles for free, layers map naturally to "document page image below, mask shapes above, diff overlay optional," and it is the de-facto choice for annotation UIs. **Recommendation: react-konva.** The page image sits in a locked layer; each mask is a `Rect` (or full-width/height band for area masks) in an editable layer with a `Transformer`; a third toggleable layer can show the latest diff regions to guide mask placement.

**Editing model per mask type.**

*Coordinates masks:* draw by drag; move/resize with handles; numeric fields stay in sync for precision; a unit selector (`px`/`mm`/`cm`/`pt`) converts using the run's DPI — this is the subtle part. The library converts physical units to pixels with the rendering DPI (`_convert_to_pixels`), so the editor must know the DPI of the displayed rendering (from the sidecar, default 200 today) or the conversion silently drifts. The editor therefore displays the active DPI prominently and stores whichever unit the user selected, converting only for display.

*Area masks:* not drawn freehand — selected from a side panel (location + percent slider) and previewed as a translucent band across the page; `page: all` vs specific page is a toggle. Live preview makes the percent slider self-explanatory.

*Pattern / line_pattern / word_pattern masks:* a regex input with live match preview. The backend exposes `POST /api/mask-preview` which runs the same text-extraction path the library uses (PDF text via MuPDF or OCR via the configured engine) for the displayed page and returns the bounding boxes the pattern would mask; the editor highlights them. This gives users, for the first time, *visual confirmation that a date-pattern mask actually catches the date* before committing it — arguably the editor's strongest feature, and only possible because the backend can import the library directly. Implementation note: extraction can be slow on OCR paths, so preview results are cached per (file, page, engine) and the request is debounced.

**Create-mask-from-diff.** In the diff viewer, hovering a detected diff region (from sidecar `diff_regions`) shows "Add ignore mask"; clicking opens the editor pre-seeded with a `coordinates` mask of that box (plus configurable padding), targeting a masks.json chosen via a file picker rooted at the test's data directory. This single interaction is what makes mask maintenance routine instead of a chore.

**Multi-page & files.** A page strip on the left; masks listed with name, type, page, and visibility toggles; `page: "all"` masks render on every page. Import merges or replaces; export writes pretty-printed JSON with stable key order to keep git diffs clean. The editor also accepts the shorthand string format on import (`top:10;bottom:10`) by normalizing it through the same logic as `_parse_mask_string`, but always exports the JSON form.

## 10. Data Model and API Sketch

SQLite tables (abridged): `runs(id, name, output_xml_path, started, imported_at, rf_version, pabot)`, `tests(id, run_id, suite, name, status, keyword, message)`, `pages(id, test_id, page_no, status, ssim, ref_img, cand_img, diff_img, regions_json)`, `decisions(id, page_id|test_id, action accept|reject, actor, reason, prev_hash, new_hash, created_at)`, `mask_files(path, last_seen_hash)`.

REST surface (abridged):

```
POST /api/ingest                 {output_xml: path}            → run summary
GET  /api/runs                   ?status=unresolved
GET  /api/runs/{id}/tests        ?status=fail
GET  /api/tests/{id}             full detail incl. pages, sidecar
POST /api/pages/{id}/accept      {reason?}                     → promotes baseline
POST /api/tests/{id}/accept      {scope: test|document}
POST /api/tests/{id}/reject      {reason}                      → optional bug bundle
GET  /api/tests/{id}/bugdata     → zip download
GET  /api/assets/{token}         image serving (paths never exposed raw)
GET  /api/masks?file=...         normalized mask list
PUT  /api/masks                  {file, masks[]}               atomic write + backup
POST /api/mask-preview           {file, page, pattern, type}   → bounding boxes
WS   /api/live/{run_id}          listener event stream (v2)
```

Security posture: local-first, binds to 127.0.0.1 by default, no auth in local mode. Self-hosted mode adds a simple token or sits behind existing SSO/tunnel auth (Cloudflare Access fits your current infra). The accept endpoint writes into the test-data tree, so the server runs with the same user permissions as the test author and refuses paths outside configured roots — path traversal is the main thing to defend.

## 11. Roadmap

**M1 — Ingest & browse (≈2 wks).** CLI ingest of `output.xml` (HTML-message scraping fallback, evaluator-style), SQLite store, run/test grid, asset serving, basic side-by-side viewer.

**M2 — Sidecar in the library (≈1 wk, separate PR to doctestlibrary).** `result_json` option, `DOCTEST_RESULT:` log line, schema v1, docs. Ingester prefers sidecars when present.

**M3 — Review workflow (≈2 wks).** Overlay/blink/swipe modes, diff-region navigation, accept/reject with promotion + audit, bug-data export, keyboard-driven triage.

**M4 — Mask editor (≈2–3 wks).** react-konva editor, coordinate + area masks with unit/DPI handling, masks.json round-trip, create-mask-from-diff. Pattern preview endpoint lands at the end of M4 or as M4.5.

**M5 — Listener live mode & polish (≈1–2 wks).** Listener v3 with spooling, WebSocket run view, pabot merge handling, `gate` command for CI, deprecation note on the tkinter editor.

## 12. Risks and Open Questions

The DPI/unit conversion mismatch between editor preview and execution is the most likely source of subtle bugs; mitigated by always sourcing DPI from the sidecar and showing it in the UI. `embed_screenshots=True` runs cannot be reviewed without extracting base64 from messages — feasible but ugly; document file mode as a prerequisite instead. PDF page-level accept is semantically impossible at file level (Section 8) and the UI must communicate that honestly rather than pretend. Pattern-preview parity requires the backend environment to mirror the test environment's OCR setup (Tesseract availability, engine choice) — acceptable for a companion tool, but worth a capability check on startup (the library's `CapabilityCheck` can be reused). Finally, whether the sidecar lands in the core library or the dashboard ships a wrapper keyword is a governance call; the core PR is strongly preferred since the data already exists in memory and other tools (Allure adapter, rf-mcp reporting) would benefit from the same schema.

## 13. Sources

Repository and keyword docs: github.com/manykarim/robotframework-doctestlibrary (README, `DocTest/VisualTest.py`, `DocTest/IgnoreAreaManager.py`, `DocTest/DocumentRepresentation.py`, `utilities/mask_editor.py`, main branch, June 2026). PyQt5 prototype: github.com/manykarim/rbcn22de-visual-document-testing, `utilities/TestResultEvaluator.py`. Robot Framework User Guide 7.4.2, §3.4 Post-processing outputs and §4.3 Listener interface (robotframework.org). Feature-bar references: Applitools Eyes review workflow, Percy build review model, reg-suit/BackstopJS/Lost Pixel open-source viewers (vendor documentation, general knowledge).
