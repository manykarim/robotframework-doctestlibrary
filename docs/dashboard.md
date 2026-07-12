# doctest-dashboard — Visual Review Dashboard & Mask Editor

The **doctest-dashboard** is a companion web application for robotframework-doctestlibrary. It lets you:

- review visual comparison failures in a purpose-built UI (instead of `log.html`)
- **accept** differences by promoting the candidate to the new reference — with a full audit trail
- **reject** differences and export a ready-to-attach bug-data bundle
- create and edit **masks/ignore areas visually**, with live preview and instant re-comparison of past runs against the adjusted masks

It runs locally with zero infrastructure: one process, one SQLite file, your baselines stay plain files in your repository (accepting a change produces a normal git diff).

---

## 1. Installation

The dashboard ships as the `[dashboard]` extra of `robotframework-doctestlibrary` — release wheels include the prebuilt web UI, so no Node.js is needed:

```bash
pip install robotframework-doctestlibrary[dashboard]
# or everything (LLM keywords + dashboard):
pip install robotframework-doctestlibrary[all]
```

### Development setup

The repository is managed with [uv](https://docs.astral.sh/uv/); the dashboard backend lives in `doctest_dashboard/`, its tests in `utest/dashboard/`, browser journeys in `e2e/`, and the web UI in `frontend/`:

```bash
uv sync --all-extras                           # backend + library + test tooling
cd frontend && npm install && npm run build    # build the web UI once
uv run doctest-dashboard serve                 # serves the built UI
uv run pytest utest/dashboard                  # backend tests
uv run playwright install chromium
uv run pytest e2e --browser chromium           # end-to-end journeys (real robot runs, no mocks)
```

During frontend development, `npm run dev` starts a Vite dev server that proxies `/api` to `127.0.0.1:8008`.

## 2. Prerequisites in your test suites

The dashboard works best when the library emits its machine-readable **result sidecar**. Enable it (plus file-mode PNG screenshots) in suites you want to review:

```robotframework
*** Settings ***
Library    DocTest.VisualTest    result_json=true    take_screenshots=true    screenshot_format=png
Library    DocTest.PdfTest      result_json=true
```

This writes one JSON file per comparison into `{OUTPUT_DIR}/doctest_results/` containing statuses, SSIM scores, diff regions, resolved masks, and separate per-page reference/candidate/diff renderings.

Runs **without** `result_json` can still be ingested: the dashboard falls back to scraping screenshot references from the log and shows those comparisons as **degraded** — combined images only, no per-page review, no accept, no diff-region navigation. The UI tells you exactly what is missing and how to enable it.

> `embed_screenshots=true` runs cannot be reviewed — use file mode (the default).

## 3. Starting the dashboard

```bash
doctest-dashboard serve
```

- binds **127.0.0.1:8008** by default — open <http://127.0.0.1:8008>
- `--port 9000` to change the port
- `--root /path/to/testdata` — allowlist a directory the server may read images from and write baselines/masks into (repeatable). Directories of ingested `output.xml` files are allowed automatically.
- `--data-dir` — where the SQLite database lives (default `./.doctest_dashboard/`)

**Team mode** (shared instance):

```bash
doctest-dashboard serve --host 0.0.0.0 --token <secret>
```

All API calls then require `Authorization: Bearer <secret>`. Put the instance behind your existing tunnel/SSO (e.g. Cloudflare Access) for anything beyond a trusted network. Note the server runs with *your* file permissions and writes accepted baselines into your test-data tree — every write is confined to the configured roots.

## 4. Ingesting results

After a Robot Framework run, three equivalent ways:

```bash
doctest-dashboard ingest results/output.xml
```

- **UI, by path**: paste the `output.xml` path into the field on the start page and click **Ingest output.xml** (the file must be readable by the server).
- **UI, from disk**: click **Upload results folder…** and pick your Robot Framework output directory in the browser's folder dialog. The whole tree (`output.xml`, `screenshots/`, `doctest_results/` sidecars and renderings) uploads with its structure intact into the dashboard workspace and is ingested immediately — no path or `--root` configuration needed. Irrelevant file types are filtered out before upload; 500 MB limit per upload.

Re-ingesting the same `output.xml` path updates the run instead of duplicating it. Pabot-merged outputs work — each comparison carries its own sidecar.

> Note on uploaded runs: reference/candidate *source documents* usually live outside the output folder, so for uploads from another machine those paths may not resolve — reviewing diffs works fully (the images travel with the upload), while accept/recompare need the source files to exist on the server.

The ingester reads comparison status at **keyword level**, so failures wrapped in `Run Keyword And Expect Error` are still recorded as failing comparisons.

The UI follows your OS light/dark preference; the 🌙/☀ toggle in the top bar overrides it (persisted per browser). A fresh dashboard shows an onboarding card with these ingestion options until the first run arrives.

## 5. Reviewing comparisons

Housekeeping: runs can be **deleted** from the run list (dashboard records and asset registrations only — files on disk are never touched); engine scratch results older than 7 days and uploads older than 30 days are swept at server startup (configurable); runs are never auto-deleted.

1. **Runs** page → click a run → test grid with diff thumbnails (runs produced by the current library render highlighted candidates — differences outlined on the real page — instead of raw diff images). Runs are named `<suite> · <output folder>`; click ✏ to give a run your own label, and import times display relatively (exact stamp in the tooltip).
2. Click a comparison to open the **diff viewer** (the header names the test — or the keyword's `name=` label — with the keyword as a chip):

   | Key | Mode |
   |-----|------|
   | `1` | side-by-side |
   | `2` | overlay (default, with opacity slider) |
   | `3` | blink |
   | `4` | swipe |
   | `5` | highlight — the library's rendering with all differences outlined (sidecars from current library versions) |
   | `n` / `p` | next / previous diff region |

   **Root cause**: with a diff region selected, *Explain region (text)* extracts and compares the text inside the region (reference vs candidate, via the library's own engine at the comparison's DPI) — showing *what* changed, not just where. Failing `Compare Pdf Documents` comparisons additionally render their differences as per-facet sections (text, metadata, structure, …).

   Every mode supports **zoom** (mouse wheel, 0.25×–8×) and **pan** (drag), synchronized across side-by-side panes. Pages auto-fit the pane when opened; **Fit** / **100%** buttons and double-click (refit) switch between fitted and actual size, and jumping to a diff region centers it at the current zoom. All detected diff regions are outlined by default (toggleable via the *diff regions* checkbox); the selected region gets the solid highlight.

3. Decide:
   - **Accept page / Accept document** — copies the candidate file over the reference (identical layout to a `REFERENCE_RUN`), records actor, reason, and SHA-256 before/after in the audit table, and marks the comparison `accepted`. For **multi-page PDFs**, page-level accept is impossible at file level — the UI offers document-level accept or mask creation instead, and never silently writes partial files.
   - **Reject** — stores the reason, marks the comparison `rejected`, and **Download bug data** gives you a ZIP with reference, candidate, all failing diff images, the sidecar JSON, and decision metadata — ready to attach to a ticket.

   **Review by similarity**: the *By similarity* view groups unresolved failures whose differences are identical (same diff-region geometry and pixel content — one layout change hitting 50 invoices becomes one card). *Accept group (n)* promotes every member with one confirmed action; strict matching keeps batch accepts safe, and unique failures stay listed individually.

   **Batch review**: the grid offers multi-select checkboxes with *Accept selected (n)* and an *Accept all unresolved in run* action — both behind a confirmation stating how many baseline files will be written. Comparisons that cannot be promoted (degraded records, files outside the configured roots) are skipped with a reason instead of aborting the batch; every promotion keeps its own audit row.

When a newer run of the same test is ingested, pages whose images changed return to `unresolved`; unchanged pages keep their accepted/rejected state, and all past decisions stay queryable.

## 6. CI gating

`doctest-dashboard gate <run-id|latest>` turns review state into an exit code — 0 when the run has no unresolved failed comparisons, 1 with a listing of what still needs review, 2 for unknown runs. It reads the dashboard database directly, so it works on a **base install** (no `[dashboard]` extra needed on the CI runner):

```yaml
- run: doctest-dashboard ingest results/output.xml
- run: doctest-dashboard gate latest    # fails the job until changes are reviewed
```

Accepting or rejecting in the dashboard immediately turns a red gate green — review, then re-run the gate.

**History & flakiness**: the comparison view shows the same test's status/review timeline across all ingested runs, and the runs page lists *flaky comparisons* — identities whose status flipped across recent runs (informational, not gating).

## 7. Mask editor

Open it via **Mask Editor** in the top bar, or — the fastest path — step to a detected diff region in the viewer (`n`/`p` or the *next diff* button) and click **Add ignore mask**: the editor opens pre-seeded with a coordinate mask covering that region (plus padding).

The editor reads and writes the **exact mask schema** the library consumes (`IgnoreAreaManager`) — nothing proprietary:

- **Coordinates masks** — draw by dragging on the page, move/resize with handles, or type exact values. The unit selector (`px`/`mm`/`cm`/`pt`) converts for display only using the **rendering DPI shown in the banner** (sourced from the comparison sidecar); the file stores whatever unit you chose.
- **Area masks** — pick location (top/bottom/left/right) and percentage in the side panel; a translucent band previews the covered area live.
- **Pattern masks** (`pattern`, `line_pattern`, `word_pattern`) — type a regex and watch the matched text regions highlight on the page within half a second. The preview runs the library's *own* text-extraction path (PDF text or OCR), so what you see is exactly what a test run will mask. If Tesseract is not installed, pattern preview is disabled with an explanation — other mask types keep working.

  Matching levels: `word_pattern` matches single words; `line_pattern` matches whole text lines (anchored — wrap with `.*…​.*` to match anywhere); `pattern` matches single words, and regexes containing whitespace are searched anywhere within each line, masking **exactly the words the match span covers** — `Robot Framework` masks just the phrase, `.*Robot Framework.*` masks the whole containing line. Matching is case-sensitive; add `(?i)` for case-insensitive.

**Workflow:**

1. Get a document onto the canvas:
   - **Upload…** — pick any image or PDF from your machine; it is stored in the dashboard's workspace folder (`{data-dir}/uploads/`, always browsable) and opened immediately. A `masks.json` target next to the upload is suggested automatically. This works with zero configuration.
   - **Browse…** — a file picker that navigates the server's configured roots (ingested run folders are browsable automatically). Selecting an existing `masks.json` loads it immediately; for a new file, navigate to the folder and type a name.
   - Paths can still be typed or pasted directly.
2. **Load** an existing file, or **Import** masks as JSON or shorthand (`top:10;bottom:5`).
3. Edit. If you arrived from a comparison, click **Recompare with these masks** — the server re-runs the *actual* comparison engine on the stored reference/candidate with your draft masks and shows the would-be result (PASS/FAIL) without touching the original run or your baselines. Tune until it passes.
4. **Save** — pretty-printed JSON with stable key order (clean git diffs), written atomically with a `.bak` of the previous version.

> The old tkinter tool `utilities/mask_editor.py` is deprecated in favor of this editor.

## 8. Command and API reference (abridged)

```bash
doctest-dashboard serve [--host] [--port] [--token] [--root DIR ...] [--data-dir DIR]
doctest-dashboard ingest <output.xml>
```

| Endpoint | Purpose |
|---|---|
| `POST /api/ingest` | ingest an output.xml |
| `GET /api/runs`, `GET /api/runs/{id}/tests` | browse |
| `PATCH /api/runs/{id}` | set/clear a user label for a run |
| `GET /api/comparisons/{id}` | full detail incl. pages and sidecar |
| `POST /api/pages/{id}/accept`, `POST /api/comparisons/{id}/accept` | baseline promotion |
| `POST /api/comparisons/{id}/reject`, `GET .../bugdata` | reject + bug bundle |
| `GET/PUT /api/masks` | masks.json round-trip |
| `POST /api/mask-preview` | resolve masks to pixel boxes (live preview) |
| `POST /api/recompare`, `POST /api/recompare-batch` | re-run comparisons with adjusted masks |
| `GET /api/capabilities` | OCR/engine availability |
| `GET /api/browse` | root-confined directory listing (file picker) |
| `GET /api/runs/{id}/groups` | similarity groups of unresolved failures |
| `POST /api/comparisons/{id}/region-text` | explain a diff region (reference vs candidate text) |
| `GET /api/comparisons/{id}/history` | cross-run timeline for the comparison's identity |
| `GET /api/flaky` | identities with status flips across recent runs |
| `doctest-dashboard gate <run\|latest>` | CI exit-code gate on review state |
| `POST /api/upload` | store a local file in the dashboard workspace (images/PDF/JSON, 100 MB limit) |
| `POST /api/upload-results` | upload a whole results folder (relative paths preserved) and ingest its output.xml |

## 9. Troubleshooting

- **Yellow banner "server is older than this user interface" / 405 or "Not Found" on new buttons** — the UI files are re-read from disk on every request, but the Python server process keeps running the code it started with. After updating the dashboard, restart `doctest-dashboard serve`. The UI detects this skew at load time and tells you.

- **Comparison shows "degraded"** — the run was executed without `result_json=true`. Re-run with the sidecar enabled.
- **403 on accept/masks/preview** — the file lies outside the configured roots. Start the server with `--root` covering your test-data directory.
- **File picker shows no useful locations** — use **Upload…** to bring a file from your machine (no configuration needed), start the server with `--root /your/testdata`, or ingest a run first (run folders become browsable automatically).
- **Pattern preview disabled** — Tesseract is not installed in the dashboard's environment. Install it (and keep the OCR engine consistent with your test environment) — check `GET /api/capabilities`.
- **Recompare is slow on OCR-heavy pages** — first run is computed (process pool, 120 s timeout); identical requests are served from cache.
