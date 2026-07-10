# doctest-dashboard — Analysis & Improvement Proposal

**Date:** 2026-06-15 · **Scope:** evaluation of the dashboard shipped on `feature/dashboard` (PR #139), grounded in code inspection and competitor research, with a prioritized improvement plan.

---

## 1. Executive Summary

The dashboard delivers a working end-to-end review loop — ingest → review → accept/reject → mask-edit → recompare — with an unusually strong verification story (90+ backend tests against real robot runs, 12 Playwright journeys, wheel-parity and version-skew gates). Its architectural bets are sound: sidecar as data contract, plain-file baselines, embedded comparison engine, root-confined file access.

Its weaknesses are the predictable ones of a fast v1: **list endpoints that won't survive large runs** (N+1 queries, no pagination), **no data lifecycle** (nothing is ever deleted — runs, scratch dirs, uploads, caches all grow forever), **missing table-stakes viewer ergonomics** (no zoom/pan, despite docs claiming it), and **no batch operations** — the single feature every competitor treats as the heart of the review workflow. Competitor research points to one high-leverage differentiator we're well-positioned for: **grouping similar diffs across documents** (Percy's "matching diffs", Applitools' "accept all similar"), which maps perfectly onto document testing where one layout change hits 50 invoices identically.

Recommended sequencing: fix hygiene first (P0, small and mechanical), then build the two differentiators that exploit data we already collect (diff-region grouping, text-in-region root cause), then the strategic tier (CI gating, live mode, annotations).

---

## 2. Current State

### 2.1 Architecture (as built)

```
                    Robot Framework run
                          │  result_json sidecars + PNGs
                          ▼
   ┌──────────────── doctest-dashboard ────────────────────┐
   │                                                       │
   │  React SPA (Vite/TS, hand-rolled hash router)         │
   │   ├─ RunList + folder upload      (App.tsx, 267 L)    │
   │   ├─ Diff viewer: side-by-side/overlay/blink/swipe    │
   │   │   + region nav + accept/reject (ComparisonView)   │
   │   └─ Mask editor: konva canvas, live pattern preview, │
   │       recompare, file browser     (MaskEditor, 755 L) │
   │                        │ REST (26 endpoints, app.py)  │
   │  FastAPI backend (single create_app closure, 550 L)   │
   │   ├─ ingest.py    ResultVisitor, sidecar-first        │
   │   ├─ review.py    accept/reject, SHA-256 audit, ZIP   │
   │   ├─ engine.py    ProcessPool(2), preview/recompare   │
   │   ├─ masks.py     IgnoreAreaManager-parity I/O        │
   │   └─ db.py        sqlite3 WAL, thin DAL               │
   │                                                       │
   │  SQLite: runs/tests/comparisons/pages/decisions/      │
   │          mask_files/assets     Files: stay in place   │
   └───────────────────────────────────────────────────────┘
```

### 2.2 Feature inventory vs. the original proposal (docs/doctest-dashboard-proposal.md)

| Capability | Status |
|---|---|
| Sidecar data contract (schema v1) | ✅ shipped, contract-tested |
| Ingest (path, API, browser folder upload) + degraded fallback | ✅ shipped (folder upload exceeds original scope) |
| Run/test grid, filters, thumbnails | ✅ shipped |
| Viewer modes side-by-side/overlay/blink/swipe + region nav | ✅ shipped |
| **Synced zoom/pan in viewer** | ❌ **not implemented — but claimed in spec & docs** |
| Accept per page/document, SHA-256 audit, PDF honesty | ✅ shipped |
| **Accept all in suite / batch accept** | ❌ missing (was in the original parity matrix) |
| Reject + bug-data ZIP | ✅ shipped |
| Decision carry-forward across runs (identity + content hash) | ✅ shipped — competitive with Percy's "approval carryforward" |
| Mask editor (coordinates/area/pattern, live preview, from-diff-region) | ✅ shipped |
| Recompare stored runs with adjusted masks (single + batch) | ✅ shipped (unique feature — no competitor has this) |
| Baseline history / rollback | ⏸ deferred (v1.5 in proposal) |
| Listener live mode (WebSocket) | ⏸ deferred (v2) |
| CI gating (`gate <run-id>`) | ⏸ deferred (v2) |
| AI auto-triage (reuse DocTest.Ai) | ⏸ deferred (v2) |

---

## 3. Evaluation

### 3.1 Strengths worth preserving

- **The recompare loop is a genuine differentiator.** No competitor lets you re-run a *historical* comparison with adjusted masks before committing them. Applitools' region annotations are static; ours execute the real engine.
- **Plain-file baselines + git-diffable accepts** (the reg-suit lesson) — kept intact.
- **Honest degraded mode** and honest PDF page-accept semantics — competitors tend to hide such limits.
- **Test discipline**: real robot runs as fixtures, no mocked engine in e2e, contract tests binding library↔dashboard schema, version-skew detection with UI banner.

### 3.2 Implementation issues (verified in code)

| # | Issue | Evidence | Impact |
|---|---|---|---|
| I1 | **N+1 queries** in both list endpoints: per-run counts loop (`app.py` `list_runs`), per-row thumbnail subquery (`list_tests`) | `app.py:121,146` | Run list with 50 runs = 51 queries; 500-test run = 501 queries per grid load |
| I2 | **No pagination anywhere**; `status`/`review_state` filters applied in Python after `SELECT *` | `list_tests` | Large suites (1000+ comparisons) will make grid loads multi-second and payloads huge |
| I3 | **No data lifecycle**: no run deletion (API or UI), engine scratch dirs never GC'd, uploads never expire, in-memory engine cache unbounded, `assets` table append-only | `engine.py` (no rmtree), `db.py` (no DELETE), `uploads/` | Disk + memory grow monotonically; long-lived team server degrades |
| I4 | Engine pool hardcoded (`MAX_WORKERS=2`, `JOB_TIMEOUT=120`); no queue-depth surfacing | `engine.py:26-27` | Batch recompare of 50 comparisons is slow with no feedback |
| I5 | `app.py` is one 550-line `create_app` closure with 26 defs; `MaskEditor.tsx` is 755 lines | measured | Violates the repo's 500-line guideline; onboarding and review friction |
| I6 | Frontend: hand-rolled hash router, no data-fetch layer (raw `fetch` in components), single 465 KB JS bundle (konva always loaded, even on the runs page) | `App.tsx`, build output | Fine today; friction for every next feature |
| I7 | Asset tokens are `sha256(path)[:32]` — deterministic; fine given DB lookup + root check, but registering happens at ingest only, so re-served recompare scratch outputs depend on process lifetime of registrations | `ingest.py:asset_token` | Minor; document or move to random tokens |

### 3.3 UX gaps

| # | Gap | Notes |
|---|---|---|
| U1 | **No zoom/pan** in any viewer mode — document pages at 200 DPI are large; reviewers need to inspect details | Also a **docs-vs-reality bug**: spec `dashboard-review` and docs claim "synced zoom/pan" |
| U2 | **No batch operations**: no accept-all-in-run/suite, no multi-select in the grid | Every competitor centers the workflow on this |
| U3 | Mask editor: no undo/redo, no canvas zoom (fixed fit-to-900px — precision masking on large pages is hard), area/pattern masks not selectable on canvas, masks for other pages still rendered, no duplicate-mask action | 755-line component also mixes concerns |
| U4 | No run management UI (delete/rename/notes); the runs table grows forever with `suite.robot` noise from experiments | |
| U5 | Uploads: no progress indication for large folders, no drag-and-drop | |
| U6 | No dark mode; no keyboard-shortcut help overlay (shortcuts exist but are undiscoverable) | |
| U7 | PdfTest comparisons surface only as note strings ("[text] Page 1 text content differs") — no structured diff view | We *have* facet payloads in the sidecar notes; Argos ships text-artifact diffing as a headline feature |
| U8 | No visibility into engine state (queue depth, running jobs) during batch recompare | |

### 3.4 Product gaps vs. the competitive field

Synthesis of the research (sources at the end):

| Pattern | Applitools | Percy | Chromatic | Argos | **doctest-dashboard** |
|---|---|---|---|---|---|
| Accept/reject per step | ✅ | ✅ | ✅ | ✅ | ✅ (per page — finer) |
| **Group matching diffs, one-click approve group** | ✅ "accept all similar" (premium) | ✅ identical geometry+pixels | — | ✅ groups changes | ❌ |
| Batch accept (build/suite) | ✅ | ✅ | ✅ "Accept all" | ✅ | ❌ |
| Approval carryforward across builds | ✅ | ✅ | ✅ | ✅ | ✅ (identity+content hash) |
| Root-cause context (DOM/CSS ↔ our text-in-region) | — | ✅ DOM/CSS diff | — | — | ❌ (data exists: OCR/PDF text per region) |
| Discussions/annotations pinned to a change | — | — | ✅ | ✅ | ❌ (only free-text reason) |
| Flakiness analytics across runs | — | — | — | ✅ tests-by-flakiness dashboard | ❌ (history is in DB) |
| PR/CI status gating | ✅ | ✅ build-blocking | ✅ denied→fail | ✅ + merge queue | ❌ (deferred `gate`) |
| Non-image artifact diffs (text/JSON) | — | — | — | ✅ | ❌ (PdfTest facets are text!) |
| Skip-unchanged economics (TurboSnap) | — | — | ✅ | — | n/a (local) — but cache-hit recompare is analogous |
| Re-run historical comparisons with new masks | ❌ | ❌ | ❌ | ❌ | ✅ **unique** |

**The one to steal first:** Percy-style *matching-diff grouping*. Document testing amplifies its value: a header change affects every page of every invoice identically. We already store `diff_regions` (geometry) and diff images (hashable) per page — grouping key = normalized region set + diff-image perceptual hash, then "accept group" writes N promotions with one audit trail entry each.

---

## 4. Improvement Proposal (prioritized)

### P0 — Hygiene & table stakes (small, mechanical, do first)

| Item | Sketch |
|---|---|
| **P0.1 Query layer fix** | Replace N+1s with aggregate SQL (`GROUP BY run_id` counts; thumbnail via window function or one `IN` query); push `status`/`review_state` filters into WHERE; add `limit/offset` (or keyset) params + total counts to list endpoints; UI pagination on grid |
| **P0.2 Data lifecycle** | `DELETE /api/runs/{id}` (cascades exist) + UI action; scratch/upload GC (age-based sweep on startup + daily); LRU+TTL bound on engine cache; assets pruned with their runs |
| **P0.3 Viewer zoom/pan** | Wheel-zoom + drag-pan, synchronized across side-by-side panes (transform state shared); honors the existing region-navigation centering. Closes the docs-vs-reality gap |
| **P0.4 Batch accept** | `POST /api/runs/{id}/accept` (+ per-suite), multi-select checkboxes in grid, one decision row per promotion; confirmation dialog with counts. Was promised in the original parity matrix |
| **P0.5 Docs truth pass** | Until P0.3 lands, remove the zoom/pan claim from spec/docs; add screenshots to docs/dashboard.md |

### P1 — Differentiators (leverage data we already collect)

| Item | Sketch |
|---|---|
| **P1.1 Matching-diff groups** | Group unresolved failures by (sorted normalized `diff_regions` geometry, diff-image dHash). Grid gains a "grouped" view: one card = N affected comparisons, expandable, single accept/reject for the group. Percy's rules (identical geometry + pixel-identical diff) are a good strictness baseline; document testing may want a tolerance knob |
| **P1.2 Text-in-region root cause** | Viewer panel: for the selected diff region, show reference vs candidate text (engine already has `_compare_text_content_in_area_with`); highlights *what changed*, not just where — our analog of Percy's DOM/CSS diff, and uniquely valuable for documents |
| **P1.3 PdfTest facet diff view** | Store facet payloads structurally in the sidecar (today: pformat'd note strings); render text/structure facets as proper side-by-side text diffs (Argos-style text artifacts) |
| **P1.4 History & flakiness** | Comparison detail gains a run-over-run timeline (status, score sparkline — data already in DB via `identity`); a "flaky" surface listing identities that flip status across recent runs (Argos' tests-by-flakiness) |
| **P1.5 CI gate + PR surface** | `doctest-dashboard gate <run-id>` exit-code command (deferred v2 item) + optional GitHub PR comment/status posting — the reg-suit/Argos adoption wedge |

### P2 — Strategic

- **Live mode**: Listener v3 → WebSocket run view (original M5; the API/DB are ready for it).
- **Annotations**: region-pinned comments on pages (Chromatic discussions); extends `decisions` with an `annotations` table.
- **Baseline history/rollback**: record promoted-over file content (or lean on git with a "revert" helper); pairs with the audit trail already in place.
- **AI triage**: batch "ask the LLM" over unresolved failures using the library's existing `DocTest.Ai` verdict machinery; verdict shown as a suggestion chip, never auto-accepting (Applitools' lesson: silent auto-accept erodes trust).
- **Identity/auth**: actor from authenticated user once team mode grows beyond a shared token.

### Refactoring plan (enables the above, no behavior change)

1. **Split `server/app.py`** (550 L closure) into routers (`ingest`, `review`, `masks`, `engine`, `assets`, `browse`) with an app-state dataclass — FastAPI `APIRouter` fits; keeps the token dependency shared.
2. **Split `MaskEditor.tsx`** (755 L) into canvas, side-panel, file/IO-toolbar, and hooks (`useMasks`, `usePagePreview`); add component tests (vitest) — currently the frontend has zero unit tests, only e2e.
3. **Introduce a tiny fetch layer** (typed api client already exists — add SWR-style caching or at least a `useApi` hook with loading/error states) and route-level code splitting so konva loads only for the editor (cuts initial bundle roughly in half).
4. **DAL**: move aggregate queries into `db.py` so endpoints stop assembling SQL; add `EXPLAIN`-checked indices for the new filters (`comparisons(review_state)`, `pages(status)`).

### Suggested change slicing (OpenSpec)

```
change: dashboard-hygiene        → P0.1–P0.5 + refactors 1,4   (1 wk)
change: dashboard-diff-groups    → P1.1 + P0.4 interactions    (1 wk)
change: dashboard-root-cause     → P1.2 + P1.3                 (1 wk)
change: dashboard-ci-gate        → P1.5 (+ P1.4 if slack)      (0.5–1 wk)
later:  live-mode / annotations / rollback / ai-triage
```

---

## 5. Risks & open questions

- **Grouping strictness** (P1.1): Percy requires pixel-identical diffs; document renderings may differ by ±1px anti-aliasing across pages. Start strict (exact), add perceptual-hash tolerance behind a knob; wrong grouping that leads to wrong batch-accepts is the failure mode to fear.
- **Batch accept vs. PDF document granularity**: run-level accept must aggregate honestly — PDFs promote per document; the confirmation dialog must state exactly what files will be written.
- **Retention defaults**: silent deletion is worse than growth; GC should default to generous windows (e.g., scratch 7 days, uploads 30, runs never — manual only) and be configurable.
- **Sidecar schema evolution** (P1.3 needs structured facet payloads): additive fields keep v1; bump to v1.1 with the dashboard accepting both.
- Open from the original proposal and still undecided: flip `result_json` default to on; PyPI naming of a discoverability stub.

---

## 6. Sources

- [Applitools — Analyzing Results / review workflow](https://applitools.com/docs/eyes/concepts/reviewing-tests/review-results) · [Applitools Baselines](https://applitools.com/docs/eyes/getting-started/applitools-workflow/baselines) · [Baseline Variations & automated maintenance](https://applitools.com/docs/eyes/concepts/best-practices/baseline-variations)
- [Percy — Build Review & Approval](https://www.browserstack.com/percy/features/build-review-and-approval) · [Snapshot Grouping with Matching Diffs](https://www.browserstack.com/docs/percy/visual-testing-workflows/view-percy-build-results/snapshot-grouping-with-matching-diffs) · [Approval Workflow](https://www.browserstack.com/docs/percy/build-results/approval)
- [Chromatic — Quickstart / review flow](https://docs.chromatic.com/docs/quickstart/) · [TurboSnap](https://docs.chromatic.com/docs/turbosnap/) · [The power of visual testing](https://www.chromatic.com/blog/visual-testing/)
- [Argos CI](https://argos-ci.com/) · [Argos Changelog (flakiness dashboard, tag filters, text-artifact diffs, merge queue)](https://argos-ci.com/changelog) · [argos-ci/argos on GitHub](https://github.com/argos-ci/argos)
- [Lost Pixel](https://www.lost-pixel.com/) · [lost-pixel/lost-pixel](https://github.com/lost-pixel/lost-pixel) · [awesome-regression-testing](https://github.com/mojoaxel/awesome-regression-testing)
