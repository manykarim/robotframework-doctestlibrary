# Design: dashboard-hygiene

## Context

Verified issues (analysis I1–I6, U1/U2/U4): N+1 loops at `server/app.py:121,146`; no pagination; no deletion/GC anywhere; unbounded `EngineService._cache`; 550-line app closure; 755-line MaskEditor; single 465 KB bundle. Unreleased — free to restructure.

## Goals / Non-Goals

**Goals:** P0 items land with tests; backend restructure into routers; all 90+ existing dashboard tests and 12 e2e journeys stay green (testids stable).
**Non-Goals:** MaskEditor decomposition (deferred until diff-groups needs it), auth, dark mode, upload progress.

## Decisions

- **D1 Queries**: `db.list_runs(limit, offset)` returns runs joined with one aggregate subquery (`COUNT`, `SUM(CASE…)`); `db.list_grid(run_id, status, review_state, limit, offset)` returns rows + thumbnail via a single correlated subquery (SQLite handles it with the new `idx_pages_comparison_status` index) and a `total` count. Endpoints become thin.
- **D2 Lifecycle**: `db.delete_run(id)` relies on existing `ON DELETE CASCADE` (comparisons/pages/tests) plus explicit asset pruning (assets referenced only by that run's pages/comparisons). GC: `storage.py` sweep — scratch dirs older than `scratch_ttl_days` (default 7), upload dirs older than `uploads_ttl_days` (default 30) — run at app startup; runs are deleted manually only. Engine cache becomes an LRU (`OrderedDict`, max 256 entries).
- **D3 Zoom/pan**: a `useViewTransform` hook (scale 0.25–8, translate) + a `ZoomPane` wrapper applying one CSS transform; side-by-side renders two panes sharing the hook state (that *is* the sync). Wheel = zoom around cursor; drag = pan; double-click = reset; region centering sets translate from region center × scale. Highlight boxes live inside the transformed container, so they scale for free.
- **D4 Batch accept**: `review.accept_many(db, config, comparison_ids, actor, reason)` loops `accept_comparison`, collecting `{accepted:[], skipped:[{id, reason}]}` — degraded records and root violations are skipped with reasons, never aborting the batch. Endpoints: `POST /api/comparisons/accept-batch {ids}` and `POST /api/runs/{id}/accept` (server resolves unresolved-failed ids first, honoring current filters is the client's job). UI: header checkbox + row checkboxes; sticky action bar shows selection count; confirm dialog lists counts incl. how many are PDFs (document-level promotion) before writing.
- **D5 Routers**: `server/routers/{ingest,review,masks,engine,assets,browse}.py`, each an `APIRouter` receiving `AppState` (config, db, engine) via dependency; `create_app` shrinks to wiring + static mount + health. Feature list unchanged (`API_FEATURES` + `batch-accept`, `lifecycle` entries so the skew banner works).
- **D6 Bundle**: `React.lazy(() => import("./MaskEditor"))` + Suspense fallback; konva chunk loads only on the editor route.

## Risks / Trade-offs

- [Batch accept writing many baseline files] → confirmation dialog with explicit counts; per-file audit rows; skipped-with-reason semantics instead of transactional all-or-nothing (file ops aren't transactional anyway).
- [GC deleting scratch under a running recompare] → sweep only dirs older than TTL (days), recompare scratch lives minutes; startup-only sweep avoids mid-request races.
- [Router split breaking the version-skew contract test] → `API_FEATURES` stays in one module imported by the contract test; test updated for the new location.

## Migration Plan

Single change; e2e suite is the regression net. No data migration (new columns/indices via `CREATE INDEX IF NOT EXISTS`; DB file is dev-local and unreleased).
