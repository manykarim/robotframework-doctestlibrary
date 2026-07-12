# Proposal: dashboard-hygiene

## Why

The dashboard's v1 list endpoints issue N+1 queries and load entire tables (a 500-test run costs 501 queries per grid view), nothing is ever deleted (runs, engine scratch, uploads, caches grow forever), the diff viewer lacks zoom/pan that the docs already claim, and there are no batch review operations — the workflow every comparable product centers on. See `docs/dashboard-improvement-analysis.md` (P0 tier). The dashboard is unreleased: no backward compatibility required.

## What Changes

- **Query layer**: aggregate SQL replaces per-row loops (run counts, grid thumbnails); `status`/`review_state` filters move into SQL; list endpoints gain `limit`/`offset` + total counts; supporting indices; grid pagination in the UI.
- **Data lifecycle**: `DELETE /api/runs/{id}` (with asset pruning) + delete action in the UI; age-based garbage collection for engine scratch and uploads (startup sweep, configurable windows, generous defaults); bounded LRU engine cache.
- **Viewer zoom/pan**: wheel zoom + drag pan, synchronized across side-by-side panes, double-click reset; region navigation centers within the transform. Makes the existing docs claim true.
- **Batch accept**: accept-all-unresolved at run level and accept-selected via grid multi-select, with a confirmation dialog stating exactly what will be promoted; per-promotion audit rows preserved; PDF/degraded comparisons reported honestly in the result.
- **Refactors (no behavior change)**: split the 550-line `create_app` closure into FastAPI routers with shared state; move endpoint SQL into the DAL; lazy-load the mask editor route so konva leaves the initial bundle.

## Capabilities

### New Capabilities

- `dashboard-lifecycle`: run deletion, storage garbage collection, bounded caches, paginated queries.

### Modified Capabilities

- `dashboard-review`: batch accept operations; viewer zoom/pan requirement made real.

## Impact

- `doctest_dashboard/` (server split into `server/routers/*`, db.py aggregate queries, engine.py cache/GC), `frontend/src/` (grid pagination + selection, viewer transforms, route splitting), tests in `utest/dashboard/` + one new e2e journey. No library changes. Existing e2e data-testids preserved.
