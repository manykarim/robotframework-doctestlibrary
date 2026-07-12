# Tasks: dashboard-hygiene

## 1. Backend restructure + query layer

- [x] 1.1 Split `server/app.py` into `server/routers/{ingest,review,masks,engine,assets,browse}.py` with shared `AppState`; `API_FEATURES` in one importable place; all existing tests green
- [x] 1.2 DAL aggregate queries: `list_runs` (single aggregate join), `list_grid` (filters in SQL, thumbnail subquery, limit/offset + total); indices for `comparisons(review_state)`, `pages(comparison_id,status)`
- [x] 1.3 Endpoint + UI pagination (grid pager, page size 50); tests for pagination, filters-in-SQL, constant query count

## 2. Lifecycle

- [x] 2.1 `db.delete_run` + `DELETE /api/runs/{id}` with asset pruning; UI delete with confirm; tests incl. cross-run isolation
- [x] 2.2 `storage.py` GC sweep (scratch/uploads TTLs, startup hook) + bounded LRU engine cache; tests with aged dirs and cache eviction

## 3. Review UX

- [x] 3.1 `review.accept_many` + `POST /api/comparisons/accept-batch` + `POST /api/runs/{id}/accept`; skip-with-reason semantics; tests (run accept, selection accept, degraded skip)
- [x] 3.2 Grid multi-select + sticky action bar + confirmation dialog; run-level accept-all button; e2e journey (multi-select accept → files changed on disk)
- [x] 3.3 Viewer zoom/pan (`useViewTransform` + `ZoomPane`), synced side-by-side, region centering under zoom; e2e assertions for zoom state

## 4. Polish

- [x] 4.1 Lazy-load MaskEditor route (konva out of initial bundle); frontend builds; e2e green
- [x] 4.2 Docs: dashboard.md review section (batch accept, zoom controls, run deletion, GC defaults); feature flags for skew banner
- [x] 4.3 Full verification: dashboard tests + all e2e journeys green
