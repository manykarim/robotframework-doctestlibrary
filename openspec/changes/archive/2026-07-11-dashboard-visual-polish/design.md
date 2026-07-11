# Design: dashboard-visual-polish

## Decisions

- **D1 Thumbnail preference**: `thumb` → `candidate_with_diff` → `diff` → `candidate` in `list_grid`/`list_groups`; ingest registers the new images as assets like any others (no special casing).
- **D2 Viewer fit**: image `onLoad` reports natural size; `useViewTransform.fitTo(w, h, paneRect)` sets `scale = min(paneW/w, paneH/h, 1)` centered; runs once per page/image change. Toolbar gains `Fit` and `100%`; double-click now resets to *fit* (the useful default). `highlight` mode (key 5) renders `candidate_with_diff` when present, hidden otherwise.
- **D3 Region outlines**: all `page.regions` render as dashed outlines inside the transformed content (checkbox, default on); the selected region keeps the solid highlight. Outline widths divide by scale.
- **D4 Naming**: ingest stores `runs.label = NULL` (user-editable via `PATCH /api/runs/{id} {label}`) and derives `name` as `"{suite} · {output-dir-name}"`; UI renders `label || name` plus relative time (`title` keeps the ISO stamp). Comparisons store the sidecar `name` as `comparisons.label`; identity becomes `name::<label>` when a label exists (stable across test renames), else the existing scheme. Grid/header prefer the label.
- **D5 Chrome/state**: `viewMode` switch clears selection; `accept-selected` renders only in flat view; comparisons with zero pages skip the mode/zoom toolbar; the decision bar gets `position: sticky; bottom: 0`.
- **D6 Browser roots**: two-line entries — basename bold, full path dimmed small.
- **D7 Schema**: `label` columns via defensive `ALTER TABLE` (unreleased; dev DBs only).

## Risks

- [Identity change orphans carry-forward for previously ingested labeled runs] → labels are new; old runs have none, so old identities are untouched.
- [Fit-on-load racing region centering] → fit runs on image load; centering runs on explicit region selection afterwards — ordered by user action.
