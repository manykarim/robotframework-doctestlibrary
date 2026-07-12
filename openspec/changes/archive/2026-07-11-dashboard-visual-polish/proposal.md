# Proposal: dashboard-visual-polish

## Why

The screenshot review exposed concrete usability defects: grid/group thumbnails are near-black absolute-diff images; every document loads cut-off because the viewer never fits the page; runs are indistinguishable ("Suite", raw ISO timestamps); the comparison header never names the test; diff regions are invisible until keyboard-stepped; stale grid selection leaks into the groups view; PdfTest comparisons render dead viewer chrome; the file browser lists unscannable full paths; and decisions sit at the bottom of long pages.

## What Changes

- **Thumbnails**: ingest prefers the sidecar's new `thumb`/`candidate_with_diff` renderings (library-sidecar-v11) for grid and group thumbnails, falling back for old sidecars.
- **Viewer**: auto-fit on image load, Fit/100% controls, a fifth `highlight` mode showing `candidate_with_diff`, and an always-on (toggleable) outline of all diff regions.
- **Identification**: run display names derive from suite + folder; runs gain an editable label (`PATCH /api/runs/{id}`); timestamps render relatively; the comparison header shows the test name (or the sidecar `name` label) with the keyword demoted to a chip; sidecar `name` labels also become the comparison identity for robust history.
- **State/chrome fixes**: switching grid views clears selection; accept-selected hidden in groups view; page-less (PdfTest) comparisons hide the mode toolbar; decision bar sticks to the viewport bottom.
- **Browser/editor polish**: picker roots show basename + dimmed path; consistent default mask names; disabled Save/Load explain themselves; editor canvas fits height too.

## Capabilities

### Modified Capabilities

- `dashboard-review`: viewer fit/highlight/region-outline requirements; identification requirements.
- `dashboard-ingest`: thumbnail selection and label-based identity.

## Impact

`doctest_dashboard/` (ingest, db columns `runs.label`/`comparisons.label`, runs router), `frontend/src/` (viewer, grid, browser, editor). Tests updated + extended; e2e journey J13.
