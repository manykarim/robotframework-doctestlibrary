# Proposal: dashboard-diff-groups

## Why

Document testing multiplies identical differences: one layout change hits every page of every invoice the same way, and today each comparison must be reviewed individually. Percy's "matching diffs" grouping and Applitools' premium "accept all similar" show this is the heart of an efficient review workflow — and our sidecars already carry the data (diff-region geometry + diff images) to build it (analysis P1.1).

## What Changes

- **Grouping at ingest**: every failing sidecar-backed comparison gets a `group_key` — a hash of its failing pages' normalized diff-region geometry plus a perceptual hash (dHash) of each diff image. Identical differences across documents/tests yield identical keys; grouping is strict (Percy-style) to make batch accepts safe.
- **Groups API**: `GET /api/runs/{id}/groups` returns unresolved failure groups (count, members, sample thumbnail) plus the ungrouped remainder.
- **Grouped review UI**: the run page gains a "by similarity" view — one card per group with member list and *Accept group (n)* / *Reject group* actions, reusing the batch machinery and confirmation bar from dashboard-hygiene.

## Capabilities

### New Capabilities

- `dashboard-diff-groups`: similarity grouping of failing comparisons with group-level review actions.

## Impact

`doctest_dashboard/ingest.py` (group key computation), `db.py` (column + group query), new router endpoint, `frontend/src/App.tsx` grouped view. Additive sidecar-consumer change only; degraded records stay ungrouped. Tests: unit grouping determinism, groups endpoint, group accept; one e2e journey.
