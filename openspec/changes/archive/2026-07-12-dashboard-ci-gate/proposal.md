# Proposal: dashboard-ci-gate

## Why

Review state currently lives only in the UI — CI cannot block a merge on "unreviewed visual changes", which is the adoption wedge every competitor ships (Percy build-blocking, Argos merge-queue checks, reg-suit status checks). And although the database already holds full cross-run history per comparison identity, reviewers can see neither a comparison's history nor which comparisons are flaky (Argos' flakiness dashboard) — analysis P1.4/P1.5.

## What Changes

- **`doctest-dashboard gate <run-id|latest>`**: exit-code command for CI — 0 when a run has no unresolved failed comparisons, 1 otherwise with a summary of what is unreviewed. Works on a base install (database only, no server dependencies); the CLI dependency guard moves to the commands that actually need the extra.
- **History**: `GET /api/comparisons/{id}/history` returns the identity's status/review timeline across runs; the comparison view renders it.
- **Flakiness**: `GET /api/flaky` lists identities whose status flipped across recent occurrences; the runs page surfaces them.
- Feature flag `history`.

## Capabilities

### New Capabilities

- `dashboard-ci-gate`: CI gating on review state plus cross-run history and flakiness surfaces.

## Impact

`doctest_dashboard/cli.py` (gate + guard scoping), `db.py` (history/flaky queries), runs router, viewer/run-list UI panels. Tests: gate exit codes, history/flaky endpoints, e2e history panel. GitHub PR-comment posting stays out of scope (documented follow-up).
