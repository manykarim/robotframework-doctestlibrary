# Design: dashboard-ci-gate

## Context

`comparisons.identity` (suite::test::keyword::index) plus `review_state` already encode everything gating and history need; decision carry-forward across runs (content-hash based) means "unresolved" is exactly "needs human eyes".

## Goals / Non-Goals

**Goals:** dependency-light gate command usable in any CI after `pip install robotframework-doctestlibrary`; history + flakiness read models; UI surfaces.
**Non-Goals:** GitHub PR comments/status posting (follow-up), gate polling/waiting modes.

## Decisions

- **D1 Gate**: `gate <run-id|latest>` opens the SQLite DB directly (`--data-dir` as elsewhere), counts `status='FAIL' AND review_state='unresolved'` for the run, prints per-test lines, exits 0/1; unknown run exits 2. The CLI's dashboard-dependency guard moves into the `serve`/`ingest` branches — `gate` runs on the base install.
- **D2 History**: `db.comparison_history(comparison_id)` joins on identity across runs ordered by import time: run id/name/imported_at, status, review_state, first failing score. Endpoint 404s on unknown comparison. Viewer renders a compact timeline table.
- **D3 Flakiness**: `db.flaky_identities(window=10, min_flips=1)` — for each identity with ≥2 occurrences, count status transitions across its last `window` occurrences; return those with flips, ordered by flip count. Runs page shows a collapsible "Flaky comparisons" panel.
- **D4 Feature flag** `history` appended to `API_FEATURES`/`REQUIRED_FEATURES` (skew banner contract).

## Risks / Trade-offs

- [Gate semantics vs. accepted-after-run] → gate reads *current* review state, not state at ingest time: accepting in the dashboard immediately turns a red gate green, which is the desired workflow (review → re-run gate).
- [Flakiness noise from intentional changes] → accepted flips still count as flips; the panel is informational, not gating.

## Migration Plan

Additive; no schema change (queries over existing columns/indices — `idx_comparisons_identity` exists).
