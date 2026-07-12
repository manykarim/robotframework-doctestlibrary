# Spec: dashboard-ci-gate

## ADDED Requirements

### Requirement: CI gate command
`doctest-dashboard gate <run-id|latest>` SHALL exit 0 when the run has no unresolved failed comparisons and 1 otherwise, printing which comparisons remain unreviewed; unknown runs exit 2. The command SHALL work on a base install without the dashboard extra.

#### Scenario: Unreviewed failures block
- **WHEN** the gate runs against a run with unresolved failed comparisons
- **THEN** it exits 1 and lists the unresolved comparisons

#### Scenario: Reviewed run passes
- **WHEN** every failed comparison of the run has been accepted or rejected
- **THEN** the gate exits 0

#### Scenario: Base install sufficient
- **WHEN** the gate runs in an environment without fastapi/uvicorn installed
- **THEN** it works (no dashboard-extra dependency)

### Requirement: Cross-run history
`GET /api/comparisons/{id}/history` SHALL return the comparison identity's timeline across ingested runs (run, time, status, review state, score), newest first, and the comparison view SHALL render it.

#### Scenario: Timeline across two runs
- **WHEN** the same test identity was ingested in two runs
- **THEN** the history of either comparison lists both occurrences with their statuses

### Requirement: Flakiness surface
`GET /api/flaky` SHALL list comparison identities whose status flipped across their recent occurrences, ordered by flip count, and the runs page SHALL surface them.

#### Scenario: Flipping identity reported
- **WHEN** an identity passed in one ingested run and failed in another
- **THEN** it appears in the flaky listing with at least one flip
