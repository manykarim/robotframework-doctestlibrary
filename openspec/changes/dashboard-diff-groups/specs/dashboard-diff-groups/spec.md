# Spec: dashboard-diff-groups

## ADDED Requirements

### Requirement: Similarity grouping of failing comparisons
Failing sidecar-backed comparisons SHALL receive a deterministic group key derived from their failing pages' diff-region geometry and a perceptual hash of each diff image, computed at ingest. Comparisons with identical differences SHALL share a key; degraded and passing comparisons SHALL never carry one.

#### Scenario: Identical differences group
- **WHEN** two comparisons in a run fail with pixel-identical differences of the same geometry
- **THEN** they share a group key

#### Scenario: Different differences do not group
- **WHEN** two comparisons fail with visually different diffs
- **THEN** their group keys differ

### Requirement: Group review API and view
`GET /api/runs/{id}/groups` SHALL return unresolved failure groups (size ≥ 2) with member ids, names, count, and a sample thumbnail, plus the count of ungrouped unresolved failures. The run page SHALL offer a "by similarity" view with group cards and group-level accept/reject through the existing batch machinery and confirmation flow.

#### Scenario: Group listed with members
- **WHEN** a run contains three unresolved failures, two of them identical
- **THEN** the groups response contains one group of size 2 and reports one ungrouped failure

#### Scenario: Accept a group
- **WHEN** a reviewer confirms *Accept group* on a group of N members
- **THEN** all N candidates are promoted with per-promotion audit rows and the group disappears from the unresolved view

#### Scenario: Resolved members leave the group view
- **WHEN** a member of a group is accepted individually
- **THEN** subsequent group listings exclude it
