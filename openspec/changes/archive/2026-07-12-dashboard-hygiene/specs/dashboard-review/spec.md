# Spec delta: dashboard-review (dashboard-hygiene)

## ADDED Requirements

### Requirement: Batch accept
Reviewers SHALL be able to accept multiple comparisons at once: all unresolved failures of a run, or an explicit selection from the grid. Each promotion SHALL produce its own audit row; comparisons that cannot be promoted (degraded, outside roots, missing candidate) SHALL be skipped with a reason without aborting the batch. The UI SHALL require a confirmation that states how many files will be written.

#### Scenario: Accept whole run
- **WHEN** a reviewer confirms "accept all" on a run with unresolved failures
- **THEN** every eligible candidate is promoted with its own decision row and the response lists accepted and skipped items with reasons

#### Scenario: Accept selection
- **WHEN** a reviewer selects specific comparisons and confirms
- **THEN** exactly those comparisons are promoted

#### Scenario: Degraded rows skipped honestly
- **WHEN** a batch contains a degraded comparison
- **THEN** it is skipped with a reason and the rest of the batch proceeds

## MODIFIED Requirements

### Requirement: Diff viewer modes
The detail view SHALL offer side-by-side (synced zoom/pan), overlay, blink, and swipe modes, plus diff-region navigation (next/previous difference) when region data is available from the sidecar. Mode switching SHALL be keyboard-accessible. Zoom (mouse wheel, 0.25×–8×) and pan (drag) SHALL be available in every mode, synchronized across side-by-side panes, with double-click reset; navigating to a diff region SHALL center it at the current zoom.

#### Scenario: Synced zoom in side-by-side
- **WHEN** a reviewer zooms and pans in one side-by-side pane
- **THEN** the other pane shows the same viewport

#### Scenario: Region centering under zoom
- **WHEN** a reviewer at 4× zoom presses "next difference"
- **THEN** the viewport centers on that region without changing the zoom level

#### Scenario: Navigate diff regions
- **WHEN** a reviewer presses "next difference" on a page with three diff regions
- **THEN** the viewport centers on the next region in document order, wrapping after the last

#### Scenario: Degraded record limits viewer honestly
- **WHEN** a comparison was ingested without a sidecar
- **THEN** the viewer shows the combined images, disables region navigation and per-page accept, and explains how to enable `result_json`
