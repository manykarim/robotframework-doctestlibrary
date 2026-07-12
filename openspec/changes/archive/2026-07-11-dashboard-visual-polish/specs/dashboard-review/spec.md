# Spec delta: dashboard-review (dashboard-visual-polish)

## ADDED Requirements

### Requirement: Reviewable thumbnails
Grid and group thumbnails SHALL use the highlighted candidate renderings when the sidecar provides them, never the raw absolute-diff image.

#### Scenario: v1.1 sidecar thumbnails
- **WHEN** a run produced with the current library is ingested
- **THEN** grid thumbnails reference the sidecar's `thumb` images

### Requirement: Viewer fit and highlight mode
The viewer SHALL fit the page into the pane on load (Fit and 100% controls provided, double-click refits) and SHALL offer a `highlight` mode rendering the highlighted candidate when available.

#### Scenario: Page visible on open
- **WHEN** a comparison larger than the pane opens
- **THEN** the whole page is visible without manual zooming

### Requirement: Region visibility
All detected diff regions SHALL be outlined in the viewer by default (toggleable), with the selected region visually distinct.

#### Scenario: Regions visible without stepping
- **WHEN** a failing page with three regions opens
- **THEN** three outlines are visible before any keyboard interaction

### Requirement: Identification
Runs SHALL display distinguishable names (suite + source folder, user-editable label, relative timestamps); the comparison header SHALL show the test name or sidecar label, and labeled comparisons SHALL keep their identity across test renames.

#### Scenario: Editable run label
- **WHEN** a reviewer renames a run
- **THEN** the list shows the label and the original name remains as fallback metadata

#### Scenario: Label-stable identity
- **WHEN** two runs contain comparisons labeled `name=Invoice header` under different test names
- **THEN** they share one identity (history joins them)

### Requirement: Focused chrome and state
Grid selection SHALL reset when switching between flat and similarity views; page-less comparisons SHALL NOT render image-viewer controls; decision actions SHALL remain visible while scrolling.

#### Scenario: No stale selection in groups view
- **WHEN** rows are selected in the flat view and the reviewer switches to the similarity view
- **THEN** no selection-dependent actions are offered there
