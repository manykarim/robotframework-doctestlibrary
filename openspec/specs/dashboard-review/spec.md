# Spec: dashboard-review

## Purpose
Reviewing visual comparison results in a web dashboard: viewers, decisions, batch review, identification and visual identity.
## Requirements
### Requirement: Run and test browsing
The dashboard SHALL present a run list with pass/fail/unresolved counts and a test grid per run with status filters and diff thumbnails (first failing page's diff). Failed comparisons SHALL enter review state `unresolved`.

#### Scenario: Filter unresolved failures
- **WHEN** a reviewer filters the test grid by `unresolved`
- **THEN** only comparisons with failed pages lacking a decision are listed, each with a diff thumbnail

### Requirement: Diff viewer modes
The detail view SHALL offer side-by-side (synced zoom/pan), overlay, blink, and swipe modes, plus diff-region navigation (next/previous difference) when region data is available from the sidecar. Mode switching SHALL be keyboard-accessible.

#### Scenario: Navigate diff regions
- **WHEN** a reviewer presses "next difference" on a page with three diff regions
- **THEN** the viewport centers on the next region in document order, wrapping after the last

#### Scenario: Degraded record limits viewer honestly
- **WHEN** a comparison was ingested without a sidecar
- **THEN** the viewer shows the combined images, disables region navigation and per-page accept, and explains how to enable `result_json`

### Requirement: Accept promotes the baseline with audit
Accept SHALL copy the candidate source file over the reference file. For single-image comparisons accept SHALL be offered per page; for multi-page PDF sources accept SHALL be offered at document granularity only, with an explanation and a mask-creation alternative. Every promotion SHALL record actor, timestamp, optional reason, and SHA-256 of the file before and after in the `decisions` table. The resulting file layout SHALL be identical to what a `REFERENCE_RUN` would produce.

#### Scenario: Accept image candidate
- **WHEN** a reviewer accepts a failed single-image comparison
- **THEN** the reference file's bytes equal the candidate's, the comparison state becomes `accepted`, and a decision row stores both hashes

#### Scenario: PDF page accept redirected
- **WHEN** a reviewer attempts a page-level accept on a multi-page PDF comparison
- **THEN** the UI offers document-level accept or mask creation, and no partial file write occurs

#### Scenario: Promotion refused outside roots
- **WHEN** a comparison's reference path resolves outside the configured roots
- **THEN** accept is rejected with an explanatory error and no file is written

### Requirement: Reject with bug-data export
Reject SHALL store an optional reason, mark the comparison `rejected`, and offer a downloadable ZIP bundle containing reference, candidate, diff images, the sidecar, and decision metadata.

#### Scenario: Bug bundle contents
- **WHEN** a reviewer rejects a failed comparison and downloads the bundle
- **THEN** the ZIP contains the reference file, candidate file, every failing page's diff image, the sidecar JSON, and a metadata file with reason and timestamps

### Requirement: New runs reset stale decisions
When a newer run containing the same comparison (same suite/test/keyword identity) is ingested, pages whose images changed SHALL return to `unresolved` while prior decisions remain in history.

#### Scenario: Changed page reopens
- **WHEN** an accepted comparison reappears in a newer ingested run with a different candidate image
- **THEN** its state is `unresolved` and the earlier decision is still queryable

### Requirement: End-to-end review journey
The full journey — ingest a real run, browse, view diffs, accept one comparison, reject another with a bundle — SHALL be covered by automated browser tests against a served instance with no mocked backend.

#### Scenario: Journey test passes
- **WHEN** the Playwright suite runs the ingest→review→accept→reject journey against real robot output
- **THEN** all steps pass, the accepted reference file changed on disk, and the audit rows exist

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

### Requirement: Design token layer
All dashboard chrome colors, radii and shadows SHALL be defined as CSS custom properties in a single token layer; component styles SHALL reference tokens only.

#### Scenario: One place to restyle
- **WHEN** a token value changes
- **THEN** every component using that semantic role follows without per-component edits

### Requirement: Dark mode
The dashboard SHALL support a dark theme: initial value from the OS preference, a persistent user toggle in the top bar, and document renderings kept on neutral-light surfaces in both themes.

#### Scenario: Toggle persists
- **WHEN** a reviewer switches to dark mode and reloads the page
- **THEN** the dashboard renders dark without flashing light first

### Requirement: Empty states
A dashboard without ingested runs SHALL show an onboarding empty state naming the ingestion paths, and the mask editor without a loaded document SHALL point to Browse/Upload instead of rendering non-functional controls.

#### Scenario: Fresh install
- **WHEN** the runs page loads with zero runs
- **THEN** an onboarding card explains how to ingest results instead of an empty table

