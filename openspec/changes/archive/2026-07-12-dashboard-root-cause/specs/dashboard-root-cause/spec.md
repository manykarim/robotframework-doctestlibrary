# Spec: dashboard-root-cause

## ADDED Requirements

### Requirement: Region text explanation
For a sidecar-backed comparison, the dashboard SHALL extract and compare the text inside a given diff region of a given page — using the library's own region text comparison at the comparison's recorded DPI — and return reference text, candidate text, and a same/different verdict. The viewer SHALL offer this for the selected diff region on demand.

#### Scenario: Changed text explained
- **WHEN** a reviewer requests the text of a diff region that covers changed text
- **THEN** the response contains the differing reference and candidate texts and `same` is false

#### Scenario: Degraded comparison refused
- **WHEN** region text is requested for a degraded comparison
- **THEN** the request fails with a conflict explaining that source paths are unavailable

#### Scenario: Root confinement
- **WHEN** the stored reference/candidate paths lie outside the configured roots
- **THEN** the request is rejected with 403

### Requirement: Structured PdfTest facets
PdfTest sidecars SHALL carry the computed facet differences as a structured `facets` list (facet, description, details) in addition to flattened notes, and the comparison detail view SHALL render one section per facet with its formatted payload.

#### Scenario: Facets stored structurally
- **WHEN** `Compare Pdf Documents` fails with text and metadata differences under `result_json`
- **THEN** the sidecar's `facets` list contains one entry per difference with facet name, description, and details

#### Scenario: Facets rendered
- **WHEN** a reviewer opens a failed PdfTest comparison in the dashboard
- **THEN** each facet difference is shown as its own section with the formatted payload
