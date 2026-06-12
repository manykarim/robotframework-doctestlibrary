# Spec: live-recompare

## ADDED Requirements

### Requirement: Embedded comparison engine
The dashboard server SHALL import the DocTest library directly and execute comparisons outside any Robot Framework run, in a worker pool with per-job timeouts, never blocking the API event loop. At startup the server SHALL run the library's capability check and expose detected capabilities (e.g., tesseract availability) via the API.

#### Scenario: Engine works without Robot
- **WHEN** the server executes a comparison job on two stored artifacts
- **THEN** results (status, per-page scores, diff regions, diff images) are produced without a Robot Framework process

#### Scenario: Capability surfaced
- **WHEN** the server starts in an environment without tesseract
- **THEN** the capabilities endpoint reports OCR unavailable and OCR-dependent features are flagged in API responses

### Requirement: Mask preview resolution
`POST /api/mask-preview` SHALL accept a document/image reference, page number, and mask definitions, resolve them through `DocumentRepresentation`, and return the resolved pixel ignore areas and the page's rendering DPI.

#### Scenario: Pattern resolves to pixel boxes
- **WHEN** a pattern mask matching on-page text is previewed
- **THEN** the response contains the bounding boxes the library would mask and the page DPI

### Requirement: Re-comparison of stored runs with adjusted masks
`POST /api/recompare` SHALL re-run a stored comparison's reference and candidate with user-supplied masks (and optionally adjusted threshold/move tolerance), writing outputs to a scratch area — original run artifacts and baselines SHALL never be modified — and returning fresh per-page status, scores, diff regions, and diff image references.

#### Scenario: Adjusted mask flips result
- **WHEN** a stored failing comparison (date differs) is re-compared with a pattern mask covering the date
- **THEN** the response reports PASS and the original run's records and files are unchanged

#### Scenario: Recompare batch across history
- **WHEN** a user requests recompare of a mask against all stored comparisons that reference the same masks.json
- **THEN** each comparison reports its would-be status, enabling "which historical failures would this mask suppress" review

### Requirement: Responsiveness controls
Recompare and preview jobs SHALL be queued with progress reporting; identical requests SHALL be served from cache keyed on (artifact hashes, mask set, settings); concurrent job count SHALL be bounded.

#### Scenario: Duplicate request cached
- **WHEN** the same recompare request is submitted twice in a row
- **THEN** the second response is served from cache without re-executing the engine
