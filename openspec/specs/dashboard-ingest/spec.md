# Spec: dashboard-ingest

## Requirements

### Requirement: Ingest output.xml via CLI and API
The dashboard SHALL ingest Robot Framework `output.xml` files through `doctest-dashboard ingest <path>` and `POST /api/ingest`, parsing with `robot.api.ExecutionResult`, and persist runs, tests, comparisons, and pages to SQLite. Ingestion SHALL be idempotent per `output.xml` (re-ingesting the same file updates rather than duplicates the run).

#### Scenario: Real run is ingested
- **WHEN** an `output.xml` produced by an actual robot run of the library's acceptance data is ingested
- **THEN** the run appears in `GET /api/runs` with test counts, and each DocTest comparison keyword appears with its status

#### Scenario: Re-ingest is idempotent
- **WHEN** the same `output.xml` is ingested twice
- **THEN** exactly one run record exists for it

### Requirement: Keyword-level status extraction
Ingestion SHALL derive comparison status from the DocTest library keyword's own result, not the test's, so comparisons wrapped in `Run Keyword And Expect Error` (test PASS, keyword FAIL) are recorded as failed comparisons.

#### Scenario: Expect-error wrapper does not hide the failure
- **WHEN** a test wraps a failing `Compare Images` in `Run Keyword And Expect Error` and the run is ingested
- **THEN** the comparison record's status is FAIL while the owning test's status is PASS

### Requirement: Sidecar-first with scraping fallback
For each comparison keyword, ingestion SHALL load the sidecar referenced by a `DOCTEST_RESULT:` message when present; otherwise it SHALL fall back to extracting `<img src>` paths from HTML log messages and mark the comparison as `degraded` (combined images only, no per-page granularity).

#### Scenario: Sidecar preferred
- **WHEN** a comparison logged both a sidecar reference and HTML images
- **THEN** the stored record contains per-page data from the sidecar

#### Scenario: Legacy output still ingestible
- **WHEN** an `output.xml` from a run without `result_json` is ingested
- **THEN** comparisons are stored with their combined screenshot paths and flagged `degraded`

### Requirement: Asset serving is root-confined
Screenshot, rendering, and sidecar files SHALL be served only through opaque asset tokens resolving to paths under configured roots; requests resolving outside any root (including via symlinks or `..`) SHALL be rejected with 403, and raw filesystem paths SHALL never appear in API responses.

#### Scenario: Traversal rejected
- **WHEN** an asset token is forged or manipulated to reference `/etc/passwd`
- **THEN** the server responds 403 and serves nothing

#### Scenario: Run artifact served
- **WHEN** a page record's diff image is requested via its asset token
- **THEN** the PNG bytes are returned with cache headers
