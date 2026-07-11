# Spec delta: result-sidecar (library-sidecar-v11)

## ADDED Requirements

### Requirement: Highlighted renderings and thumbnails
Failing page records SHALL additionally reference a `candidate_with_diff` PNG (the clean candidate rendering with the page's diff regions outlined) and a small `thumb` PNG derived from it.

#### Scenario: Failing page ships highlighted images
- **WHEN** a comparison fails with `result_json` enabled
- **THEN** the failing page's images include existing `candidate_with_diff` and `thumb` files, and the highlighted candidate differs from the clean candidate

### Requirement: Run manifest
The first sidecar write of a run SHALL create `doctest_results/run.json` containing start time, DocTest/Robot Framework/Python versions, platform, and the tesseract version when detectable.

#### Scenario: Manifest written once
- **WHEN** two comparisons run with `result_json` enabled
- **THEN** exactly one `run.json` exists with the version fields populated

### Requirement: Pre-extracted region text
When PDF text is available without OCR, failing page records SHALL include `regions_text` — per diff region, the reference and candidate text with a same/different verdict.

#### Scenario: PDF failure carries region text
- **WHEN** a PDF comparison fails on a page with embedded text
- **THEN** the page record's `regions_text` pairs each diff region with its reference/candidate text

#### Scenario: Image-only comparisons skip it
- **WHEN** an image comparison fails (OCR would be required)
- **THEN** `regions_text` is absent or empty and nothing slows the comparison down

### Requirement: Structured facet data
PdfTest facet entries SHALL carry a JSON-safe `data` payload alongside the human-formatted `details` string.

#### Scenario: Metadata facet is machine-readable
- **WHEN** a PdfTest comparison fails on metadata
- **THEN** the facet's `data` contains the structured difference (not just a formatted string)

### Requirement: Comparison labels
`Compare Images` and `Compare Pdf Documents` SHALL accept an optional `name` argument recorded as the sidecar's top-level `name`.

#### Scenario: Label lands in the sidecar
- **WHEN** `Compare Images    ref    cand    name=Invoice header`
- **THEN** the sidecar's `name` is "Invoice header"
