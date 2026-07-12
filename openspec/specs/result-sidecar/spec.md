# Spec: result-sidecar

## Purpose
Machine-readable per-comparison result sidecars that external tools (the review dashboard in particular) consume.
## Requirements
### Requirement: Opt-in machine-readable comparison result
`VisualTest` and `PdfTest` SHALL accept a `result_json` option (constructor argument and `Set Result Json` keyword, default off). When enabled, every comparison keyword SHALL write a JSON sidecar to `{OUTPUT_DIR}/doctest_results/{uuid}.json` and log exactly one INFO message of the form `DOCTEST_RESULT: <path relative to OUTPUT_DIR>`, for both passing and failing comparisons.

#### Scenario: Failing image comparison emits sidecar
- **WHEN** `Compare Images` runs with `result_json=True` on two differing images
- **THEN** a sidecar file exists under `doctest_results/`, its `status` is `FAIL`, and the log contains one `DOCTEST_RESULT:` message pointing to it

#### Scenario: Passing comparison emits sidecar
- **WHEN** `Compare Images` runs with `result_json=True` on identical images
- **THEN** a sidecar with `status` `PASS` and per-page records (score, no diff images) is written

#### Scenario: Disabled by default
- **WHEN** a comparison runs without `result_json`
- **THEN** no `doctest_results/` directory is created and no `DOCTEST_RESULT:` message is logged

### Requirement: Sidecar schema v1 content
The sidecar SHALL declare `schema_version: 1` and contain: keyword name, library name, overall status, reference and candidate descriptors (path, page count, rendering DPI), comparison settings (threshold, move tolerance, check_text_content, watermark, screenshot_format, ocr_engine), masks (the source as given plus resolved per-page pixel areas), per-page records (page number, status, SSIM score, threshold, diff regions as `{x, y, width, height}` lists, image paths), and timing (start, elapsed ms).

#### Scenario: Diff regions match the engine's rectangles
- **WHEN** a failing page produces diff rectangles via the engine's contour detection
- **THEN** the page record's `diff_regions` contains exactly those rectangles in absolute pixel coordinates of the rendered page

#### Scenario: Masked comparison records resolved masks
- **WHEN** a comparison runs with a `placeholder_file` containing a pattern mask
- **THEN** the sidecar's masks section contains both the abstract mask definition and the resolved pixel boxes per page

### Requirement: Separate per-page renderings for review
When `result_json` is enabled, the library SHALL save separate reference and candidate renderings per page (and the diff image for failing pages) as lossless PNG regardless of `screenshot_format`, and reference them from the page records.

#### Scenario: Failing PDF comparison saves review images
- **WHEN** `Compare Images` on a multi-page PDF pair fails on a page with `result_json=True`
- **THEN** that page's record references existing PNG files for reference, candidate, and diff renderings, each loadable and matching the page dimensions

#### Scenario: PdfTest emits document-level sidecars
- **WHEN** `Compare Pdf Documents` (text/structure comparison) runs with `result_json=True`
- **THEN** a sidecar with overall status, facet differences as notes, and document descriptors is written; per-page records are reserved for visual comparisons

### Requirement: Reference-run promotion is implemented
When `reference_run` is true (via `${REFERENCE_RUN}` or `Set Reference Run`) and the candidate differs or the reference is missing, the library SHALL save the candidate as the new reference at the reference path and pass the comparison. (Repairs the currently documented but non-functional behavior.)

#### Scenario: Missing reference is created
- **WHEN** `Compare Images` runs in reference-run mode and the reference file does not exist
- **THEN** the candidate file is copied to the reference path and the keyword passes

#### Scenario: Differing candidate replaces reference
- **WHEN** `Compare Images` runs in reference-run mode and images differ
- **THEN** the reference file's content equals the candidate's content after the run and the keyword passes

### Requirement: Unit conversion preserves fractional values
`_convert_to_pixels` SHALL apply the DPI conversion to the original numeric value (int or float) and round only the final pixel result.

#### Scenario: Fractional millimetres convert exactly
- **WHEN** a coordinates mask with `width: 25.4, unit: mm` is resolved at 200 DPI
- **THEN** the resolved pixel width is 200 (not 196)

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

### Requirement: Capture context field
The sidecar schema SHALL carry an optional additive `context` object describing
how the candidate was captured; absent for non-web comparisons.

#### Scenario: Additive compatibility
- **WHEN** a consumer reads a sidecar without `context`
- **THEN** parsing succeeds exactly as before

