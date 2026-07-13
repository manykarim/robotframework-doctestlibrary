# Spec: comparison-performance

## Purpose
Performance contract of the comparison engine: direct pixmap conversion with identical rendering output, on-demand PDF text extraction, and a benchmark harness guarding the render/compare hot paths.
## Requirements
### Requirement: Direct pixmap conversion preserves rendered output exactly
PDF page rendering SHALL convert the PyMuPDF pixmap buffer directly to the BGR image array without an intermediate PNG encode/decode, and the resulting image MUST be bit-identical to the previous PNG-round-trip output for the same document, page, and DPI.

#### Scenario: Bit-exact rendering
- **WHEN** a PDF page is rendered at a given DPI via the direct pixmap conversion
- **THEN** the numpy image equals (array-equal) the image produced by encoding the same pixmap to PNG and decoding it with OpenCV

### Requirement: PDF text extraction is on demand
The four per-page PDF text representations (`text`, `dict`, `words`, `blocks`) SHALL be extracted only when first accessed. A comparison that needs none of them MUST NOT invoke the corresponding extraction. Accessing any representation SHALL return the same value as eager extraction did, including after the source document handle is closed.

#### Scenario: Pixel-only comparison skips dict extraction
- **WHEN** `Compare Images` runs on two PDFs with no masks, no text checks, and no move tolerance
- **THEN** the expensive `dict` text extraction is never invoked for either document

#### Scenario: Lazy values equal eager values
- **WHEN** a text-based check (e.g. pattern ignore areas or `check_text_content`) accesses a page's text representations
- **THEN** the returned values are identical to those previously produced by eager extraction, even after the document is closed

### Requirement: Benchmark harness guards the hot paths
The repository SHALL provide a pytest-benchmark suite covering PDF page loading and single-page visual comparison, runnable via pytest, so performance regressions in the render/compare hot paths are measurable.

#### Scenario: Benchmarks run
- **WHEN** the benchmark suite is executed with pytest
- **THEN** it reports timings for PDF page loading and image comparison without failures
