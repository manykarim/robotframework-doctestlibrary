# Proposal: library-sidecar-v11

## Why

The screenshot-grounded UX review (docs + session findings) traced several dashboard weaknesses to data the library doesn't emit: grid thumbnails are near-black absolute-diff images because no highlighted rendering is available; runs are unidentifiable ("Suite") because no run-level metadata exists; Explain-Region re-runs extraction the library already performed; PdfTest facets arrive as `pformat` blobs; and comparison identity is synthesized from test names, breaking on renames.

## What Changes

All additive to sidecar schema v1 (consumers tolerate extra fields):

- **Highlighted renderings + thumbnails**: failing pages additionally save `candidate_with_diff` (clean candidate with diff-region outlines) and a small `thumb` PNG — proper grid thumbnails and a highlight view for the dashboard.
- **Run manifest**: first sidecar write also creates `doctest_results/run.json` (suite start time, library/robot/python versions, tesseract version if detectable, platform) for run naming and environment display.
- **Pre-extracted region text**: when PDF text is available without OCR, failing page records include per-diff-region reference/candidate text (`regions_text`) — instant Explain-Region for the common case.
- **Structured facet payloads**: PdfTest facet entries gain a `data` field with the JSON-safe payload alongside the formatted `details` string.
- **`name=` comparison labels**: `Compare Images`/`Compare Pdf Documents` accept an optional `name` argument recorded in the sidecar for stable identity and human-readable display.

## Capabilities

### Modified Capabilities

- `result-sidecar`: adds highlighted renderings/thumbnails, run manifest, region text, structured facet data, and comparison names to the sidecar contract.

## Impact

`DocTest/VisualTest.py`, `DocTest/PdfTest.py`, `DocTest/ResultWriter.py`; dashboard model `doctest_dashboard/models/sidecar.py` (additive fields). Unit tests in `utest/test_result_json.py`; contract tests keep passing. Dashboard consumption lands separately in `dashboard-visual-polish`.
