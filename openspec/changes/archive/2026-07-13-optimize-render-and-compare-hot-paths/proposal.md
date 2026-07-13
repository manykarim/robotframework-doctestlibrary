# Proposal: optimize-render-and-compare-hot-paths

## Why

Profiling (docs/ultradeep-analysis-solution-proposal.md §4) showed the PDF comparison hot path wastes most of its time on avoidable work: a PNG encode/decode round-trip per rendered page (276 → 85 ms/page when removed, verified bit-exact), eager extraction of four text formats even for pure pixel compares (~75 ms/page), int64 temporaries in the anti-aliasing pixel count (2.6×), and ~11 MB/page of diff-image copies that only the LLM path reads. Together: ~3.9× faster PDF page loading with zero output change.

## What Changes

- Add a pytest-benchmark harness (dev dependency + canonical benchmarks for render/compare) so gains are measured and future regressions surface.
- `DocumentRepresentation`: convert `pix.samples` directly to a BGR numpy array instead of `pix.tobytes("png")` + `cv2.imdecode`, in both `_load_pdf` and `_render_pdf_page` (bit-exact output).
- `DocumentRepresentation`: make `pdf_text_data`/`pdf_text_dict`/`pdf_text_words`/`pdf_text_blocks` lazily extracted cached properties; a pixel-only comparison no longer pays for text-dict extraction.
- `VisualTest.count_real_difference_pixels`: uint8 `cv2.subtract`-based morphology instead of int64 `.astype(int)` arithmetic (identical result).
- `VisualTest.compare_images`: copy `absolute_diff`/`combined_diff` into `detected_differences` only when an LLM was requested.

No breaking changes: outputs (rendered images, comparison verdicts, sidecars, logs) are identical; keyword signatures and defaults untouched.

## Capabilities

### New Capabilities
- `comparison-performance`: performance contract of the comparison engine — direct pixmap conversion with identical rendering output, on-demand PDF text extraction, and a benchmark harness guarding the render/compare hot paths.

### Modified Capabilities
<!-- none — no existing spec's requirements change; all outputs are identical -->

## Impact

- Code: `DocTest/DocumentRepresentation.py`, `DocTest/VisualTest.py`; new `utest/benchmarks/` (or marker-guarded benchmark module); `pyproject.toml` dev group gains `pytest-benchmark`.
- APIs: none. Page attributes `pdf_text_*` remain readable (and writable by tests) with unchanged values — laziness is internal.
- Users: faster comparisons, lower peak memory; no behavioral difference.
