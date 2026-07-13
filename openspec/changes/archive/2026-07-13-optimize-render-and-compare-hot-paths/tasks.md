# Tasks: optimize-render-and-compare-hot-paths

## 1. Benchmark harness first

- [x] 1.1 Add `pytest-benchmark` to the dev dependency group and create `utest/benchmarks/test_benchmarks.py` with canonical benchmarks: PDF document load (sample.pdf @ 200 DPI) and single-page visual comparison; record baseline numbers

## 2. Rendering

- [x] 2.1 Replace the PNG round-trip with direct `pix.samples` conversion in `_load_pdf` and `_render_pdf_page` (handle n=1/3/4 channels and non-contiguous stride)
- [x] 2.2 Equivalence test: direct conversion is array-equal to the PNG round-trip for a real multi-page PDF at the default DPI

## 3. Lazy text extraction

- [x] 3.1 Convert `pdf_text_data`, `pdf_text_dict`, `pdf_text_blocks` to lazy cached properties on `Page` (with working setters); keep `pdf_text_words` eager; add a re-open-by-path fallback for access after document close
- [x] 3.2 Tests: pixel-only compare never triggers dict extraction (spy); lazy values equal eager values including after `close()`

## 4. Compare-loop micro-optimizations

- [x] 4.1 `count_real_difference_pixels`: uint8 `cv2.subtract(dilate, erode)`; equivalence test against the int64 implementation on real diff data
- [x] 4.2 Copy `absolute_diff`/`combined_diff` into `detected_differences` only when `llm_requested`; verify the LLM payload path still receives images

## 5. Verification

- [x] 5.1 Run benchmarks before/after and record the improvement in the change notes
- [x] 5.2 Full unit suite green (`uv run pytest utest`) — 789 passed, 3 skipped

## Measured results (task 5.1)

pytest-benchmark means on this machine, before → after:

- PDF document load (sample.pdf @ 200 DPI): 1518 ms → 435 ms (3.5×)
- Visual compare, identical 1-page PDFs: 625 ms → 209 ms (3.0×)
- Visual compare with movement detection: 1348 ms → 948 ms (1.4×)
