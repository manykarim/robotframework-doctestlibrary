# Tasks: fix-silent-failures-and-correctness

## 1. False-PASS fixes

- [x] 1.1 PdfTest: route `llm`, `llm_enabled`, `use_llm`, `llm_override` kwargs through `_as_bool` (PdfTest.py:251-256); regression test that string `"False"` disables the LLM path and string `"True"` enables it
- [x] 1.2 VisualTest: forward `contains_barcodes` into both `DocumentRepresentation` constructions, exclude detected barcode areas from pixel comparison, compare decoded barcode values and fail on mismatch; regression tests for same-content jitter pass and different-content fail
- [x] 1.3 DocumentRepresentation: remove `abs()` from block-SSIM lowest-score tracking (:557) and clamp the reported score for negative SSIM (:482); regression test with an inverted block that must fail

## 2. Crash-to-clear-error fixes

- [x] 2.1 PrintJobTests: `else: raise ValueError(...)` for unsupported print-job types in `compare_print_jobs`; regression test for `afp` and a typo type
- [x] 2.2 PrintJobTests: handle `None` counterpart in `compare_properties` as a recorded difference; regression test with asymmetric property sets
- [x] 2.3 Ocr: fix `_resize_image` return order to `(image, width, height, ratio_w, ratio_h)` (:135); regression test with non-square dimensions

## 3. Metadata and resource fixes

- [x] 3.1 VisualTest: rename watermark loop variables so the `mask` argument is not shadowed (:636, :678); regression test that sidecar `masks.mask` equals the supplied mask when watermark + result_json + diffs coincide
- [x] 3.2 DocumentRepresentation: close the fitz document in `_load_pdf` via context manager; PdfTest: `try/finally: doc.close()` in `check_text_content`, `PDF_should_contain_strings`, `PDF_should_not_contain_strings`; regression test asserting handles are closed
- [x] 3.3 DocumentRepresentation: integer `page_number` in `_load_pcl` and `_load_ps` (:1210, :1267); regression test that a page-scoped mask applies to a PCL/PS-style page

## 4. Downloader hardening

- [x] 4.1 Replace `urlretrieve` with an opener whose redirect handler re-validates schemes per hop; drop `ftp` from `ALLOWED_SCHEMES`; regression test that a redirect to a disallowed scheme is rejected
- [x] 4.2 Stream downloads with early abort at `max_size` and unlink partial temp files; regression test with a local HTTP server serving an oversized body

## 5. Verification

- [x] 5.1 Run the full unit suite (`uv run pytest utest`) and confirm no regressions
- [x] 5.2 Confirm keyword signatures/defaults unchanged (libdoc spot-check of `Compare Images`, `Compare Pdf Documents`, `Compare Print Jobs`)
