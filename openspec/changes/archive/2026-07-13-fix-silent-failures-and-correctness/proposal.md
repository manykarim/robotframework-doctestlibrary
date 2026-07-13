# Proposal: fix-silent-failures-and-correctness

## Why

An ultradeep analysis (docs/ultradeep-analysis-solution-proposal.md, §2 and §3.1) found nine verified correctness defects in the comparison engine, three of which can silently convert a genuinely failing comparison into a PASS — the worst failure class for a testing library. All are fixable without touching keyword signatures, defaults, or import paths.

## What Changes

- `Compare Pdf Documents`: coerce the `llm`, `llm_enabled`, `use_llm`, and `llm_override` kwargs through `_as_bool` so the Robot string `False` no longer enables the LLM or lets an LLM verdict override a real failure (PdfTest.py:251-256).
- `Compare Images`: wire up the documented-but-dead `contains_barcodes` argument so barcode areas are detected, excluded from pixel comparison, and content-checked (VisualTest.py:262).
- Block-based SSIM: remove the `abs()` that hides strongly negative (inverted-content) block scores; report sane scores for negative SSIM (DocumentRepresentation.py:557, 482).
- `Compare Print Jobs`: raise a clear `ValueError` for unsupported/unknown print-job types instead of `UnboundLocalError`; treat a property missing from the candidate as a difference instead of crashing with `TypeError` (PrintJobTests.py:354-376).
- EAST OCR: fix the swapped width/height return of `_resize_image` so non-square detector sizes work (Ocr.py:135).
- `Compare Images`: stop the watermark loops from shadowing the user's `mask` argument so the JSON sidecar records the real mask definition (VisualTest.py:636, 678, 1135) — restores the existing `result-sidecar` requirement.
- Close `fitz` documents deterministically in `_load_pdf` and the three PdfTest text keywords (try/finally or context manager).
- PCL/PS loaders: use integer page numbers so page-scoped ignore areas apply (DocumentRepresentation.py:1210, 1267).
- Downloader hardening: re-validate scheme (and drop `ftp://` from defaults) on redirect hops, enforce `max_size` while streaming instead of after download, clean up temp files on failure (Downloader.py:59-118).
- One regression test per fix in `utest/`.

No breaking changes: all keyword names, argument names, argument defaults, and import paths are unchanged; behavior changes only where current behavior contradicts documented behavior.

## Capabilities

### New Capabilities
- `comparison-correctness`: correctness guarantees of the comparison keywords — boolean option coercion for LLM flags, active barcode masking, block-SSIM sensitivity to inverted content, explicit print-job type/property errors, page-scoped masks for all document types, deterministic document-handle release, and hardened reference downloading.

### Modified Capabilities
<!-- none — the sidecar mask fix restores behavior the existing result-sidecar spec already requires -->

## Impact

- Code: `DocTest/PdfTest.py`, `DocTest/VisualTest.py`, `DocTest/DocumentRepresentation.py`, `DocTest/PrintJobTests.py`, `DocTest/Ocr.py`, `DocTest/Downloader.py`; new tests in `utest/`.
- APIs: none removed or renamed; `contains_barcodes` starts doing what its docs promise; string `llm=False` now disables the LLM (previously enabled it — bug-direction change).
- Dependencies/systems: none.
- Users relying on the broken behaviors (none plausibly are: crashes, no-ops, or false PASSes) would see corrected results; release notes will call out each fix.
