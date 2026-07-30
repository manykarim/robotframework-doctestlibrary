# Proposal: report-all-differing-pages

## Why

GitHub issue #98 (with a second reporter confirming): comparing a multi-page PDF appears to stop at the first differing page, so users cannot see whether later pages also differ.

Investigation shows the comparison itself is fine — **every page is already compared**. For a 3-page pair differing on pages 1 and 3, both differences are collected and screenshots are emitted for both. The defect is purely in reporting: the final loop in `compare_images` calls `self._raise_comparison_failure()` *inside* the `for diff in detected_differences` loop, so only the first difference is ever logged. Worse, that one message (`Visual differences detected. SSIM score: ...`) does not say which page it refers to, so the log gives no hint that other pages were even examined.

## What Changes

- `compare_images` logs **every** collected difference and then raises **once**, instead of raising on the first one.
- A leading summary line names the affected pages up front (e.g. `Visual differences detected on 2 of 3 page(s): 1, 3`) so a long log stays scannable.
- Per-difference messages are prefixed with their page number. Messages that already name their page — the barcode difference message — are left alone, so nothing is labelled twice.
- The raised `AssertionError` text stays exactly `The compared images are different.` More than 39 assertions across `atest/`, `utest/` and `e2e/` match that string exactly, and Robot's report shows only the exception text; page detail belongs in the log body.

`PdfTest.compare_pdf_documents` was checked and already accumulates into a single flag before raising once — no change needed there.

## Capabilities

### Modified Capabilities
- `comparison-correctness`: adds a requirement that all differing pages are reported, not just the first.

## Impact

- Code: `DocTest/VisualTest.py` — the final reporting block, plus a page prefix on the difference messages that lack one.
- Tests: new `utest/test_multipage_reporting.py` building multi-page fixtures with PyMuPDF (no existing fixture differs on more than one page).
- Users: failing multi-page comparisons now list every differing page. No change to pass/fail outcomes, exception text, sidecar contents (which already record all pages) or keyword signatures.
- `WebVisualTest` retries `compare_images` until its timeout, so a failing web comparison logs proportionally more lines than before; acceptable at INFO level and unchanged in outcome.
