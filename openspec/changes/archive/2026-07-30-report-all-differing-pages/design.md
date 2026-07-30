# Design: report-all-differing-pages

## Context

`compare_images` collects differences into `detected_differences` while iterating page pairs, then ends with:

```python
for diff in detected_differences:
    robot_logger.info(diff["message"])
    self._raise_comparison_failure()
```

The raise is inside the loop, so iteration stops after the first entry. Everything upstream is already correct: all pages are compared, screenshots for every differing page are emitted inside the page loop, and the JSON sidecar records a per-page entry for every page.

The entries are heterogeneous:
- SSIM and dimension-mismatch entries carry `ref_page`/`cand_page` and a message with **no** page number.
- Barcode entries carry `page` (an int) and no `ref_page`, and their message **already** reads `The barcodes on page N differ: ...`.

## Goals / Non-Goals

**Goals:** report every differing page; name the page in each message exactly once; keep the failure text and every pass/fail outcome identical.

**Non-Goals:**
- Changing the `AssertionError` text. 39+ exact-string assertions depend on it (`atest/`, `utest/`, and generated suites in `e2e/e2e_helpers.py`, `e2e/test_journeys.py`), and Robot's report shows only that text.
- Changing the page-count-mismatch behavior. `iter_page_pairs` raises `ValueError("Documents have different number of pages.")` before any page is compared, so documents of unequal length still compare zero pages. That is a different (and more invasive) change than the reported one.
- Revisiting `check_text_content=True` passing when text matches. The issue's second remark touches this, but "differences accepted because the text is identical" is the documented purpose of that option.

## Decisions

1. **Log all, then raise once.** `for ...: log(...)` followed by `if detected_differences: self._raise_comparison_failure()`. The `if` guard is essential: `detected_differences` is deliberately cleared when an LLM override approves the differences or a reference run promotes the candidate, and in those cases execution must continue to the existing `Images/Document comparison passed.` line.
2. **Page prefix only where it is missing.** Prefix `Page {n}: ` when the entry has a `ref_page`; leave entries without one (barcode) untouched, since their message already names the page. This avoids the "Page 1: The barcodes on page 1 differ" double label without editing the barcode message — which also flows into the sidecar `notes` and the LLM payload and must stay as it is.
3. **Leading summary, not trailing.** The summary is emitted before the per-difference lines so a large multi-page failure is scannable from the top.
4. **Read `page_number` after the loop is safe.** In streaming mode pages are released as iteration advances, but `Page.release_resources()` clears image/text/barcode data and never touches `page_number`, and `detected_differences` holds strong references to the `Page` objects.
5. **New fixtures, built in the test.** No committed fixture pair differs on more than one page, so the regression test builds 3-page PDFs with PyMuPDF in a temp directory.
6. **Assert on `robot_logger`, not `caplog`.** `robot_logger` is `robot.api.logger`; outside a Robot run it routes to the Python logger named `RobotFramework`, not `DocTest.VisualTest`, so `caplog` on the module logger captures nothing.

## Risks / Trade-offs

- [Log volume grows with the number of differing pages, amplified by `WebVisualTest`'s retry loop, which swallows the failure and retries until its timeout] → INFO level, one line per differing page; the leading summary keeps it navigable and outcomes are unchanged.
- [Users may still want page numbers in the failure message itself, which Robot's report shows] → Deliberately out of scope given the exact-match assertions; an additive, default-off argument could offer it later without breaking the baseline.

## Migration Plan

Single patch release; log output only. Rollback = revert.

## Open Questions

None.
