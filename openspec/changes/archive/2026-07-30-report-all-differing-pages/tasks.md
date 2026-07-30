# Tasks: report-all-differing-pages

## 1. Fix reporting

- [x] 1.1 `DocTest/VisualTest.py`: log every entry in `detected_differences`, then raise once via `if detected_differences: self._raise_comparison_failure()` (guard required so LLM-override and reference-run paths still reach the "comparison passed" line)
- [x] 1.2 Emit a leading summary line naming the affected page numbers when differences were found
- [x] 1.3 Prefix each logged message with `Page {n}: ` only for entries carrying a `ref_page`; leave barcode messages (which already name their page) unchanged

## 2. Regression tests

- [x] 2.1 Add `utest/test_multipage_reporting.py` building 3-page PDF fixtures with PyMuPDF that differ on pages 1 and 3; assert a message is logged for both pages and the failure text is still `The compared images are different.`
- [x] 2.2 Assert the leading summary names both page numbers
- [x] 2.3 Assert barcode difference messages are not double-labelled
- [x] 2.4 Assert a passing comparison logs no difference messages and still logs the "passed" line

## 3. Verification

- [x] 3.1 Confirm the failure message text is byte-identical (the 39+ exact-match assertions in `atest/`, `utest/` and `e2e/` stay green)
- [x] 3.2 `uv run pytest utest -q -n auto --timeout=300` green; `uvx ruff check DocTest` green
