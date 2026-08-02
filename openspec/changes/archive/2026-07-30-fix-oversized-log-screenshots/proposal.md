# Proposal: fix-oversized-log-screenshots

## Why

GitHub issue #140: every screenshot embedded in the Robot Framework log is forced to exactly half the log column width (`style="width:50%; height: auto;"`). Small crops — the moved-area and diff-area images emitted when a `move_tolerance` check fails — are therefore *upscaled*, rendering huge and blurry instead of at their natural size. The reporter proposed `max-width:50%`, which keeps large screenshots exactly as they are today while letting small ones render at their real size.

## What Changes

- `VisualTest.add_screenshot_to_log`: the non-`original_size` style becomes `max-width:50%; height: auto;`. Large images (any image at least half the log column wide — all full-page renderings, combined views and diff images) render byte-identically; only images narrower than half the column change, which is exactly the population issue #140 is about.
- The `original_size=True` branch (`width: auto; height: auto;`) is deliberately left untouched: it exists specifically so template screenshots keep their natural size, and widening it would reverse that earlier fix.
- Add a regression test that pins the emitted `style` attribute for both branches and both emit paths (base64-embedded and the default file-link path).

No behavioral change to comparison logic, no keyword signature change.

## Capabilities

### Modified Capabilities
- `comparison-correctness`: adds a requirement that logged screenshots are never upscaled beyond their natural size.

## Impact

- Code: `DocTest/VisualTest.py` (one line: the non-`original_size` style string).
- Tests: new `utest/test_screenshot_log_style.py`.
- Consumers: `WebVisualTest` inherits `add_screenshot_to_log` unchanged and picks the fix up automatically. The dashboard's legacy log-scraping regex (`doctest_dashboard/ingest.py`) matches on `src=` only and is indifferent to the style attribute.
- Users: logs become readable for small diff crops; nothing else observable changes.
