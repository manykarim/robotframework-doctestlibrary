# Proposal: antialiasing-tolerance

## Why

Cross-browser and cross-OS font rendering makes ~1% of pixels differ (measured:
chromium vs firefox on a simple page → 0.90% of pixels differ by >20 intensity).
Raw pixel budgets large enough to absorb that would also hide real regressions.
The verified edge-based classifier separates the two: 5065 of 5070 differing
pixels in the cross-browser pair are anti-aliasing, while added elements (0%
misclassified) and text changes (439 residual pixels) stay caught.

## What Changes

- `Compare Images` (and all WebVisualTest keywords) accepts
  `ignore_antialiasing=True`: differing pixels that sit on a local intensity edge
  in BOTH images (pixelmatch-style 3×3 heuristic) are classified as anti-aliasing
  and do not count toward failure or pixel budgets.
- Alone, `ignore_antialiasing` implies a zero budget for the remaining "real"
  pixels; combined with `max_diff_pixels`/`max_diff_ratio` the budgets apply to
  the real pixels only. The INFO log reports both counts.
- Cross-browser acceptance test: a chromium-created baseline compared against a
  firefox capture passes with `ignore_antialiasing=True` and fails without it;
  the CI web job initializes firefox in addition to chromium.

## Capabilities

### Modified Capabilities

- `web-visual-testing`: anti-aliasing tolerance requirement.

## Impact

`DocTest/VisualTest.py` (classifier + option), `utest/test_pixel_tolerance.py`,
`atest/web/xbrowser_vrt.robot` (new), `.github/workflows/ci.yml` (firefox), docs.
