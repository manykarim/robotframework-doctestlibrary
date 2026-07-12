# Proposal: pixel-diff-tolerance

## Why

Cross-OS font rendering and anti-aliasing produce a handful of slightly-changed
pixels that fail SSIM-based comparison, and the global `threshold` knob is too
blunt — raising it can hide real regressions. Playwright/BackstopJS solve this
with pixel-count acceptance (`maxDiffPixels`/`maxDiffPixelRatio`); the engine
needs the same escape hatch, usable from web keywords and `Compare Images` alike.

## What Changes

- `Compare Images` (and therefore all WebVisualTest keywords) accepts:
  - `max_diff_pixels=` — pass when at most N pixels differ meaningfully,
  - `max_diff_ratio=` — pass when at most this fraction of pixels differs,
  - `pixel_intensity_threshold=` (default 20/255) — how strong a per-pixel
    difference must be to count (sub-threshold anti-aliasing noise is free).
- Applied per page after the SSIM verdict: a page that fails SSIM but stays
  within the pixel budget is accepted with an INFO log naming the counted pixels.
  Pages with different dimensions are never rescued.
- Tests: unit tests with generated images (count boundaries, ratio boundaries,
  intensity threshold, dimension mismatch untouched, string conversion from RF
  kwargs), plus a web acceptance scenario tolerating a tiny mutation.

## Capabilities

### Modified Capabilities

- `web-visual-testing`: pixel-tolerance acceptance requirement (engine-level,
  exposed through `Compare Images` and the web keywords).

## Impact

`DocTest/VisualTest.py` (kwarg parsing + per-page acceptance), `utest/`,
`atest/web/browser_vrt.robot`, docs.
