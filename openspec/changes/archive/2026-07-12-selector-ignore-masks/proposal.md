# Proposal: selector-ignore-masks

## Why

Dynamic page regions (clocks, ads, avatars, feeds) are the #1 source of false
visual diffs. Today users must measure pixel coordinates by hand to mask them.
Every serious VRT tool masks by *selector*; Browser Library's own capture-time
masking burns pink boxes into the screenshot (baseline pollution) and Selenium has
nothing. Comparison-time masks from live element geometry keep baselines clean,
work identically for both libraries, and stay visible in the sidecar/dashboard.

## What Changes

- New `ignore_elements=` option on `Compare Page To Baseline` and
  `Compare Element To Baseline`: one locator, a `;`-separated list, or a Robot
  list. All matching elements' bounding boxes are read from the live session,
  scaled from CSS pixels to capture pixels (Browser `scale=css` → 1:1; Selenium →
  × devicePixelRatio), padded slightly, and passed to the engine as coordinate
  masks — merged with any user-provided `mask=`/`placeholder_file=`.
- For element comparisons the ignore rects are translated into the captured
  element's coordinate space (page rect minus element origin).
- Locators matching nothing contribute no masks (logged), they do not fail the
  keyword — a vanished dynamic element is not an error.
- Tests: unit (rect→mask math via real comparisons of generated images, offset
  translation, merging with user masks) + acceptance tests in both real-browser
  suites (mutate a region, pass with `ignore_elements=`).

## Capabilities

### Modified Capabilities

- `web-visual-testing`: selector-based ignore masks requirement.

## Impact

`DocTest/WebVisualTest.py` (option handling, rect collection/translation),
`DocTest/WebCapture.py` (already provides `element_rects`/`capture_scale`/
`ElementRect.to_mask`), `utest/test_web_visual.py`, `atest/web/*.robot`, docs.
