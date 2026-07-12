# Proposal: web-vrt-examples

## Why

Adoption needs runnable proof. There is no place a user can see the whole
web-VRT feature set working end to end — dynamic content, responsive layouts,
form states, canvas/SVG limits, long pages — and the edge cases are exactly where
trust is won or lost.

## What Changes

- **Example page gallery** (`examples/web-visual/pages/`): an app-like dashboard
  (live clock + rotating ad + avatar → ignore_elements), a form page (values &
  validation states → semantic DOM changes), a responsive product grid
  (per-viewport baselines), an SVG+canvas chart page (the documented DOM-analysis
  blind spot), and a long typographic article (full-page capture, font AA).
- **Runnable example suite** (`examples/web-visual/web_visual_demo.robot`) users
  can execute directly against the gallery with Browser Library.
- **End-to-end acceptance** (`atest/web/examples_vrt.robot`) driving the gallery
  through every reliability feature: baseline lifecycle, ignore_elements,
  dom_analysis + accept_rendering_only, canvas blind-spot still caught by pixels,
  per-viewport split baselines, anti-aliasing tolerance, pixel budgets — the
  scenarios and edge cases as executable tests in CI.
- Docs: "Examples" section in the web guide pointing at the gallery.
- **Capture-stability fix** (defect found by the gallery): consecutive full-page
  captures of the same static page can differ in height (scrollbar-induced
  reflow during capture). Captures now repeat until two consecutive ones are
  byte-identical (bounded), so baselines are never created from a half-settled
  layout.

## Capabilities

### Modified Capabilities

- `web-visual-testing`: executable example coverage requirement.

## Impact

`examples/web-visual/` (new), `atest/web/examples_vrt.robot` (new), docs. No
library code changes.
