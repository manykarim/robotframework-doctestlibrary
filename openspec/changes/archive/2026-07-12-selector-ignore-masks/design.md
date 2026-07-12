# Design: selector-ignore-masks

## Decisions

- **D1 Option shape**: `ignore_elements` accepted by both keywords via `**kwargs`
  (popped before compare passthrough). String → split on `;`, trimmed, empties
  dropped; list/tuple → used as-is. Documented example: `ignore_elements=id=clock`
  (named-arg parsing splits only on the first `=`, so locators keep their `=`).
- **D2 Rect collection**: `adapter.element_rects(locator)` per locator; all matches
  contribute. Empty match → `logger.info`, skipped. Rects are CSS px; mask =
  `ElementRect.to_mask(scale=adapter.capture_scale(), padding=2)` — outward
  rounding + 2px padding absorbs sub-pixel rendering at the mask edge.
- **D3 Element-space translation**: for `Compare Element To Baseline` the compared
  element's own first rect is the origin; ignore rects are translated by
  `(-origin.x, -origin.y)` before mask conversion. Rects fully outside the captured
  element produce off-image masks — harmless (engine clips).
- **D4 Merging**: user `mask=` (str/dict/list) is normalized to a list and the
  ignore masks are appended; the merged list goes to `compare_images(mask=...)`.
  `placeholder_file=` is untouched (both can coexist — engine already merges).
- **D5 Retry interaction**: rects are re-read on every recapture attempt (elements
  move between attempts — the mask must follow the current geometry).

## Risks

- [Locator syntax differs between libraries] → locators are passed verbatim to the
  active library; no parsing in DocTest.
- [Fractional rects under-cover] → outward rounding + 2px padding (D2).
