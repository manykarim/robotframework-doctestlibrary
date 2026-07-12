# Design: pixel-diff-tolerance

## Decisions

- **D1 Options**: popped from `compare_images` kwargs (`max_diff_pixels`,
  `max_diff_ratio`, `pixel_intensity_threshold`), converted with `int()`/`float()`
  (RF passes `**kwargs` values as strings). No import-level defaults — this is a
  per-comparison decision; web keywords inherit via passthrough.
- **D2 Counting**: differing pixels = `np.count_nonzero(absolute_diff > pixel_intensity_threshold)`
  on the grayscale absolute diff `compare_with` already returns — zero extra
  computation. Ratio = count / absolute_diff.size.
- **D3 Placement**: immediately after `compare_with` in the page loop, before
  watermark handling: `if not similar and budget given and absolute_diff is not
  None` → rescue when within budget (`similar = True`, INFO log with count/total).
  `absolute_diff is None` covers the dimension-mismatch and identical shortcuts.
- **D4 Both given**: both limits must hold (AND) — the stricter interpretation is
  the safe one.

## Risks

- [Budget hides real one-line change] → counting uses intensity > threshold so a
  visible change of any size consumes budget; documentation recommends small
  budgets (≤ 0.1%).
