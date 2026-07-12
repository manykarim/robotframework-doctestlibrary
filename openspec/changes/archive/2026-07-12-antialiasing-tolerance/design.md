# Design: antialiasing-tolerance

## Decisions

- **D1 Classifier** (validated by experiment): differing pixel (grayscale
  absdiff > `pixel_intensity_threshold`) is anti-aliasing iff the 3×3 local
  intensity range (dilate − erode) exceeds 60 in BOTH images — i.e. the pixel
  lies on a rendered edge in each. Vectorized with cv2 morphology; zero cost when
  the option is off.
- **D2 Semantics**: `ignore_antialiasing=True` → real_pixels = differing − AA;
  pass iff real_pixels within budgets; with no explicit budget the implied budget
  is 0 (every differing pixel must be AA). Composes with
  `pixel_intensity_threshold`.
- **D3 Placement**: inside the existing pixel-budget block after `compare_with`;
  grayscale images recomputed from the page images (cheap, only on failing pages
  with the option set). Dimension mismatches are never rescued (absolute_diff is
  None there).
- **D4 Cross-browser proof**: `atest/web/xbrowser_vrt.robot` creates the baseline
  from chromium, then compares a firefox capture of the same page —
  must fail plain and pass with `ignore_antialiasing=True max_diff_pixels=100`.

## Risks

- [Dense small text change classified as AA] → measured residual for a text
  change is ~28% of differing pixels — far above the implied 0 budget; the DOM
  analysis proposal adds the semantic backstop.
