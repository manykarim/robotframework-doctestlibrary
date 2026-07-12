# Spec delta: web-visual-testing (pixel-diff-tolerance)

## ADDED Requirements

### Requirement: Pixel-difference tolerance
Visual comparison SHALL accept `max_diff_pixels` and/or `max_diff_ratio` budgets:
a page failing structural comparison passes when the count of pixels whose
absolute difference exceeds `pixel_intensity_threshold` stays within every given
budget. Dimension-mismatched pages SHALL never be rescued.

#### Scenario: Anti-aliasing noise within budget
- **WHEN** 50 pixels differ and `max_diff_pixels=100` is given
- **THEN** the comparison passes and the accepted pixel count is logged

#### Scenario: Real change exceeds budget
- **WHEN** more pixels differ than the budget allows
- **THEN** the comparison fails exactly as without the option

#### Scenario: Faint noise below intensity threshold
- **WHEN** pixels differ by less than `pixel_intensity_threshold`
- **THEN** they consume no budget
