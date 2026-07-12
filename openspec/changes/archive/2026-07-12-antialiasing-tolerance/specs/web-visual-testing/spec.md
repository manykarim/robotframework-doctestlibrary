# Spec delta: web-visual-testing (antialiasing-tolerance)

## ADDED Requirements

### Requirement: Anti-aliasing tolerance
Visual comparison SHALL accept `ignore_antialiasing`: differing pixels lying on a
local intensity edge in both images are excluded from failure decisions and pixel
budgets; remaining pixels must stay within the given budgets (zero when none are
given).

#### Scenario: Cross-browser rendering passes
- **WHEN** a firefox capture is compared against a chromium-created baseline with `ignore_antialiasing=True`
- **THEN** the comparison passes although thousands of edge pixels differ

#### Scenario: Content change still fails
- **WHEN** an element or text actually changed
- **THEN** non-edge difference pixels remain and the comparison fails
