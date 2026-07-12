# Spec delta: web-visual-testing (dom-assisted-comparison)

## ADDED Requirements

### Requirement: DOM-assisted classification
With `dom_analysis` enabled, web comparisons SHALL store a semantic DOM snapshot
beside the baseline, diff the live snapshot against it before comparing pixels,
and record the verdict (identical / changed with summary / missing-baseline) in
the sidecar context. Snapshots SHALL refresh on every visual pass.

#### Scenario: CSS-only change classified rendering-only
- **WHEN** only styling changed and the visual comparison fails
- **THEN** the sidecar context reports the DOM verdict `identical`

### Requirement: Accept rendering-only differences
The keywords SHALL pass a failing visual comparison with a warning when
`accept_rendering_only` is enabled and the DOM verdict is `identical`, while the
sidecar keeps the failure for review; any DOM change SHALL keep the test failing.

#### Scenario: Non-important change accepted
- **WHEN** a color-only change fails pixel comparison with both options enabled
- **THEN** the test passes with a warning and the failure stays reviewable

#### Scenario: Semantic change never auto-accepted
- **WHEN** text content changed
- **THEN** the comparison fails despite `accept_rendering_only`
