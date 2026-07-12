# Spec delta: web-visual-testing (web-vrt-examples)

## ADDED Requirements

### Requirement: Executable example coverage
The repository SHALL ship an example page gallery and suites exercising the web
visual testing features end to end — dynamic content masking, DOM-assisted
acceptance, canvas/SVG blind-spot behavior, per-viewport baselines, cross-render
tolerance — executed in CI.

#### Scenario: Edge cases stay covered
- **WHEN** the web CI job runs
- **THEN** the example-driven acceptance suite verifies every reliability feature against the gallery pages

#### Scenario: Canvas blind spot documented and tested
- **WHEN** only canvas pixels change on the chart page
- **THEN** the DOM verdict is identical yet the visual comparison still fails

### Requirement: Capture stability
Captures SHALL be repeated until two consecutive captures are identical (bounded
retries) before being used as baseline or candidate.

#### Scenario: Half-settled layout never becomes the baseline
- **WHEN** the first capture of a page differs from an immediate re-capture
- **THEN** the settled capture is used and subsequent comparisons in the same session pass
