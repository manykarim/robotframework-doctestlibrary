# Spec delta: web-visual-testing (web-visual-testing)

## ADDED Requirements

### Requirement: Page and element baseline comparison keywords
`DocTest.WebVisualTest` SHALL provide `Compare Page To Baseline` and
`Compare Element To Baseline` keywords that capture through the running Browser
Library or SeleniumLibrary session (auto-detected) and compare against a named
baseline with all `Compare Images` options available.

#### Scenario: Page comparison with Browser Library
- **WHEN** a suite importing Browser and WebVisualTest calls `Compare Page To Baseline    home`
- **THEN** a full-page CSS-pixel screenshot is compared against `{baseline_directory}/home.png`

#### Scenario: Element comparison with SeleniumLibrary
- **WHEN** a suite importing SeleniumLibrary calls `Compare Element To Baseline    id=header    header`
- **THEN** the element screenshot is compared against the `header` baseline

#### Scenario: No web library loaded
- **WHEN** neither Browser nor SeleniumLibrary is imported
- **THEN** the keyword fails with an error naming both supported libraries

### Requirement: Automatic baseline creation
When the named baseline does not exist, the capture SHALL be stored as the new
baseline and the keyword SHALL pass with a logged warning; when `${REFERENCE_RUN}`
is truthy the capture SHALL overwrite the baseline. Baseline names SHALL be
sanitized so they cannot escape the baseline directory.

#### Scenario: First run creates the baseline
- **WHEN** `Compare Page To Baseline    new-page` runs and no baseline exists
- **THEN** the capture is saved as the baseline and the test passes with a warning

#### Scenario: Traversal attempt
- **WHEN** a baseline name contains path separators or `..`
- **THEN** the stored baseline still resolves inside the baseline directory

### Requirement: Stabilization retry
On comparison failure the keyword SHALL recapture and recompare until
`retry_timeout` expires (default 3 seconds, disable with 0), re-raising the last
failure afterwards.

#### Scenario: Late-settling page passes
- **WHEN** the page reaches its final state during the retry window
- **THEN** a later recapture matches and the keyword passes

#### Scenario: Real difference still fails
- **WHEN** the page differs from the baseline permanently
- **THEN** the keyword fails after the retry window with the comparison failure
