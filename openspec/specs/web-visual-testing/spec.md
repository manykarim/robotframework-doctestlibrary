# web-visual-testing Specification

## Purpose
TBD - created by archiving change web-visual-testing. Update Purpose after archive.
## Requirements
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

### Requirement: Selector-based ignore masks
The web comparison keywords SHALL accept `ignore_elements` locators whose live
bounding boxes (all matches, CSS pixels scaled to capture pixels, padded) are
applied as coordinate ignore masks at comparison time, merged with any
user-provided masks, and re-read on every stabilization recapture. Locators with
no matches SHALL be skipped without failing. Baseline images SHALL remain
unmasked.

#### Scenario: Dynamic region ignored on page comparison
- **WHEN** `Compare Page To Baseline    home    ignore_elements=id=clock` runs against a page whose clock changed
- **THEN** the comparison passes and the clock region is recorded as a mask

#### Scenario: Ignore rect translated for element comparison
- **WHEN** `Compare Element To Baseline` is used with `ignore_elements` inside the captured element
- **THEN** the mask is applied at the element-relative position

#### Scenario: Vanished dynamic element
- **WHEN** an `ignore_elements` locator matches nothing
- **THEN** the keyword proceeds without that mask and logs the fact

### Requirement: Capture context in results
Web comparisons SHALL record the capture context (library, browser, viewport,
device pixel ratio, URL — best effort) in the result sidecar.

#### Scenario: Context written
- **WHEN** a web comparison runs with `result_json` enabled
- **THEN** the sidecar's `context` object names the capturing library and configuration

### Requirement: Split baselines by configuration
`split_baselines_by=browser,viewport` SHALL give each configuration its own
baseline path segment and a config-qualified sidecar label; unknown split keys
SHALL fail at import time.

#### Scenario: Per-browser baselines
- **WHEN** the same comparison runs under two browsers with `split_baselines_by=browser`
- **THEN** each browser reads and writes its own baseline file and history stays separate

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

### Requirement: Optional AI-assisted review
Web comparisons SHALL support the library's optional LLM assist (`llm`,
`llm_override`, `llm_prompt` options) and SHALL provide the model with the
capture context — browser, viewport, URL and the DOM-analysis verdict when
available — alongside the visual evidence. Absence of the [ai] extra SHALL
degrade gracefully (warning, normal comparison result).

#### Scenario: AI approves a non-important web change
- **WHEN** `Compare Page To Baseline    home    llm=True    llm_override=True` fails visually and the model approves
- **THEN** the keyword passes and the LLM decision is recorded in the sidecar

#### Scenario: Model sees the web context
- **WHEN** the LLM assist runs for a web comparison
- **THEN** its input includes the capture context and DOM verdict

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

