# Spec delta: web-visual-testing (selector-ignore-masks)

## ADDED Requirements

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
