## ADDED Requirements

### Requirement: Logged screenshots are never upscaled
Screenshots embedded in the Robot Framework log SHALL be constrained to at most half the log column width without being enlarged beyond their natural size. An image narrower than half the column SHALL render at its natural size; an image at least half the column wide SHALL render exactly as before (bounded to half the column). The dedicated natural-size mode (`original_size=True`) SHALL remain unbounded.

#### Scenario: Small crop renders at natural size
- **WHEN** a movement-tolerance failure logs a small diff-area crop
- **THEN** the emitted `<img>` style constrains the maximum width rather than setting a fixed width, so the crop is not upscaled

#### Scenario: Full-page screenshot is unchanged
- **WHEN** a full-page rendering or combined diff image is logged
- **THEN** it is still bounded to half the log column width

#### Scenario: Natural-size mode still unbounded
- **WHEN** a screenshot is logged with `original_size=True` (template screenshots)
- **THEN** its style still imposes no width bound
