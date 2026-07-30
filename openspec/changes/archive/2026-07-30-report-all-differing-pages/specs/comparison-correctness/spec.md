## ADDED Requirements

### Requirement: Every differing page is reported
When a multi-page comparison detects differences on more than one page, the library SHALL log a message for every detected difference before failing, not only the first. Each logged difference SHALL identify its page exactly once, and the comparison SHALL still fail with the unchanged message `The compared images are different.`

#### Scenario: Differences on several pages are all reported
- **WHEN** `Compare Images` compares a 3-page pair that differs on pages 1 and 3
- **THEN** the log contains a difference message for page 1 and for page 3, and the keyword fails with `The compared images are different.`

#### Scenario: Summary names the affected pages
- **WHEN** a multi-page comparison fails on more than one page
- **THEN** the log contains a leading summary line naming every affected page number

#### Scenario: Pages are not labelled twice
- **WHEN** a difference message already names its page (barcode content differences)
- **THEN** that message is logged unchanged, without an additional page prefix

#### Scenario: Single-page failure is unchanged in outcome
- **WHEN** a single-page comparison fails
- **THEN** the keyword still fails with `The compared images are different.`
