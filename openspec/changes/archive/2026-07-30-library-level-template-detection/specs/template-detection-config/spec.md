# Spec: template-detection-config

## ADDED Requirements

### Requirement: Library-level template detection setting
`VisualTest` SHALL accept a `template_detection` import argument and provide a `Set Template Detection` keyword, both selecting the default detection method used by `Image Should Contain Template`. Supported values are `template`, `orb` and `sift`, with `classic` accepted as an alias for `template`. Values SHALL be case-insensitive. An unsupported value SHALL raise an error naming the offending value and listing the supported values.

#### Scenario: Import argument sets the default for every call
- **WHEN** the library is imported with `template_detection=sift` and `Image Should Contain Template` is called without a `detection` argument
- **THEN** SIFT-based detection is used

#### Scenario: Keyword changes the setting mid-suite
- **WHEN** `Set Template Detection    sift` runs and a later `Image Should Contain Template` call omits `detection`
- **THEN** SIFT-based detection is used for that call

#### Scenario: Unsupported value is rejected
- **WHEN** the library is imported with `template_detection=bogus`
- **THEN** an error is raised naming `bogus` and listing `orb`, `sift`, `template`

#### Scenario: Case and alias handling
- **WHEN** `template_detection=SIFT` or `template_detection=classic` is given
- **THEN** they resolve to `sift` and `template` respectively

### Requirement: Detection resolution order preserves existing behavior
The detection method for `Image Should Contain Template` SHALL resolve as: the explicit `detection` call argument, else the library-level `template_detection`, else `template`. An omitted or empty `detection` argument SHALL be treated as "not provided". A suite that sets neither SHALL behave exactly as before this change.

#### Scenario: Call argument wins over library setting
- **WHEN** the library sets `template_detection=sift` but a call passes `detection=template`
- **THEN** template matching is used for that call

#### Scenario: Unchanged default for existing suites
- **WHEN** neither `template_detection` nor `detection` is provided
- **THEN** template matching is used, as before

### Requirement: Movement detection is configured independently
The library SHALL keep `movement_detection` and `template_detection` as separate settings, because `movement_detection` additionally supports `text`, which is not a valid template-matching method. Setting `movement_detection` SHALL NOT change the method used by `Image Should Contain Template`.

#### Scenario: movement_detection does not leak into template matching
- **WHEN** the library is imported with `movement_detection=text` and `Image Should Contain Template` is called without `detection`
- **THEN** template matching is used and no error about `text` is raised

### Requirement: Documented movement detection aliases are accepted at import
The `movement_detection` import argument SHALL accept every documented value including the `classic` alias, resolving it to `template`. Type conversion SHALL NOT reject documented aliases before the library validates them.

#### Scenario: classic alias works at import
- **WHEN** a suite imports the library with `movement_detection=classic`
- **THEN** the import succeeds and the effective movement detection method is `template`
