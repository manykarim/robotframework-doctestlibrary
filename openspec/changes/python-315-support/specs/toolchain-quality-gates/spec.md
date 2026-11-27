# Spec Delta

## MODIFIED Requirements

### Requirement: Test execution is parallel and hang-proof
The unit test suite SHALL run under pytest-xdist (`-n auto`) with a per-test timeout so a hung external process cannot block CI until the job-level timeout. Tests SHALL be isolated from each other under parallel execution: a test SHALL NOT write into the shared, version-controlled test-data directories; generated files go to per-test temporary locations. Tests SHALL be platform-portable: the unit suite SHALL pass on both Windows and Linux.

#### Scenario: Hung test aborts
- **WHEN** a test exceeds the per-test timeout
- **THEN** that test fails with a timeout error and the run continues/finishes

#### Scenario: Parallel tests do not share generated files
- **WHEN** the unit suite runs with `-n auto` and several tests generate derived images (e.g. upscaled inputs)
- **THEN** each test writes and reads its own temporary copy, and no test fails because another worker overwrote or partially wrote that file

#### Scenario: Test run leaves the checkout clean
- **WHEN** the full unit suite completes
- **THEN** no new untracked files exist under `utest/testdata`

#### Scenario: Unit suite passes on Windows
- **WHEN** the full unit suite runs on Windows with Tesseract installed
- **THEN** it passes; in particular, tests that generate Robot Framework suites embed file paths in a form Robot does not interpret as escape sequences

## ADDED Requirements

### Requirement: First-party code is free of interpreter-level deprecations
On every supported Python version, the shipped packages SHALL import and run without SyntaxWarnings (e.g. invalid escape sequences in string or grammar literals) and without deprecation warnings caused by first-party use of a deprecated third-party API for which a replacement is available.

#### Scenario: No SyntaxWarning on the newest interpreter
- **WHEN** the PostScript and PCL print-job grammars are built and every module is compiled under Python 3.15 with SyntaxWarnings treated as errors
- **THEN** no error is raised

#### Scenario: Current PDF API is used
- **WHEN** the library is imported with PyMuPDF 1.28 or newer
- **THEN** no "`fitz` API is deprecated" warning is emitted

### Requirement: Repository checks out cleanly on case-insensitive file systems
The repository SHALL NOT track two paths that differ only by letter case, so a checkout on Windows or macOS is clean and generated files map to exactly one tracked path.

#### Scenario: Case-only duplicate is rejected
- **WHEN** a change adds a path that differs from a tracked path only by case (e.g. `docs/AI.html` next to `docs/Ai.html`)
- **THEN** the unit suite fails and names both paths
