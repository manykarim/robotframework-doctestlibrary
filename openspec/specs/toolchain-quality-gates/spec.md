# Spec: toolchain-quality-gates

## Purpose
Repository quality gates: lint/format/type checks, parallel and timeout-guarded tests, keyword-signature compatibility snapshot, dependency hygiene, and CI that tests the checked-out code.
## Requirements
### Requirement: Keyword-signature compatibility snapshot
The repository SHALL contain a libdoc-derived snapshot of every public keyword's name, argument names, and argument defaults for `DocTest.VisualTest`, `DocTest.PdfTest`, `DocTest.PrintJobTests`, and `DocTest.WebVisualTest`, and a test that fails when a keyword or argument is removed, renamed, or has its default changed. Additive changes (new keywords, new trailing arguments with defaults) SHALL NOT fail the test.

#### Scenario: Non-additive change is caught
- **WHEN** a keyword argument's default value is changed in the source
- **THEN** the signature snapshot test fails, naming the keyword and the difference

#### Scenario: Additive change passes
- **WHEN** a new keyword is added to a library
- **THEN** the signature snapshot test passes (with a note to refresh the baseline)

### Requirement: Lint, format, and type gates
The repository SHALL provide ruff lint + format checks and a mypy configuration covering `DocTest/`, runnable locally (pre-commit) and enforced in CI. The gates MUST pass on the current codebase.

#### Scenario: CI runs the gates
- **WHEN** a pull request is opened
- **THEN** CI runs ruff check, ruff format --check, and mypy as a required job

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

### Requirement: Dependency hygiene
The runtime dependency set SHALL NOT contain packages that are never imported. Every runtime dependency SHALL declare a tested lower version bound. Barcode (`pylibdmtx`, `pyzbar`) and print-job (`parsimonious`) dependencies SHALL additionally be exposed as optional extras `barcode` and `printjobs` while remaining in the base set until a documented future release removes them.

#### Scenario: Extras install the niche dependencies
- **WHEN** a user installs `robotframework-doctestlibrary[barcode]`
- **THEN** pylibdmtx and pyzbar are installed

#### Scenario: No dead dependencies
- **WHEN** the dependency audit runs
- **THEN** every base dependency is imported somewhere in the shipped packages or explicitly documented as a system-integration requirement

### Requirement: CI tests the checked-out code
Continuous integration SHALL build and test the code from the pull request checkout, not from a released package. Live-LLM tests SHALL NOT be part of the required matrix for every interpreter; they run as a separate non-required (or scheduled) job on a single interpreter.

#### Scenario: Docker job uses the checkout
- **WHEN** the Docker CI job builds its image
- **THEN** the library inside the image is installed from the repository checkout

#### Scenario: PR without LLM secrets still gets full required signal
- **WHEN** a pull request runs CI without LLM credentials
- **THEN** all required jobs can pass; only the optional LLM job is skipped

### Requirement: Automated dependency and vulnerability monitoring
The repository SHALL have Dependabot configured for Python dependencies and GitHub Actions, and CI SHALL include a dependency vulnerability audit step (pip-audit).

#### Scenario: Vulnerable dependency is flagged
- **WHEN** a dependency with a known CVE is present in the lockfile
- **THEN** the audit step reports it

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
