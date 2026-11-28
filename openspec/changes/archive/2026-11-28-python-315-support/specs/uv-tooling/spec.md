# Spec Delta

## MODIFIED Requirements

### Requirement: Multi-Python validation
The toolchain SHALL validate Python 3.11 through 3.15. For each interpreter, `uv sync --all-extras` succeeds, an automated audit checks that the resolved versions of the marker-controlled packages match the expectation table, and CI runs the test suite. The audit SHALL fail when the running interpreter has no entry in the table. A local `invoke multipython` task SHALL run sync, audit and import smoke for all supported interpreters.

#### Scenario: Dashboard code runs on the oldest supported Python
- **WHEN** the dashboard unit tests run under Python 3.11 with the dashboard extra installed
- **THEN** they pass

#### Scenario: Local multi-Python task
- **WHEN** `invoke multipython` runs on a development machine
- **THEN** every supported interpreter is synced, audited against the expectation table, and import-smoked, with a per-version summary

#### Scenario: Audit covers the newest interpreter
- **WHEN** the audit runs under Python 3.15
- **THEN** it checks the 3.15 expectations (including scikit-image 0.27 and numpy 2.5.2 floors) instead of reporting an unsupported interpreter

### Requirement: CI, publish, and invoke tasks use uv
`tasks.py` SHALL invoke all tooling through `uv run`. `ci.yml` SHALL install environments with uv across the 3.11–3.15 matrix (with extras), run the dashboard/e2e job from the new paths, and run a Windows job on the newest supported interpreter that installs all extras, audits the resolved versions, and runs the unit suite. `python-publish.yml` SHALL build the frontend before `uv build` and run the artifact-parity comparison as a release gate.

#### Scenario: Publish path produces a complete wheel
- **WHEN** the publish workflow runs
- **THEN** the frontend is built first, the wheel contains the static UI, and the parity comparison gate passes before upload

#### Scenario: CI matrix covers extras per interpreter
- **WHEN** the CI matrix job runs for an interpreter version
- **THEN** the environment is created with `uv sync --all-extras` for that interpreter and the unit/acceptance suites run through `uv run`

#### Scenario: Windows wheel availability is gated
- **WHEN** a dependency change makes a package unavailable as a Windows wheel for the newest supported interpreter
- **THEN** the Windows job fails, because `uv sync --all-extras` fails there

#### Scenario: Windows-specific regressions are caught
- **WHEN** a change breaks behavior only on Windows (e.g. path handling or console encoding)
- **THEN** the Windows job's unit suite fails
