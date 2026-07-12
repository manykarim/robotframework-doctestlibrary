# Spec: uv-tooling

## ADDED Requirements

### Requirement: uv is the single environment and dependency manager
The repository SHALL be fully managed by uv: one PEP 621 `pyproject.toml` at the root, one universal `uv.lock`, dev tooling in `[dependency-groups]`, and no poetry configuration or lockfiles anywhere. `uv sync --all-extras` SHALL produce a complete development environment (library, dashboard, test tooling).

#### Scenario: Fresh clone bootstrap
- **WHEN** a contributor runs `uv sync --all-extras` on a fresh clone
- **THEN** core unit tests, dashboard tests, and acceptance suites are runnable via `uv run` without any other package manager

#### Scenario: No poetry remnants
- **WHEN** the repository is searched for poetry configuration
- **THEN** no `[tool.poetry]` tables, `poetry.lock`, or poetry CI steps remain

### Requirement: Unified repository layout
The dashboard SHALL live as a peer package: `doctest_dashboard/` next to `DocTest/`, dashboard backend tests under `utest/dashboard/`, Playwright journeys under `e2e/`, the frontend under `frontend/`, and the old `dashboard/` directory removed. All existing tests SHALL pass from their new locations.

#### Scenario: Test suites green after the move
- **WHEN** core unit tests, acceptance suites, dashboard tests, and e2e journeys run from the new layout
- **THEN** they all pass with no skips introduced by the restructuring

### Requirement: Multi-Python validation
The toolchain SHALL validate Python 3.9 through 3.13: per-interpreter `uv sync --all-extras` succeeds, an automated audit asserts the resolved versions of the marker-controlled packages match the expectation table, and the test suite runs per interpreter in CI. A local `invoke multipython` task SHALL run the sync + audit + import smoke for all supported interpreters.

#### Scenario: Dashboard code runs on Python 3.9
- **WHEN** the dashboard unit tests run under Python 3.9 with the dashboard extra installed
- **THEN** they pass (no 3.10+-only syntax in `doctest_dashboard`)

#### Scenario: Local multi-Python task
- **WHEN** `invoke multipython` runs on a development machine
- **THEN** every supported interpreter is synced, audited against the expectation table, and import-smoked, with a per-version summary

### Requirement: CI, publish, and invoke tasks use uv
`tasks.py` SHALL invoke all tooling through `uv run`; `ci.yml` SHALL install environments with uv across the 3.9–3.13 matrix (with extras) and run the dashboard/e2e job from the new paths; `python-publish.yml` SHALL build the frontend before `uv build` and run the artifact-parity comparison as a release gate.

#### Scenario: Publish path produces a complete wheel
- **WHEN** the publish workflow runs
- **THEN** the frontend is built first, the wheel contains the static UI, and the parity comparison gate passes before upload

#### Scenario: CI matrix covers extras per interpreter
- **WHEN** the CI matrix job runs for an interpreter version
- **THEN** the environment is created with `uv sync --all-extras` for that interpreter and the unit/acceptance suites run through `uv run`

### Requirement: Documentation reflects the unified setup
README and `docs/dashboard.md` SHALL document installation via extras (`pip install robotframework-doctestlibrary[dashboard]`/`[all]`) and the uv-based contributor workflow; references to the two-package-manager split and to `pipx install doctest-dashboard` SHALL be removed.

#### Scenario: Install docs match reality
- **WHEN** a user follows the README dashboard quickstart
- **THEN** the documented commands install the dashboard extra of the unified package and start the server successfully
