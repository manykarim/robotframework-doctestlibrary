# Proposal: uv-unified-packaging

## Why

The repository currently runs two package managers (poetry for the core, uv for the dashboard) with two pyprojects, two lockfiles, and two distributions — a split that already caused contributor friction (broken poetry launcher, duplicated test helpers, separate CI jobs). Consolidating on uv with a single `robotframework-doctestlibrary` distribution and extras makes the project simpler to install (`pip install robotframework-doctestlibrary[all]`), maintain, and release.

## What Changes

- **One distribution, extras for everything optional**: the wheel ships both the `DocTest` and `doctest_dashboard` packages; optional dependencies move to extras — `[ai]` (existing: pydantic-ai-slim, python-dotenv), `[dashboard]` (new: fastapi, uvicorn, pydantic, python-multipart), `[all]` (both, via self-referencing extra). The separate `doctest-dashboard` distribution disappears (never published to PyPI — no migration debt). The `doctest-dashboard` console script stays and gets a friendly "install the [dashboard] extra" error when its dependencies are missing.
- **Repository restructure**: `dashboard/src/doctest_dashboard/` moves to `doctest_dashboard/` (peer of `DocTest/`); dashboard backend tests join `utest/dashboard/`; Playwright journeys move to `e2e/`; the frontend moves to `frontend/`; `dashboard/pyproject.toml`, `dashboard/uv.lock`, and `dashboard/.venv` are removed.
- **PEP 621 + hatchling + uv**: the root `pyproject.toml` is rewritten from `[tool.poetry]` to PEP 621 with environment markers replacing poetry's multi-constraint tables (PyMuPDF, numpy, scipy, scikit-image, deepdiff, setuptools, coverage — dropping dead `<3.9` branches since `requires-python = ">=3.9"`); dev tooling becomes uv dependency-groups; one universal `uv.lock`; `poetry.lock` is removed.
- **Wheel/sdist content parity**: the previous poetry-core-built artifacts are captured as a baseline *before* the switch; an automated comparison asserts the new hatchling wheel contains the identical `DocTest` file set (including the `frozen_east_text_detection.pb` exclusion) plus only the intended additions (`doctest_dashboard/*`, bundled frontend).
- **Python 3.9–3.13 validation**: dashboard code is made 3.9-compatible (one `list[UploadFile]` annotation); per-interpreter resolution and import/test smoke runs validate that the markers reproduce poetry's version-selection intent (e.g. numpy `<2.0` on 3.9, `>=2.1` on 3.13) on every supported Python.
- **Tooling/CI/docs updates**: `tasks.py` (invoke) switches to `uv run` and gains a multi-Python validation task; `ci.yml` jobs become uv-based (matrix 3.9–3.13 with extras, dashboard e2e from new paths); `python-publish.yml` builds the frontend before the wheel; README, `docs/dashboard.md`, and contributor docs describe the single-package install and uv workflow.

## Capabilities

### New Capabilities

- `unified-packaging`: one PEP 621 distribution with `ai`/`dashboard`/`all` extras, hatchling build, wheel/sdist content parity with the poetry-built baseline, and a working console script across install variants.
- `uv-tooling`: uv as the single environment/dependency manager — universal lockfile, dependency groups, multi-Python (3.9–3.13) resolution and test validation, uv-based invoke tasks, CI, and publish workflow.

### Modified Capabilities

(none — `openspec/specs/` has no archived specs; packaging requirements are introduced fresh)

## Impact

- **Files**: root `pyproject.toml` rewritten; `dashboard/` directory dissolved into `doctest_dashboard/`, `utest/dashboard/`, `e2e/`, `frontend/`; `poetry.lock` and `dashboard/uv.lock` replaced by one root `uv.lock`; `tasks.py`, `.github/workflows/ci.yml`, `.github/workflows/python-publish.yml` updated; README/docs updated. Test-helper paths (`REPO_ROOT` depth, hatch force-include, playwright fixtures) adjusted to the new layout.
- **No runtime behavior change** in `DocTest` or the dashboard — this is packaging/tooling only; all 552 core and 90 dashboard tests must stay green from their new locations.
- **Consumers**: `pip install robotframework-doctestlibrary` keeps working identically; dashboard users switch from `pipx install doctest-dashboard` (never released) to `robotframework-doctestlibrary[dashboard]`. Docker image already installs `[ai]` from PyPI — unaffected.
- **Verified facts feeding this change**: publish workflow uses backend-agnostic `python -m build`; only `ci.yml` references poetry; the dashboard's sole 3.10+ syntax is `list[UploadFile]` (`server/app.py:255`); poetry's `<3.9` constraint branches are unreachable.
