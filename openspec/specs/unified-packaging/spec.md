# Spec: unified-packaging

## Purpose
One installable package with optional extras covering AI and dashboard features.

## Requirements

### Requirement: Single distribution with extras
The project SHALL build exactly one distribution, `robotframework-doctestlibrary`, containing the `DocTest` and `doctest_dashboard` packages, with optional dependencies exposed as extras: `ai` (LLM keywords), `dashboard` (web dashboard server), and `all` (both). A base install SHALL NOT pull fastapi/uvicorn/pydantic-ai.

#### Scenario: Base install stays lean
- **WHEN** the wheel is installed without extras into a fresh environment
- **THEN** `DocTest.VisualTest` imports and works, and neither `fastapi` nor `pydantic_ai` is installed

#### Scenario: Dashboard extra enables the server
- **WHEN** the wheel is installed with `[dashboard]`
- **THEN** `doctest-dashboard serve` starts and serves the bundled web UI

#### Scenario: All extra is the union
- **WHEN** the wheel is installed with `[all]`
- **THEN** the dependency set equals the union of `[ai]` and `[dashboard]`

### Requirement: Console script degrades gracefully
The `doctest-dashboard` console script SHALL be installed with every variant and, when dashboard dependencies are missing, SHALL exit with a clear message naming `pip install robotframework-doctestlibrary[dashboard]` instead of a traceback.

#### Scenario: Friendly error without the extra
- **WHEN** `doctest-dashboard serve` runs in an environment without the dashboard extra
- **THEN** the process exits non-zero with a message containing `robotframework-doctestlibrary[dashboard]` and no traceback

### Requirement: Artifact content parity with the poetry baseline
Built artifacts SHALL contain the identical `DocTest/**` file set as the poetry-core-built baseline captured before the migration — including the exclusion of `DocTest/data/frozen_east_text_detection.pb` — with additions limited to `doctest_dashboard/**`, the bundled frontend under `doctest_dashboard/static/**`, and the console-script entry point. Core metadata (name, version, requires-python, dependency specifiers, extras) SHALL be equivalent. An automated comparison script SHALL enforce this and run on the release path.

#### Scenario: Wheel comparison passes
- **WHEN** the comparison script runs against the new wheel and the committed baseline manifest
- **THEN** it reports no missing or unexpected `DocTest` files and confirms the East model file is absent

#### Scenario: Wheel ships the built frontend
- **WHEN** the wheel is built on the release path
- **THEN** it contains `doctest_dashboard/static/index.html` and hashed asset files

### Requirement: Version-conditional dependencies preserved as markers
Poetry's Python-version-dependent constraints SHALL be translated to PEP 508 environment markers preserving the selection intent for Python 3.10–3.13 (numpy `>=2.1` on 3.13+, scipy `>=1.11` on 3.12+, scikit-image `>=0.22` on 3.12 and `>=0.25` on 3.13+, deepdiff `>=6.0` on 3.12+, setuptools present on 3.12+, PyMuPDF `>=1.26`). `requires-python` SHALL be `>=3.10` — Python 3.9 is past end of life and unsupported by maintained pydantic-ai (the `[ai]` extra).

#### Scenario: Resolution matches the expectation table per interpreter
- **WHEN** the environment is synced for each of Python 3.10, 3.11, 3.12, and 3.13
- **THEN** the resolved versions of numpy, scipy, scikit-image, deepdiff, and PyMuPDF satisfy the documented expectation table for that interpreter
