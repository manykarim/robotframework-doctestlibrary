# Design: uv-unified-packaging

## Context

Today: core library is poetry-managed (`[tool.poetry]`, poetry-core backend, `poetry.lock`), the dashboard is a separate uv-managed PEP 621 package under `dashboard/` with its own lock, venv, and wheel (which force-includes the built frontend). CI runs poetry jobs (matrix 3.9–3.13) plus a uv dashboard job. The publish workflow runs `python -m build` (backend-agnostic). Poetry expresses Python-version-dependent pins via multi-constraint tables; several `<3.9` branches are dead because `requires-python` is `^3.9`.

Facts verified in-session: `uv lock` refuses the poetry-style pyproject (no `[project]` table); `uv build` against the current root builds via poetry-core, so a **baseline wheel identical to poetry's output can be captured before migrating**; the dashboard's only 3.10+ syntax is `list[UploadFile]`; `python-publish.yml` needs a frontend build step once the wheel bundles the UI.

## Goals / Non-Goals

**Goals:**
- Single distribution `robotframework-doctestlibrary` with extras `ai`, `dashboard`, `all`; uv as the only environment/dependency manager; one universal lockfile.
- Byte-set parity of the `DocTest` portion of built artifacts with the poetry baseline (same files in, same files out, `frozen_east_text_detection.pb` still excluded).
- Faithful translation of every version-conditional dependency to PEP 508 markers, validated by resolving and testing on each of Python 3.9, 3.10, 3.11, 3.12, 3.13.
- Clean repo layout: `DocTest/` and `doctest_dashboard/` as peer packages, tests under `utest/` (+ `utest/dashboard/`), `atest/`, `e2e/`, `frontend/`.
- All existing tests green from their new locations; CI, invoke tasks, publish workflow, and docs updated.

**Non-Goals:**
- No behavior changes in library or dashboard code (except the 3.9 typing fix and a friendlier CLI error).
- No src/ layout migration for `DocTest` (it would churn imports, coverage configuration, and CI for no functional benefit — keep flat packages).
- No PyPI deprecation work for `doctest-dashboard` (never published).
- No dependency upgrades beyond what marker translation requires.

## Decisions

### D1 — One distribution, two top-level packages, extras for optional deps

`[tool.hatch.build.targets.wheel] packages = ["DocTest", "doctest_dashboard"]`. Extras:

```toml
[project.optional-dependencies]
ai = ["pydantic-ai-slim[openai]", "python-dotenv"]
dashboard = ["fastapi>=0.110", "uvicorn>=0.29", "pydantic>=2.6", "python-multipart>=0.0.9"]
all = ["robotframework-doctestlibrary[ai,dashboard]"]
```

Self-referencing extras are standard (pip ≥21.2, uv). The `doctest-dashboard` console script is always installed; `cli.py` catches the `ImportError` for fastapi/uvicorn and prints `pip install robotframework-doctestlibrary[dashboard]`. Alternative considered: keep two distributions with `[dashboard]` depending on `doctest-dashboard` — rejected (user wants one package; two-package version lockstep is the maintenance cost we're removing).

### D2 — Layout: dissolve `dashboard/`

- `dashboard/src/doctest_dashboard/` → `doctest_dashboard/`
- `dashboard/tests/` → `utest/dashboard/` (pytest discovers it with the existing `utest` runs; the dashboard conftest/fixtures stay self-contained in that folder; `helpers.py` REPO_ROOT depth changes from `parents[2]` to `parents[2]`→`parents[1]` — recompute, don't assume)
- `dashboard/e2e/` → `e2e/`
- `dashboard/frontend/` → `frontend/` (wheel force-include becomes `frontend/dist` → `doctest_dashboard/static`)
- delete `dashboard/pyproject.toml`, `dashboard/uv.lock`, `dashboard/.venv`, `dashboard/.gitignore` (merge entries into root `.gitignore`); `dashboard/README.md` content folds into `docs/dashboard.md`/root README.

Consequence: core `utest` runs would import dashboard tests — they require the `dashboard` extra + dev deps; that is fine because the dev environment installs `--all-extras`. CI jobs that must NOT install dashboard extras (3.9-matrix poetry job today) run `pytest utest --ignore=utest/dashboard` only if extras are absent; simpler: install `[all]` everywhere in CI (3.9-compatible after D5).

### D3 — Marker translation table (poetry multi-constraints → PEP 508)

| Package | PEP 508 requirement(s) |
|---|---|
| PyMuPDF | `PyMuPDF>=1.26.0` |
| numpy | `numpy>=1.25,<2.0; python_version < '3.10'` · `numpy>=1.26; python_version >= '3.10' and python_version < '3.13'` · `numpy>=2.1.0; python_version >= '3.13'` |
| scipy | `scipy; python_version < '3.12'` · `scipy>=1.11; python_version >= '3.12'` |
| scikit-image | `scikit-image; python_version < '3.12'` · `scikit-image>=0.22.0; python_version == '3.12'` · `scikit-image>=0.25.0; python_version >= '3.13'` |
| deepdiff | `deepdiff; python_version < '3.12'` · `deepdiff>=6.0; python_version >= '3.12'` |
| setuptools | `setuptools; python_version >= '3.12'` |
| coverage (dev group) | `coverage>=7.10.7,<7.11; python_version < '3.10'` · `coverage>=7.13.5; python_version >= '3.10'` |

All `<3.9` branches are dropped (`requires-python = ">=3.9"`). Unconditional deps (imutils, opencv-python-headless, parsimonious, pytesseract, robotframework>=4, Wand, pylibdmtx, pyzbar, robotframework-assertion-engine) carry over verbatim. Dev group additionally: pytest, invoke, robotframework-stacktrace, httpx, pytest-playwright (merged from the dashboard's dev group).

### D4 — Build backend hatchling, parity-gated

Baseline capture happens as the *first* implementation step, while the poetry backend is still in place: `uv build` (poetry-core) → record wheel + sdist file lists. After migration, a comparison script asserts: identical `DocTest/**` file set, `frozen_east_text_detection.pb` absent, README/LICENSE/metadata fields equivalent (name, version, requires-python, dependency set incl. markers, extras), additions limited to `doctest_dashboard/**` + `doctest_dashboard/static/**` + entry point. The script lives in `scripts/compare_wheel_contents.py` and runs in CI on the publish path. Hatchling config: `exclude = ["DocTest/data/frozen_east_text_detection.pb"]`, `artifacts = ["frontend/dist"]` for both targets (frontend/dist is gitignored), force-include mapping as today.

### D5 — Python 3.9 compatibility of the dashboard extra

Fix `list[UploadFile]` → `typing.List[UploadFile]` (FastAPI evaluates annotations at runtime, so this is functional, not cosmetic) and audit `doctest_dashboard` for other 3.10+ syntax (`X | Y` annotations, match statements) via running the dashboard unit tests under 3.9. fastapi/pydantic2/uvicorn/python-multipart all support 3.9.

### D6 — Multi-Python validation strategy

uv's lock is universal (markers resolved per environment at sync time). Validation per interpreter v ∈ {3.9, 3.10, 3.11, 3.12, 3.13}:
1. `uv sync --python $v --all-extras` (resolution + install works),
2. resolved-version audit: script reads `uv pip list` per env and asserts the expectation table from D3 (e.g. numpy<2 on 3.9, numpy≥2.1 on 3.13, scikit-image≥0.25 on 3.13),
3. import smoke (`import DocTest.VisualTest, doctest_dashboard.server.app`),
4. unit tests: full `utest/` in the CI matrix; locally the new `invoke multipython` task runs steps 1–3 plus a fast test subset for every interpreter (uv downloads interpreters via `uv python install`).

### D7 — Tooling and CI

- `tasks.py`: all subprocess invocations run through `uv run --` (utests, atests, tests, libdoc, coverage); new `multipython` task per D6; coverage `--source=DocTest` gains `,doctest_dashboard`.
- `ci.yml`: replace `snok/install-poetry` with `astral-sh/setup-uv` everywhere. Matrix job: `uv sync --python ${{ matrix }} --all-extras` + `uv run invoke tests`. Dashboard job: paths updated (`utest/dashboard`, `e2e/`, `frontend/`), still builds the frontend + runs Playwright on one Python. Smoke job: uv, plus the resolved-version audit step.
- `python-publish.yml`: add Node setup + `npm ci && npm run build` in `frontend/` **before** building (the wheel force-includes `frontend/dist`); build via `uv build`; keep pypa publish action. Run `scripts/compare_wheel_contents.py` against the committed baseline manifest as a release gate.
- Docker: unaffected (`pip install robotframework-doctestlibrary[ai]` from PyPI).

### D8 — Lockfiles and groups

One root `uv.lock`; `poetry.lock` and `dashboard/uv.lock` deleted. Dev tooling in `[dependency-groups] dev = [...]` (uv installs it by default on `uv sync`). Playwright/e2e deps stay in `dev` (small, avoids a second group; revisit if weight matters).

## Risks / Trade-offs

- [Marker translation subtly diverges from poetry's resolution] → expectation-table audit per interpreter (D6.2) + full test matrix in CI; baseline metadata comparison includes the dependency set.
- [Hatchling wheel differs from poetry-core in non-obvious ways (data files, normalization)] → automated content comparison against a pre-captured baseline, release-gated.
- [Path-depth bugs after the move (REPO_ROOT, conftest sys.path, force-include, playwright fixtures)] → all 90 dashboard tests + 552 core tests must pass from new locations before the change is complete; grep for `parents[`/`dashboard/` literals.
- [Wheel built without the frontend silently ships a stale/missing UI] → publish workflow builds the frontend first; the parity script asserts `doctest_dashboard/static/index.html` + hashed assets exist in the wheel.
- [`doctest-dashboard` script crashes without the extra] → guarded import with actionable message; unit test for it.
- [3.9 environment for dashboard tests pulls fastapi stack onto the oldest interpreter] → supported per upstream; validated in the matrix.

## Migration Plan

1. Capture baseline artifacts (poetry-core build) and commit their file-list manifest.
2. Restructure directories; fix paths; keep both old pyprojects momentarily so tests still run.
3. Swap root pyproject to PEP 621/hatchling/uv; delete dashboard pyproject + locks; `uv lock`; fix 3.9 typing.
4. Parity + multi-Python validation; tooling/CI/docs updates.
5. Rollback: single revert commit restores poetry layout (no external state — nothing published, no data formats changed).

## Open Questions

- Whether to also publish a `doctest-dashboard` *stub* distribution that just depends on `robotframework-doctestlibrary[dashboard]` for discoverability — deferred; nothing was ever published under that name.
- Version for the first unified release: continue `0.33.0` line (suggest `0.34.0` since packaging surface changes).
