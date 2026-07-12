# Tasks: uv-unified-packaging

## 1. Baseline capture (while poetry backend is still active)

- [x] 1.1 Build wheel + sdist with the current poetry-core backend (`uv build` at root); record file lists and core metadata (name, version, requires-python, dependency specifiers incl. markers, extras) into a committed baseline manifest under `scripts/` (e.g. `scripts/wheel_baseline.json`)
- [x] 1.2 Write `scripts/compare_wheel_contents.py`: compares a wheel/sdist against the baseline — identical `DocTest/**` set, `frozen_east_text_detection.pb` absent, metadata equivalent, additions limited to `doctest_dashboard/**` + `doctest_dashboard/static/**` + entry points; self-test it against the baseline wheel (must pass trivially)

## 2. Restructure the repository

- [x] 2.1 Move `dashboard/src/doctest_dashboard/` → `doctest_dashboard/`; fix internal path constants (`server/app.py` DEV_DIST_DIR depth, any `parents[n]`)
- [x] 2.2 Move `dashboard/tests/` → `utest/dashboard/` and `dashboard/e2e/` → `e2e/`; recompute `helpers.py` REPO_ROOT depth and `sys.path` inserts; keep dashboard fixtures self-contained so core-only environments can deselect `utest/dashboard`
- [x] 2.3 Move `dashboard/frontend/` → `frontend/`; update the vite proxy comment, wheel force-include source path, and e2e server fixture working assumptions
- [x] 2.4 Delete `dashboard/pyproject.toml`, `dashboard/uv.lock`, `dashboard/.venv`, `dashboard/.gitignore` (merge ignore entries into root `.gitignore`); fold `dashboard/README.md` content into `docs/dashboard.md` + root README dev section; remove the now-empty `dashboard/`
- [x] 2.5 Run dashboard tests + e2e from new locations against the still-poetry root (using a temporary uv env) to isolate move bugs from packaging bugs

## 3. Unified pyproject + lock

- [x] 3.1 Rewrite root `pyproject.toml` to PEP 621: metadata parity with poetry fields (authors, license, readme, homepage, keywords/classifiers if present), `requires-python = ">=3.9"`, dependencies with the design's marker translation table, extras `ai`/`dashboard`/`all` (self-referencing), `[dependency-groups] dev` (pytest, invoke, coverage markers, robotframework-stacktrace, httpx, pytest-playwright), console script `doctest-dashboard`
- [x] 3.2 Hatchling build config: wheel packages `DocTest` + `doctest_dashboard`, exclude `DocTest/data/frozen_east_text_detection.pb`, force-include `frontend/dist` → `doctest_dashboard/static`, `artifacts` for the gitignored dist, sdist includes tests/docs equivalently to the poetry sdist baseline
- [x] 3.3 Fix Python 3.9 compatibility in `doctest_dashboard` (`list[UploadFile]` → `List[UploadFile]`; audit for other 3.10+ syntax); add guarded import in `cli.py` with the `robotframework-doctestlibrary[dashboard]` hint + unit test
- [x] 3.4 `uv lock`; delete `poetry.lock`; `uv sync --all-extras` and verify imports of both packages

## 4. Parity and multi-Python validation

- [x] 4.1 Build with hatchling (`uv build`, frontend built first); run `scripts/compare_wheel_contents.py` against the baseline until clean; add a pytest wrapper so parity runs with the normal suite when `dist/` artifacts are present
- [x] 4.2 Write `scripts/audit_resolved_versions.py` asserting the marker expectation table (numpy/scipy/scikit-image/deepdiff/PyMuPDF per interpreter); run it for 3.9–3.13 via `uv sync --python X --all-extras` (interpreters via `uv python install`)
- [x] 4.3 Per-interpreter smoke: import `DocTest.VisualTest` + `doctest_dashboard.server.app` and run the fast unit subset on 3.9 and 3.13 locally; full `utest/` (incl. `utest/dashboard`) on the default interpreter
- [x] 4.4 Fresh-venv install matrix from the built wheel: base (lean check: no fastapi), `[ai]`, `[dashboard]` (serve + UI smoke), `[all]`; console-script friendly-error check on the base install

## 5. Tooling

- [x] 5.1 Update `tasks.py`: all commands through `uv run --`; coverage sources `DocTest,doctest_dashboard`; atest list unchanged; libdoc task verified; new `multipython` invoke task (sync + audit + import smoke per interpreter, summary output)
- [x] 5.2 Root `.gitignore` consolidation (frontend dist/node_modules, .venv, uploads/scratch dirs); remove stale references to `dashboard/` paths repo-wide (grep)

## 6. CI and publish

- [x] 6.1 Rework `ci.yml`: smoke + matrix jobs on `astral-sh/setup-uv` with `uv sync --all-extras --python ${{ matrix }}`, run via `uv run invoke tests`; add the resolved-version audit step; dashboard/e2e job updated to `utest/dashboard`, `e2e/`, `frontend/` paths; remove poetry installs
- [x] 6.2 Rework `python-publish.yml`: Node setup + `npm ci && npm run build` in `frontend/`, `uv build`, parity-comparison gate, publish; verify `docker-image.yml` needs no change (PyPI install)
- [x] 6.3 Bump version to 0.34.0 in `pyproject.toml` and `DocTest/__init__` (wherever `__version__` lives); regenerate keyword docs via the uv-based libdoc task

## 7. Documentation

- [x] 7.1 README: install matrix (`pip install robotframework-doctestlibrary`, `[ai]`, `[dashboard]`, `[all]`), dashboard quickstart updated away from `pipx install doctest-dashboard`, contributor section (uv sync / uv run / invoke)
- [x] 7.2 `docs/dashboard.md`: installation + development sections updated to the unified package and new paths; absorb `dashboard/README.md` development content
- [x] 7.3 Remove/replace all two-package-manager references (docs, comments, design notes in code)

## 8. Final verification

- [x] 8.1 Full core suites via uv: `uv run pytest utest -q` and the acceptance suites from `tasks.py`; dashboard e2e via `uv run pytest e2e --browser chromium`
- [x] 8.2 `invoke multipython` green for 3.9–3.13 (sync, audit, import smoke); dashboard unit tests pass under 3.9
- [x] 8.3 Release-equivalent dry run: clean checkout simulation (git stash untracked / worktree), frontend build, `uv build`, parity gate, fresh-venv `[all]` install, `doctest-dashboard serve` smoke
