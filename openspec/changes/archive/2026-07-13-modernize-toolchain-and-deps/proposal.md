# Proposal: modernize-toolchain-and-deps

## Why

The toolstack audit (docs/ultradeep-analysis-solution-proposal.md §5.3, §6) found no lint/format/type gate anywhere, a dead runtime dependency (Wand — imported nowhere), an unmaintained dependency (imutils) used for one 10-line function, seven unpinned runtime dependencies, live-LLM tests on the required CI path of every PR × 4 interpreters, and a Docker CI job that installs from PyPI so it never tests the PR's code.

## What Changes

- Adopt **ruff** (lint + format check, lenient rule set) and **mypy** (lenient) over `DocTest/`, wired into CI and a new `.pre-commit-config.yaml`.
- Add **pytest-xdist** and **pytest-timeout** to the dev group; CI runs `pytest -n auto --timeout=300`.
- Add a **coverage floor** to the existing coverage tasks.
- Dependencies: **drop Wand** (unused), **inline `imutils.grab_contours`** (~8 lines) and drop imutils, **drop the direct scipy entry** (transitive via scikit-image), add **tested lower bounds** to the remaining unpinned runtime deps.
- Add **additive extras** `barcode` (pylibdmtx, pyzbar) and `printjobs` (parsimonious) — base dependencies keep them for now (deprecation clock starts; removal only in a documented future minor/major release).
- CI hygiene: move live-LLM tests to a single-interpreter **non-required job**; fix `docker-image.yml` to build **from the checkout** and bump EOL action versions; add **dependabot.yml** and a **pip-audit** step.
- Add the **Phase-0 keyword-signature snapshot test**: libdoc-derived baseline of every public keyword's name, argument names and defaults, failing on any non-additive change.
- Quick wins: remove stray `print()` calls in `PdfTest`/`PrintJobTests`/`CapabilityCheck` (route to loggers); deduplicate the byte-identical `_coerce_label_value`/`_decision_equals_flag`/`_as_bool` helpers into an internal module.

**No breaking changes** for library users: runtime dependency set is unchanged except removals of packages that were never imported (Wand) or replaced by inlined code (imutils); version floors match what uv.lock already resolves; extras are additive.

## Capabilities

### New Capabilities
- `toolchain-quality-gates`: repository quality gates — lint/format/type checks, parallel+timeout-guarded tests, keyword-signature compatibility snapshot, dependency hygiene (floors, extras, no dead deps), and CI that tests the checked-out code.

### Modified Capabilities
<!-- none — api-compatibility already mandates signature stability; the snapshot test enforces it rather than changing it -->

## Impact

- Code: `pyproject.toml`, `.github/workflows/*`, `Dockerfile`, `.pre-commit-config.yaml` (new), `.github/dependabot.yml` (new), `DocTest/VisualTest.py` (imutils inline, print cleanup via helpers module), `DocTest/PdfTest.py`, `DocTest/PrintJobTests.py`, `DocTest/CapabilityCheck.py`, `scripts/audit_resolved_versions.py` (PRESENCE list), new `utest/test_keyword_signatures.py`.
- APIs: none.
- Users: slightly smaller install (Wand, imutils gone); everything else dev/CI-only.
