# Design: modernize-toolchain-and-deps

## Context

Third change on this branch, after the correctness and performance changes. Dev/CI-focused; the only runtime-visible edits are dead-dependency removal and helper deduplication. The `uv-tooling` and `unified-packaging` specs govern the packaging layout — nothing here may violate them.

## Goals / Non-Goals

**Goals:** quality gates that pass on the current code; dependency hygiene without changing what users effectively install; CI that tests PR code and doesn't hard-depend on a paid external LLM; the Phase-0 signature snapshot as the standing BC guard.

**Non-Goals:** OIDC trusted publishing (needs PyPI-side configuration that cannot be done or verified from this repo — documented as a follow-up); removing barcode/printjob deps from the base set (deprecation clock only); any ruff rule expansion beyond a lenient baseline; the §5.2 architecture refactor.

## Decisions

1. **ruff config**: `[tool.ruff]` in pyproject; `lint.select = ["E4", "E7", "E9", "F", "B"]` (pyflakes + core pycodestyle + bugbear) with per-file ignores where the legacy code requires; `ruff format --check` NOT enforced repo-wide initially (the codebase is not black-formatted; reformatting 10k lines in this change would swamp the diff) — instead `format` runs only in pre-commit for changed files. CI job runs `ruff check`.
2. **mypy lenient**: `ignore_missing_imports = true`, no strictness flags, scoped to `DocTest/`; goal is catching obvious type errors, not full typing. CI job runs it non-blocking? No — blocking, but config kept lenient enough to pass today.
3. **imutils inline**: `grab_contours` is a 6-line version-compat shim around `cv2.findContours` return shape; inline as `_grab_contours` in VisualTest. Drop dep.
4. **Wand/scipy**: delete from `dependencies`; update `scripts/audit_resolved_versions.py` PRESENCE list accordingly. scikit-image keeps pulling scipy transitively — `import scipy` still works for users (unchanged effective env).
5. **Version floors** from the current uv.lock resolutions rounded down to the current minor (e.g. `opencv-python-headless>=4.8`, `pytesseract>=0.3.10`, `parsimonious>=0.10`, `pylibdmtx>=0.1.10`, `pyzbar>=0.1.9`, `imutils` removed). Floors only — no upper caps.
6. **Extras additive**: `[project.optional-dependencies] barcode = [...] printjobs = [...]`; `all` extended. Base `dependencies` unchanged for these packages (deprecation note in README section for extras is out of scope here; release notes carry it).
7. **Signature snapshot**: `utest/test_keyword_signatures.py` builds `{library: {keyword: [args...]}}` via `robot.libdoc.LibraryDocumentation` and compares against a committed JSON baseline (`utest/keyword_signatures_baseline.json`). Removed/renamed keywords, removed/renamed args, changed defaults → fail; additions → pass (assert baseline ⊆ current).
8. **CI**: in `ci.yml`, required matrix jobs run with `DOCTEST_LLM_ENABLED` unset; a new single-interpreter `llm-tests` job (continue-on-error / not in required checks) carries the secret. Add a `lint` job (ruff+mypy) and a `pip-audit` step. `docker-image.yml`: bump `actions/checkout@v4`, `docker/login-action@v3`, and change the Dockerfile to `COPY . /src && pip install /src[ai]` for the CI build. Unit-test jobs gain `-n auto --timeout=300`.
9. **Helper dedup**: new internal module `DocTest/_llm_flags.py` holding `_as_bool`, `_coerce_label_value`, `_decision_equals_flag`; VisualTest/PdfTest import from it (public names in those modules preserved as aliases — external code referencing `DocTest.PdfTest._as_bool` keeps working).
10. **print() cleanup**: replace with module logger calls at debug/info level.

## Risks / Trade-offs

- [ruff/mypy fail on legacy code] → tune per-file ignores until green rather than touching logic; the gate value is preventing NEW issues.
- [Version floors too high for someone's old env] → floors match versions the suite is actually tested against; anything older is unsupported in practice already.
- [CI edits can't be fully verified locally] → validate YAML syntax + `gh` lint where possible; behavior verified on the PR run.
- [Dockerfile change alters the published image build] → the published image still installs the released package for releases if desired; the CI path installs from checkout. Keep the Dockerfile parameterized (build arg) if both are needed.

## Migration Plan

Dev-only rollout; single release. Rollback = revert. The signature snapshot baseline is generated from the current code at merge time.

## Open Questions

None blocking. OIDC publishing recorded as a follow-up requiring PyPI-side setup.
