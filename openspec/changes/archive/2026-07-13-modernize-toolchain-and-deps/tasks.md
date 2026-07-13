# Tasks: modernize-toolchain-and-deps

## 1. Backwards-compatibility guard first

- [x] 1.1 Keyword-signature snapshot test: generate `utest/keyword_signatures_baseline.json` via libdoc for VisualTest/PdfTest/PrintJobTests/WebVisualTest and add `utest/test_keyword_signatures.py` (baseline ⊆ current; defaults compared)

## 2. Dependency hygiene

- [x] 2.1 Inline `imutils.grab_contours` into VisualTest; remove `imutils` from dependencies
- [x] 2.2 Remove `Wand` and direct `scipy` from dependencies; update `scripts/audit_resolved_versions.py` PRESENCE list
- [x] 2.3 Add lower version bounds to remaining unpinned runtime deps (opencv-python-headless, pytesseract, parsimonious, pylibdmtx, pyzbar)
- [x] 2.4 Add `[barcode]` and `[printjobs]` extras (additive; base deps unchanged); extend `all`
- [x] 2.5 `uv sync` + wheel build + full test suite to prove the dependency changes are inert

## 3. Quality gates

- [x] 3.1 Add `[tool.ruff]` (lenient E4/E7/E9/F/B baseline) and fix/ignore until `uv run ruff check DocTest` is green
- [x] 3.2 Add `[tool.mypy]` (lenient, DocTest scope) and adjust until `uv run mypy DocTest` is green
- [x] 3.3 Add `.pre-commit-config.yaml` (ruff, ruff-format on changed files, whitespace hooks)
- [x] 3.4 Add pytest-xdist + pytest-timeout to dev group; verify `uv run pytest utest -n auto --timeout=300` is green

## 4. Code quick wins

- [x] 4.1 New `DocTest/_llm_flags.py` with `_as_bool`/`_coerce_label_value`/`_decision_equals_flag`; VisualTest and PdfTest import from it (aliases preserved)
- [x] 4.2 Replace stray `print()` in PdfTest, PrintJobTests, CapabilityCheck with logger calls

## 5. CI/CD

- [x] 5.1 ci.yml: remove `DOCTEST_LLM_ENABLED` from the required matrix; add a single-interpreter optional `llm-tests` job; add a `lint` job (ruff + mypy); use `-n auto --timeout=300` for unit tests
- [x] 5.2 Add `pip-audit` step and `.github/dependabot.yml` (pip + github-actions)
- [x] 5.3 docker-image.yml: bump checkout/login action majors; build the image from the checkout instead of PyPI

## 6. Verification

- [x] 6.1 Full unit suite green under `-n auto --timeout=300`; signature snapshot test green; ruff/mypy green
