# Tasks

## 1. Packaging and lock

- [x] 1.1 In `pyproject.toml`:
  - set `requires-python = ">=3.11"`
  - drop the 3.10 classifier and add 3.14/3.15 classifiers
  - set ruff `target-version = "py311"`
  - update the Python-floor comment above `dependencies`
  - verify: `uvx ruff check DocTest` passes
- [x] 1.2 Rewrite the version-conditional dependency markers per design D2:
  - close the 3.13+ branches at `< '3.15'`
  - add the 3.15 floors: numpy `>=2.5.2`, scikit-image `>=0.27.0rc1`, deepdiff `>=9.2.1`
  - drop the 3.10-only intent
  - add caps: `PyMuPDF>=1.26.0,<2`, `opencv-python-headless>=4.8,<5`, `[ai]` and `[all]` `pydantic-ai-slim[openai]>=1.107.7,<3`
  - verify: `uv lock` resolves with no error
- [x] 1.3 Regenerate `uv.lock` with uv ≥0.13 (`uv lock --upgrade`). Verify that the lock contains pymupdf 1.28.x, opencv-python-headless 4.x, scikit-image 0.27.0rc1 only under the `>= '3.15'` marker, and deepdiff ≥9.2.1.
- [x] 1.4 Verify the binary-only install from a built wheel (`uv build --wheel`) on Windows:
  - create a fresh Python 3.15 venv
  - run `pip install --only-binary :all: "<wheel>[ai,dashboard]"`
  - this must succeed and resolve scikit-image 0.27.0rc1
  - confirm `pip install` of the wheel is refused on Python 3.10

## 2. LLM client compatibility (pydantic-ai 1.107.7 – 2.x)

- [x] 2.1 In `DocTest/llm/client.py`:
  - build the agent with `retries=max(1, settings.output_retries)` instead of `output_retries=`
  - map the default and `openai` provider to the `openai-chat:<model>` identifier (azure and other providers unchanged)
  - call `os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")` before the lazy `pydantic_ai` import
- [x] 2.2 Update `utest/test_llm_support.py`:
  - assert the `retries` kwarg in the retry test
  - assert `openai-chat:` identifiers
  - add a test that a pre-set `PYDANTIC_AI_NO_BANNER` value is not overridden
  - add a FunctionModel-based test proving that `retries=3` yields 4 model calls on persistently malformed output
  - verify: `uv run pytest utest/test_llm_support.py` passes on the locked pydantic-ai 2.x
- [x] 2.3 Verify the oldest supported runtime:
  - run `uv run --with "pydantic-ai-slim[openai]==1.40.0" pytest utest/test_llm_support.py`; it must pass (code compatibility; the declared floor is 1.107.7 for security)
  - run the same with `==1.107.7`, the declared floor; it must pass
  - with 1.107, no `PydanticAIDeprecationWarning` about `output_retries` or `openai:` may be emitted
- [x] 2.4 Document in the README LLM section that the `[ai]` extra supports pydantic-ai 1.107.7–2.x and that `openai` uses Chat Completions (OpenAI-compatible base URLs keep working). Verify that the README renders and the section states the supported range.

## 3. EAST model provisioning

- [x] 3.1 In `DocTest/Ocr.py`, make `_get_east` download to a unique temp file in the model directory, then `os.replace` it onto the target (design D5):
  - clean up the temp file on failure
  - on `PermissionError` with an existing target, keep the existing file
- [x] 3.2 Add unit tests that need no network access (monkeypatch `urllib.request.urlretrieve`):
  - an interrupted download leaves no file at the model path
  - an existing model is not re-downloaded
  - two concurrent provisioning threads both end with a loadable, complete file
  - verify: `uv run pytest utest/test_ocr.py -k east -n auto` passes with the model file deleted beforehand
- [x] 3.3 Fix the EAST recognition defects in `DocTest/Ocr.py` (design D13):
  - `import urllib.request`
  - drop rows with negative confidence or empty text
  - median-line-height estimate per region, with `--psm 6` for regions taller than 1.6× the median
  - vertical padding from the line height
  - tight-crop retry when no alphanumeric word is recognized
  - delete the unused `_extract_text`

  Verify: `uv run pytest utest/test_ocr.py utest/test_masks.py utest/test_visualtest.py utest/test_correctness_fixes.py -k "east or East"` passes.
- [x] 3.4 Add unit tests for the EAST recognition contract:
  - every returned token is non-empty and none comes from a `-1` structural row (stub `pytesseract.image_to_data`)
  - a multi-line region is recognized with block mode (stubbed)
  - an empty first recognition triggers exactly one tight-crop retry (stubbed)
  - a fresh-interpreter subprocess provisioning EAST with a stubbed download raises no `AttributeError`
  - real-image assertions that `text_small.png` and `text_big.png` yield `ABCDEFGHI`, `abcdefghi` and `1234567890`

  Verify: the new tests pass on Windows (Tesseract 5.5) and in the Linux container (Tesseract 5.3).

## 4. Deprecation and interpreter-hygiene fixes

- [x] 4.1 Replace `import fitz` with `import pymupdf` and update every `fitz.` reference in `DocTest/` and `utest/`, including:
  - the monkeypatch target in `utest/test_correctness_fixes.py`
  - the `import_name` in the `DocTest/CapabilityCheck.py` registry

  Verify: `grep -rn "fitz" DocTest utest` returns no hits, and `python -W error::DeprecationWarning -c "import DocTest.VisualTest"` emits no fitz warning.
- [x] 4.2 Change the PostScript grammar literal `ws = ~"\s*"` to a raw literal in `DocTest/PrintJobTests.py`. Verify: under 3.15, `uv run python -W error::SyntaxWarning -m pytest utest/test_printjobtests.py` passes.
- [x] 4.3 Pass `data_range=255` to both `structural_similarity` calls in `DocTest/DocumentRepresentation.py`. Verify: `utest/test_visualtest.py`, `utest/test_pixel_tolerance.py` and `utest/test_performance_equivalence.py` pass unchanged; the SSIM scores are identical.
- [x] 4.4 Add `errors="replace"` to the `subprocess.run(..., text=True)` call in `DocTest/CapabilityCheck.py::_get_version`. Add a unit test that monkeypatches a binary emitting cp1252 bytes (`b"Ung\xfcltige"`) and asserts that no exception is raised and none is reported through `threading.excepthook`. Verify on 3.15.
- [x] 4.5 In `DocTest/CapabilityCheck.py`, skip a fallback binary that resolves into the Windows system directory (design D15). Add unit tests for both branches:
  - System32 `convert.exe` gives "missing" with an install hint
  - a non-system `convert` is accepted with the fallback note

  Verify on Windows: `Check DocTest Environment` no longer reports ImageMagick via System32.
- [x] 4.6 Add a compile check test that byte-compiles every module in `DocTest/` and `doctest_dashboard/` with SyntaxWarning as an error. Verify that it passes on 3.15.

## 5. Test isolation

- [x] 5.1 Make `utest/test_ocr.py` (hires test) and `utest/test_profiler.py` (both big-file tests) write the upscaled `birthday_1080_date_id_10x.png` into `tmp_path` instead of `utest/testdata`. Verify:
  - `uv run pytest utest -n auto` run three times in a row has no `Cannot load image … _10x.png` failure
  - `git status --porcelain utest/testdata` is empty afterwards
- [x] 5.2 Delete the stray untracked `utest/testdata/birthday_1080_date_id_10x.png` from the working tree and verify that `git status` no longer lists it.
- [x] 5.3 Interpolate paths with `.as_posix()` in the generated Robot suites in `utest/dashboard/conftest.py` and `utest/dashboard/test_visual_polish.py` (design D16). Verify: `uv run pytest utest/dashboard -n auto` passes on Windows.

## 5b. Mask editor on Windows

- [x] 5b.1 In `frontend/src/MaskEditor.tsx`, derive the upload folder from the last `/` or `\` and append `masks.json` with that separator (design D14). Make `utest/dashboard/test_upload.py::test_mask_save_next_to_upload` derive the folder the same way. Verify:
  - `npm run build` in `frontend/` succeeds (type-check plus bundle)
  - the test passes on Windows and Linux

- [x] 5b.2 Add `frontend/src/paths.ts` (`joinPath`, `siblingPath`) and use it for the file browser's save-mode target and the upload-relative mask target. Fix the e2e harness (design D19):
  - write the server's output to a log file instead of an undrained pipe
  - wrap every path interpolated into e2e Robot suites with `robot_path()`
  - make j6 assert the separator-preserving rule

  Verify: `npm run build` succeeds, and `uv run pytest e2e --browser chromium` passes 18/18 on Windows and in the Linux container.

## 5c. Web capture arguments

- [x] 5c.1 Escape all argument values that `DocTest/WebCapture.py` passes through `BuiltIn.run_keyword` (design D20), make `FakeBuiltIn` in `utest/test_web_visual.py` apply Robot's `unescape`, and add regression tests (Windows path with `\new`/`${}`, locator with `=` and a backslash, DOM-walker script intact). Make `atest/web/xbrowser_vrt.robot` compare the fixed viewport. Verify:
  - the new tests fail without the fix and pass with it
  - `atest/web` passes 29/29 on Windows and Linux (Linux: Browser suites in the container; Selenium suite as non-root with seccomp unconfined, like a CI VM)

## 6. Tooling: audit, parity gate, invoke

- [x] 6.1 Update the expectation table in `scripts/audit_resolved_versions.py`:
  - remove 3.10
  - add rows for 3.14 and 3.15 (3.15: numpy ≥2.5.2, scikit-image ≥0.27.0rc1, deepdiff ≥9.2.1, scipy ≥1.18.1)
  - add caps to every row: pymupdf <2, opencv-python-headless <5, pydantic-ai-slim >=1.107.7,<3
  - update the usage docstring

  Verify: `uv run python scripts/audit_resolved_versions.py` prints `AUDIT OK` under 3.11 and 3.15.
- [x] 6.2 Update `scripts/compare_wheel_contents.py` (design D12):
  - set `SUPPORTED_PYTHONS` to 3.11–3.15
  - extend the intentional-tightening allow-list so documented entries may add specifiers or replace a baseline floor; register the new caps and floors
  - anything else must still fail

  Verify: `uv build` then `uv run python scripts/compare_wheel_contents.py dist/*.whl --sdist dist/*.tar.gz` prints `Parity OK`, and fails again if a temporary unlisted change is introduced (e.g. editing one specifier locally).
- [x] 6.3 Set `SUPPORTED_PYTHONS = ["3.11", "3.12", "3.13", "3.14", "3.15"]` in `tasks.py`, and give each interpreter its own environment (`UV_PROJECT_ENVIRONMENT=.venv-multipython/<version>`). Re-syncing the active `.venv` fails on Windows, because uv cannot delete `.venv\Scripts` while invoke runs from it. Verify: `uv run invoke multipython` reports OK for all five interpreters on Windows, and the active `.venv` is unchanged.

## 7. CI and docs

- [x] 7.1 In `.github/workflows/ci.yml`:
  - set the `test` matrix to `["3.11", "3.12", "3.13", "3.14", "3.15"]`
  - move `llm-tests` to `"3.15"`

  Verify: `actionlint` (or `uvx zizmor`/YAML parse) reports no errors.
- [x] 7.2 Add a required `windows` job to `ci.yml` (design D17):
  - `runs-on: windows-latest`, setup-uv with Python 3.15
  - install Tesseract via Chocolatey and add it to `GITHUB_PATH`
  - run `uv sync --all-extras`, `uv run python scripts/audit_resolved_versions.py`, and `uv run pytest utest -n auto --timeout=300`

  Verify: the workflow YAML parses, and the same command sequence passes locally on Windows.
- [x] 7.3 Update the README:
  - supported Python versions: 3.11–3.15, tested on 3.11/3.12/3.13/3.14/3.15
  - the `invoke multipython` comment
  - a note that Python 3.10 users should pin the last 3.10-compatible release

  Verify: `grep -n "3\.10" README.md` only shows the pinning note.

## 8. Integration verification

- [x] 8.1 Linux: in an Ubuntu 24.04 container with CI's apt packages and GhostPCL, run the full unit suite and all acceptance suites from `tasks.py` on 3.11 and 3.15. Verify: the unit suite has 0 failures, and the acceptance suites match baseline (105 pass, 12 skip, 0 fail).
- [x] 8.2 Windows: on Python 3.15, run `uv sync --all-extras` and the full unit suite. Verify: 0 failures, and no `UnicodeDecodeError` thread warnings.

## 9. Release 0.35.0 documentation

- [x] 9.1 Bump `version` to 0.35.0 in `pyproject.toml` and refresh the project entry in `uv.lock`. Verify: `DocTest.__version__ == "0.35.0"` after `uv sync`.
- [x] 9.2 Remove the stale case-duplicates `docs/AI.html` and `docs/AI-0.28.0.html` (keeping the canonical `Ai*.html` that `invoke libdoc`, `docs/index.html` and the README use), and add a test that fails on tracked paths differing only by case. Verify: the test passes, the same check over the previous HEAD reports exactly these two pairs, and `git status` is clean for `docs/` on Windows.
- [x] 9.3 Run `invoke libdoc` (repo practice: unversioned plus `-<version>` files for VisualTest, PdfTest, PrintJobTest, WebVisualTest, Ai) and validate each document:
  - the embedded libdoc model reports version 0.35.0 in both files
  - the versioned and unversioned files list the same keywords
  - the keyword set equals the live library
  - no keyword was removed since 0.34.0
  - the keyword-signature snapshot tests pass

## 10. CI runner, Sonar and dependency security

- [x] 10.1 Make the ImageMagick policy copy version-agnostic in every CI job (Ubuntu 26.04 ships ImageMagick 7). Verify: CI's install step, extracted verbatim from `ci.yml`, succeeds in an `ubuntu:26.04` container and writes `/etc/ImageMagick-7/policy.xml`.
- [x] 10.2 Pin every GitHub Action to a full commit SHA (folding in Dependabot #155, #156, #157, #164), move test-reporter off `@main`, and add `--no-build` to every `uv sync`/`uv run`. Verify: `actionlint` is clean, no `uses:` without a SHA remains, and `uv sync --no-build --all-extras` installs the project.
- [x] 10.3 Handle the floating promises Sonar reports (S9383) in `FileBrowser.tsx` and `MaskEditor.tsx`. Verify: `npm run build` succeeds and e2e passes 18/18.
- [x] 10.4 Upgrade vite and @vitejs/plugin-react (folding in Dependabot #154) and fix the transitive postcss, source-map-js and nanoid advisories. Verify: `npm audit` reports 0 vulnerabilities and e2e passes.
- [x] 10.5 Raise the `[ai]` floor to pydantic-ai-slim 1.107.7. Verify: the LLM tests pass against exactly 1.107.7, and the audit and wheel-parity gates accept the floor.
- [x] 10.6 Verify every open Dependabot alert is resolved by the branch's lock files (all locked versions at or above the first patched version, across all interpreter forks).
- [x] 10.7 Make the small-sign OCR assertions (three unit tests and `atest/Compare.robot`) accept one-character OCR variance via a shared helper. Verify: the full CI-equivalent `invoke tests` passes on Ubuntu 26.04 for Python 3.11 and 3.15.
- [x] 10.8 Fix the e2e race behind the j9 CI failure: journeys clicked the run list's first row right after an ingest, while the list could still show the previous newest run. A helper now opens the run named in the ingest message by id (7 sites). Verify: e2e 18/18 on Windows, 3/3 runs on Ubuntu 26.04.
- [x] 10.9 Add `--locked` to every `uv sync`/`uv run` in the workflows (Sonar S8544), so CI fails instead of re-resolving when `uv.lock` is stale. Verify: `uv sync --locked --no-build --all-extras` succeeds, and actionlint is clean.
- [x] 10.10 Pin the Windows CI job's English Tesseract model. Chocolatey's Tesseract 5.5.3 installer downloads `tessdata_fast`, with which EAST reads the birthday image's `ABCDEFGHI` as `XBCDEFCHI` (reproduced byte-for-byte locally with the official 5.5.3 binary). The job now installs `tessdata_best` 4.1.0 `eng.traineddata`, checked by SHA-256. The README recommends that model for EAST. Verify: full unit suite passes on Python 3.15 with Tesseract 5.5.3 + tessdata_best.

## Workflow follow-up

- Push the branch (with the maintainer's go-ahead). Confirm every required CI job is green — lint, smoke-no-ai, test ×5, dashboard incl. parity gate, web, windows — and that the optional `llm-tests` job passes with pydantic-ai 2.x.
- When scikit-image 0.27.0 final is released, replace `>=0.27.0rc1` with `>=0.27.0`, refresh `uv.lock`, and update the 3.15 audit row.
- Revisit the PyMuPDF `<2` and OpenCV `<5` caps in a dedicated change (ONNX EAST model, re-baselined template matching).
- Switch SSIM to the public `skimage2.metrics.structural_similarity` once scikit-image ships the `skimage2` namespace. 0.27 emits `PendingSkimage2Change` unconditionally, and `data_range` is already explicit.
- Re-evaluate Python 3.16 once numpy, scipy, scikit-image and pydantic-core publish cp316 wheels.
- Comment on issue #136 with the release that adds Python 3.15 support.
- Dependabot #152/#153 (react-konva 19, konva 10) are feature-level majors that need React 19; handle them in a separate change.
- Archive the change after review.
