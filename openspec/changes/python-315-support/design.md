# Design

## Context

See proposal.md (Why) for motivation. Constraints established by trials (2026-11-26, CPython 3.15.2, uv 0.13.10, Windows 11 native and Ubuntu 24.04 in Docker under WSL):

- **3.15 wheels**: every compiled runtime dependency publishes cp315 (or abi3) wheels for Windows and manylinux, except scikit-image. For scikit-image only `0.27.0rc1` has them.
- **Minimum 3.15 wheel releases**:
  - numpy 2.5.2
  - scipy 1.18.1
  - Pillow 12.3.0
  - pydantic-core 2.48.0
  - cachebox 6.1.2, which is pulled in by deepdiff ≥9.2. deepdiff 9.1 pins cachebox 5.x, which has no cp315 wheels.
- **Breaking majors that the current open floors let in**:
  - PyMuPDF 2.x hard-requires `onnxruntime` (no cp315 wheels).
  - OpenCV 5 removed the TensorFlow importer (`cv2.dnn.readNet` on the `.pb` EAST model fails) and changes `matchTemplate` results (3 movement-tolerance tests fail; they pass on 4.14).
  - pydantic-ai removed the `output_retries` keyword in 1.100 (late 1.x still accepts it with a deprecation warning, 2.x raises `TypeError`). In 2.x, `openai:` resolves to the Responses API instead of Chat Completions.
- **Locked candidate**: with the proposed specifiers, a refreshed lock resolves PyMuPDF 1.28.2, opencv-python-headless 4.14.0.94, numpy 2.4.6 (3.11) / 2.5.5 (3.12+), scikit-image 0.26.0 (<3.15) / 0.27.0rc1 (3.15), deepdiff 9.2.1, pydantic-ai-slim 2.75.0.
- **Linux results for that lock**: the full unit suite passes on 3.11 and 3.15. 3.12, 3.13 and 3.14 each had exactly one failure, caused by the testdata race described below, not by the interpreter. The full acceptance suite (117 tests: 105 pass, 12 skip) passes on all five interpreters.
- **Windows results**: the failure set on 3.15 is identical to HEAD on 3.13. 56 failures already exist there, unrelated to this change: dashboard tests embed backslash paths in Robot suite text.
- **Lock format**: a lock written by uv 0.13 is revision 5, and uv 0.12.23 reads it without complaint.
- **3.16** (0.0a2): none of 13 compiled dependencies publish cp316 wheels. Out of scope.
- **Parity gate**: `scripts/compare_wheel_contents.py` checks that the wheel's effective `Requires-Dist` per interpreter equals the poetry baseline, apart from an explicit allow-list of tightenings. It runs in the dashboard CI job and on publish.

## Goals / Non-Goals

**Goals:**
- Every interpreter from 3.11 to 3.15 installs from wheels only, with all extras, on Windows and Linux.
- Behavior is unchanged on every supported interpreter: same comparison results and the same LLM API surface. EAST OCR changes only through the defect fixes in D13.
- First-party code emits no new warnings on 3.15.
- The full unit suite passes on Windows as well as Linux, and CI keeps it that way.
- EAST OCR returns the expected words with both Tesseract 5.3 (Ubuntu) and 5.5 (Windows, UB Mannheim build).

**Non-Goals:**
- Adopting PyMuPDF 2.x or OpenCV 5.x. Adopting them needs an ONNX EAST model and a re-baselined template matcher, and is its own change.
- Running the Robot acceptance suites on Windows. They need Ghostscript, GhostPCL and ImageMagick, and stay on Linux.
- Third-party deprecations we cannot fix:
  - pylibdmtx uses `distutils.LooseVersion`, which needs setuptools; keep the existing setuptools dependency on 3.12+.
  - PyMuPDF's internal `typing.ByteString`, which is removed in 3.17.
  - scikit-image 0.27's `PendingSkimage2Change`. It is a `PendingDeprecationWarning` and hidden by default.

## Decisions

### D1 — Cap PyMuPDF `<2` and OpenCV `<5` rather than adapt
Both new majors break observable behavior: PyMuPDF 2 cannot be installed on 3.15, and OpenCV 5 cannot load EAST and changes movement results. Caps restore the behavior users have today on every interpreter. Both 1.28 and 4.14 ship abi3 wheels, so the caps do not limit future interpreters.
- *Alternative*: convert EAST to ONNX and re-validate matching under OpenCV 5. This is a larger, behavior-sensitive change with no benefit for 3.15, so it is deferred.

### D2 — Interpreter-scoped floors for 3.15, including an explicit scikit-image pre-release
Add `; python_version >= '3.15'` floors for numpy (`>=2.5.2`), scikit-image (`>=0.27.0rc1`) and deepdiff (`>=9.2.1`), and close the existing 3.13+ markers at `< '3.15'`. A specifier that names a pre-release opts that package into pre-releases under PEP 440. Verified:
- pip 26.3 with `--only-binary :all:` selects 0.27.0rc1 on 3.15.
- uv's default `if-necessary-or-explicit` does the same.

Older interpreters keep resolving scikit-image 0.26.0 final. Drop the 3.10-only branches.
- *Alternative*: a global `>=0.27.0rc1`. Rejected, because it forces a pre-release onto every interpreter.

### D3 — pydantic-ai: one code path for 1.40 ≤ v < 3
Use `Agent(identifier, output_type=..., retries=n)`. An int `retries` sets the output-validation budget in both majors. Verified on 1.40, 1.107 and 2.75: `retries=3` gives exactly 4 model calls on persistent malformed output.

Map the default provider to `openai-chat:<model>`. That prefix exists from 1.40 through 2.x and resolves to `OpenAIChatModel` on all three probed versions, so OpenRouter and other OpenAI-compatible base URLs keep using Chat Completions. Other providers keep their `<provider>:` prefix.

The floor is 1.40 because it is the oldest release that imports cleanly against current OpenTelemetry. 1.0–1.20 fail with `opentelemetry._events` missing, so nothing older is verifiably installable. The cap `<3` guards against the next unverified major.
- *Alternative A*: cap `<2`. That leaves 1.100+ deprecation warnings and blocks users on 2.x.
- *Alternative B*: detect the version at runtime and pick kwargs. That is more code for no gain, since the portable form exists.

### D4 — Banner suppression via `os.environ.setdefault`
Before the lazy `pydantic_ai` import in the LLM client, call `os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")`. Using `setdefault` respects an explicit user choice (spec: "User preference wins"). It has no effect on 1.x.

### D5 — Atomic EAST download
Download to a uniquely named temporary file in the target directory, then `os.replace` it onto the model path. Remove the temp file on any exception. On Windows, `os.replace` can raise `PermissionError` when another process holds the target open. In that case, if a model file already exists, discard our temp file and use the existing one, since the other process completed first.

The early-return check stays "file exists". It becomes correct because the final path only ever appears complete.
- *Alternative*: a cross-process file lock. That needs a new dependency (filelock) or platform code, and an atomic rename is sufficient.

### D6 — Tolerant subprocess decoding
In the capability check's `_get_version`, pass `errors="replace"` alongside `text=True`. Version strings are ASCII, and replacement characters in localized error text are harmless. This prevents `UnicodeDecodeError` in the `subprocess` reader thread under 3.15's default UTF-8 mode.
- *Alternative*: `encoding=locale.getencoding()`. It is also correct, but `errors="replace"` covers mixed-encoding tools as well.

### D7 — `fitz` → `pymupdf`, full rename
Replace `import fitz` with `import pymupdf` and update the `fitz.` references in `DocTest/` and `utest/`, including:
- the monkeypatch target in `test_correctness_fixes.py`
- the capability registry's `import_name`

`pymupdf` has been the canonical module name since 1.24.3, which is below our 1.26 floor.
- *Alternative*: `import pymupdf as fitz`. Rejected, because it keeps a misleading name for a cosmetic saving.

### D8 — Explicit `data_range=255` for SSIM
Both `structural_similarity` call sites receive uint8 grayscale images. Passing `data_range=255` is numerically identical to scikit-image 1.x's dtype-derived default, and is what skimage2 will require. The pending-deprecation warning itself is emitted unconditionally by 0.27's migration decorator and is not suppressed. It is hidden by default for end users.

### D9 — Raw regex literal in the PostScript grammar
Change `ws = ~"\s*"` to `ws = ~r"\s*"`. parsimonious evaluates grammar literals through `ast.literal_eval`, which is where the 3.15 SyntaxWarning comes from. The raw form yields the identical regex.

### D10 — Test isolation
The three tests that write `birthday_1080_date_id_10x.png` into `utest/testdata` write to `tmp_path` instead:
- `test_ocr.py`: the hires test
- `test_profiler.py`: the two big-file tests

The EAST download race in tests is fixed at its root by D5. No test-only workaround is needed.

### D11 — CI layout
- `test` matrix: 3.11, 3.12, 3.13, 3.14, 3.15.
- `smoke-no-ai`: stays on the oldest supported interpreter (3.11).
- `llm-tests`: moves from 3.13 to 3.15, so the live-LLM path runs on pydantic-ai 2.x with the newest stack.
- `lint`: stays on 3.12; ruff `target-version` becomes `py311`.
- New required `windows` job on `windows-latest`, Python 3.15:
  - install Tesseract via Chocolatey and add `C:\Program Files\Tesseract-OCR` to `PATH`
  - `uv sync --all-extras`
  - `scripts/audit_resolved_versions.py`
  - `uv run pytest utest -n auto --timeout=300`
- `dashboard` and `web` jobs stay on 3.13. The goal is to vary interpreters only where the matrix adds signal.

### D13 — EAST recognition: fix the defects, keep the region strategy
A benchmark scored end-to-end `get_text()` on the five EAST test images (14 expected words) with Tesseract 5.5 (Windows) and 5.3.4 (Linux). The current code scores 10/14 on Windows and 12/14 on Linux, at about 12.7 s.

Adopted changes, within the same contour-region pipeline:
1. Import `urllib.request` explicitly.
2. Drop OCR rows whose confidence is negative or whose text is empty. Confidences are ints; the old string comparison never matched. This alone halves runtime, because the refinement step stops re-OCR'ing empty structural rows.
3. Per region, estimate the line height from the median raw EAST box height. A region taller than 1.6× the median is recognized with `--psm 6` (block) instead of `--psm 7` (single line).
4. Compute vertical padding from the line height, not the region height.
5. If a region yields no alphanumeric word, retry once with a tight crop (60% of the line height horizontally, 25% vertically) and keep that result if it has alphanumeric content.
6. Delete the unused, broken `_extract_text`.

Result: 14/14 on Linux and 13/14 on Windows, at about 6.6–8.4 s. The remaining Windows miss is lowercase text on the DPI-upscaled `text_medium.png`, where EAST itself proposes no box; no test asserts it.

Alternatives benchmarked and rejected:
- NMS plus line grouping with per-line normalized scaling: 14/14 on Windows but 9–10/14 on Linux. It is engine-sensitive.
- Tight padding everywhere: splits words ("ABC DEFGHI").
- A mean-confidence fallback threshold: replaced a correct Linux reading with a confident wrong one.

### D14 — Mask editor: separator-agnostic folder derivation
The upload API keeps returning native paths, because browse results and saved masks files use them too. The frontend derives the folder at the last `/` or `\` and appends `masks.json` with the separator it found. The backend test derives the folder the same way.
- *Alternative*: return POSIX paths from the API. That would change every path the dashboard shows Windows users, and paths stored in existing masks and sidecars.

### D15 — Environment check ignores the Windows system `convert.exe`
A *fallback* binary that resolves into the Windows system directory (`%SystemRoot%\System32`, compared case-insensitively) is skipped. That executable is a disk-conversion tool that shares ImageMagick's legacy name. The primary `magick` name is unaffected.
- *Alternative*: check `-version` output for "ImageMagick". It runs a foreign binary with arguments it doesn't understand.

### D16 — POSIX paths in generated Robot suites
The dashboard tests interpolate test-data paths into Robot suite text with `.as_posix()`. Robot Framework and the library accept forward slashes on Windows, whereas backslashes are Robot escape characters (`\U` starts a Unicode escape, `\t` is a tab).

### D19 — e2e server output goes to a log file
`e2e/conftest.py` started `doctest-dashboard serve` with `stdout=PIPE` and never read it. Once the pipe buffer fills, uvicorn blocks on its next log write and every request hangs. On Windows this happened after two journeys (pages stuck on "Loading…"); on Linux the 64 KB buffer only postponed it. Output now goes to `dashboard-server.log` in the workspace, which is also shown when startup fails. The frontend's path joins move into `frontend/src/paths.ts` (`joinPath`, `siblingPath`), shared by the file browser and the mask editor (D14).

### D20 — Escape argument values passed through `BuiltIn.run_keyword`
`run_keyword` treats string arguments like test data: backslash escapes and `${...}` are resolved. WebCapture now escapes every data value with `robot.utils.escape`. `_named(name, value)` keeps the `name=` separator literal and escapes only the value, because `escape` also escapes `=`; `_value(...)` handles positional strings. Non-string arguments (element handles, `None`) pass through unchanged. The unit-test fake applies Robot's own `unescape`, so the tests assert what the keyword receives.
- *Alternative*: pass `Path` objects instead of strings. That only covers paths, not selectors or JavaScript, and depends on each library's type conversion.

### D18 — Per-interpreter environments for `invoke multipython`
The task syncs each interpreter into `.venv-multipython/<version>` (via `UV_PROJECT_ENVIRONMENT`, already covered by the `.venv*/` ignore rule) instead of re-syncing the active `.venv`. On Windows, uv cannot delete `.venv\Scripts` while invoke runs from it, and the old final "restore" sync silently changed the developer's interpreter.

### D12 — Expectation table and parity gate
- **Audit table**: `audit_resolved_versions.py` gains rows for 3.14 and 3.15 and drops 3.10. Every row also asserts the new caps (PyMuPDF `<2`, OpenCV `<5`, pydantic-ai `>=1.40,<3`). The 3.15 row asserts the 3.15 floors.
- **Parity gate**:
  - `compare_wheel_contents.py` evaluates 3.11–3.15.
  - Its allow-list gets the new intentional tightenings: the 3.15-only floors, PyMuPDF `<2`, OpenCV `<5`, and pydantic-ai `>=1.40,<3`.
  - The allow-list semantics are extended so a documented entry may *replace* a baseline floor with a stricter one. Example: `>=1` becomes `>=1.40,<3`.
  - Any other deviation still fails.

## Risks / Trade-offs

- [Pre-release on 3.15] When 0.28.0rc1 appears, the `>=0.27.0rc1` specifier lets pip/uv select it on 3.15 over a 0.27.x final. → Follow-up task: switch to `>=0.27.0` once final ships. Tracked in tasks.
- [Upper caps age] `<2`, `<5` and `<3` will eventually block security fixes that only land in new majors. → Dependabot keeps opening update PRs. Lifting each cap is a deliberate follow-up with a known test (EAST load, movement tolerance, LLM smoke).
- [pydantic-ai 2.x semantic drift] The offline stub test cannot prove that live providers behave the same. → The `llm-tests` job runs live on 3.15 with 2.x before release. The existing utest monkeypatch assertions move to the `retries` kwarg.
- [Atomic replace on network or odd filesystems] `os.replace` within one directory is atomic on NTFS and POSIX local filesystems. On exotic mounts it can degrade to a copy. → The temp file lives in the same directory as the model, and a failure falls back to the existing-file path (D5).
- [Windows CI cost/flakiness] The Windows job adds about 6–10 minutes and depends on the Chocolatey Tesseract package. → It runs in parallel with the Linux matrix. A Chocolatey outage fails the install step loudly instead of producing false test failures.
- [EAST output changes] EAST-based assertions may now see previously dropped lines and fewer garbage tokens. → Every benchmark image improved or stayed equal on both engines. Release notes mention it.
- [Dropping 3.10] Users on 3.10 stop receiving updates. → It is end of life. Release notes must name the last release that supports 3.10.

## Migration Plan

1. Land the specifier changes, the code fixes and a refreshed `uv.lock` (generated with uv ≥0.13) in one PR.
2. CI has to be green on the 3.11–3.15 matrix, plus windows-smoke, the dashboard parity gate, and a manual or optional run of llm-tests.
3. Release as a minor version, because of the 3.10 drop and the dependency caps. The release notes cover:
   - Python 3.15 support and the 3.10 drop
   - the caps
   - that LLM calls keep using Chat Completions
4. Rollback: revert the PR. The previous release stays installable on 3.10–3.14.
5. Follow-up: once scikit-image 0.27.0 final ships, replace the rc floor and refresh the lock.
