# Proposal

## Why

Users cannot install the library on Python 3.15 (issue #136): the resolver falls back to building scikit-image 0.26 from source, which fails. Trials on Windows 11 and Ubuntu 24.04 (CPython 3.15.2) also showed that the open-ended dependency floors (`>=…`) now let in new major releases that break the library on *every* interpreter: PyMuPDF 2.x requires `onnxruntime` (no 3.15 wheels), OpenCV 5 dropped the TensorFlow importer the EAST text detector needs and changes template-matching results, and pydantic-ai ≥1.100/2.x removed the `output_retries` argument and switched `openai:` to the Responses API. Python 3.10 reached end of life in October 2026 and the scientific stack (numpy, scipy, Pillow) has already dropped it.

## What Changes

- **BREAKING**: `requires-python` raised from `>=3.10` to `>=3.11`; Python 3.10 is no longer supported or tested.
- Python 3.14 and 3.15 become supported and tested interpreters (classifiers, CI matrix, audit table, `invoke multipython`). Python 3.16 is explicitly out of scope: no compiled dependency publishes cp316 wheels yet.
- Dependency specifiers:
  - `PyMuPDF>=1.26,<2`
  - `opencv-python-headless>=4.8,<5`
  - `pydantic-ai-slim[openai]>=1.40,<3`
  - Python-3.15-specific floors: numpy `>=2.5.2`, scikit-image `>=0.27.0rc1` (a pre-release, allowed explicitly because it is the only release with 3.15 wheels), deepdiff `>=9.2.1`.
  - `uv.lock` refreshed to the newest versions these specifiers allow.
- LLM client works unchanged with pydantic-ai 1.40+ and 2.x: it uses the portable `retries=` argument, keeps OpenAI traffic on the Chat Completions API (`openai-chat:`), so OpenAI-compatible endpoints such as OpenRouter keep working, and suppresses pydantic-ai 2.x's Logfire console banner.
- The EAST model download becomes atomic. Today, concurrent processes (pytest-xdist workers, pabot) can read a half-written 96 MB file and fail with a parse error.
- EAST text recognition defects fixed. These were found while investigating OCR results that differ between Windows (Tesseract 5.5) and Linux (Tesseract 5.3):
  - The model auto-download relied on `urllib.request` being imported elsewhere.
  - Tesseract's structural rows were never filtered out (the `conf` check compared an int to the string `'-1'`). Each empty row was re-OCR'd, doubling EAST runtime.
  - Regions spanning several text lines were read in single-line mode, so whole lines (e.g. lowercase text) were lost.
  - Vertical padding was computed from the whole region height, so it reached into neighbouring lines and produced garbage tokens.
  - A crop that yields no text gets no tighter retry, so small labels in busy images were lost.
  - Unused, broken `_extract_text` code is removed.
- Web capture fix (all platforms, visible on Windows): `DocTest.WebVisualTest` passes screenshot paths, selectors and the DOM-walker JavaScript to Browser Library and SeleniumLibrary through `BuiltIn.run_keyword`, which parses string arguments as test data. Windows paths were mangled (`\new` became a newline), so every web capture failed on Windows. On every platform the DOM snapshot's whitespace regex `\s+` silently became `s+`, replacing the letter "s" in stored DOM text. Argument values are now escaped.
- Windows correctness fixes found during the Windows trials:
  - The mask editor derives the "save masks next to the upload" folder with `/` only, and the file browser joins a directory and a new file name with `/`. Both produce wrong or mixed-separator targets for Windows paths.
  - The environment check reports Windows' System32 `convert.exe` (a disk utility) as ImageMagick.
  - Dashboard tests embedded native Windows paths in Robot suite text, where `\` is an escape character.
  - The e2e fixtures start the dashboard server with an undrained stdout pipe. After a few KB of access log (sooner on Windows) the server blocks mid-request and the journeys hang. The e2e Robot suites also embedded native Windows paths.
  - The cross-browser web suite compared full-page captures, whose height differs by one pixel between Chromium and Firefox on Windows (font metrics). It now compares the fixed viewport.
  - `invoke multipython` re-synced the active `.venv`, which Windows cannot replace while the task runs from it. Each interpreter now gets its own environment.
- Deprecation and forward-compatibility fixes:
  - `import fitz` → `import pymupdf` (PyMuPDF 1.28 warns on every import).
  - A non-raw regex literal in the PostScript grammar becomes raw; it is a SyntaxWarning on 3.15 and will become a SyntaxError.
  - `structural_similarity` gets an explicit `data_range`, ahead of skimage2.
  - Subprocess output decoding in the capability check tolerates non-UTF-8 output. Python 3.15 defaults to UTF-8 mode, and localized Windows tools broke the reader thread.
- Test races fixed: unit tests no longer write generated images into the shared `utest/testdata` directory.
- Release 0.35.0: version bump and regenerated keyword documentation (`invoke libdoc`) for all five libraries, shipped with this change.
- The repository tracked `docs/AI.html` *and* `docs/Ai.html` (likewise `-0.28.0`). On case-insensitive file systems (Windows, macOS) both map to one file, so every checkout looked modified and regenerated docs landed in the wrong entry. The stale uppercase copies (superseded in October 2025, unreferenced) are removed. A test fails if two tracked paths ever differ only by case.
- CI:
  - The test matrix moves to 3.11–3.15.
  - The live-LLM job moves to the newest interpreter.
  - A Windows job is added on 3.15. It installs all extras (gating Windows wheel availability, the failure in #136), audits the resolved versions, and runs the full unit suite with Tesseract installed.

## Capabilities

### New Capabilities
- `llm-integration`: compatibility contract of the LLM client with the pydantic-ai runtime (supported major versions, provider API selection, retry budget, console hygiene).
- `ocr-text-detection`: EAST text detection — model provisioning (atomic, concurrency-safe download), the OpenCV runtime it depends on, and how detected regions are recognized (structural rows, multi-line regions, padding, empty-result retry).
- `environment-check`: the `Check DocTest Environment` capability report — genuine tool detection and robust version probing.

### Modified Capabilities
- `web-visual-testing`: a new requirement that capture arguments (paths, locators, scripts) reach the web library verbatim on every platform.
- `unified-packaging`: the "Version-conditional dependencies preserved as markers" requirement changes. The supported range becomes 3.11–3.15, with new 3.15 floors and new upper bounds for PyMuPDF, OpenCV and pydantic-ai.
- `uv-tooling`: "Multi-Python validation" and "CI, publish, and invoke tasks use uv" now cover 3.11–3.15 and add a Windows job that runs the unit suite.
- `mask-editor`: a new requirement that the "save next to upload" mask target is correct for native Windows paths.
- `toolchain-quality-gates`: "Test execution is parallel and hang-proof" gains test-isolation requirements (no writes to shared test data). It also requires the unit suite to pass on both Windows and Linux. A new requirement says first-party code must be free of interpreter-level deprecations (SyntaxWarnings, decoding failures) on supported Pythons.

## Impact

- **Packaging**: `pyproject.toml` (requires-python, classifiers, dependencies, `[ai]` extra, ruff target), `uv.lock`.
- **Code**:
  - `DocTest/llm/client.py`
  - `DocTest/Ocr.py`
  - `DocTest/CapabilityCheck.py`
  - `DocTest/PrintJobTests.py`
  - `DocTest/DocumentRepresentation.py`
  - `DocTest/PdfTest.py`
  - `DocTest/VisualTest.py`
  - `DocTest/WebCapture.py`
- **Frontend**: `frontend/src/MaskEditor.tsx`, `frontend/src/FileBrowser.tsx`, new `frontend/src/paths.ts`.
- **Tests**:
  - `utest/test_llm_support.py`
  - `utest/dashboard/conftest.py`, `utest/dashboard/test_visual_polish.py`, `utest/dashboard/test_upload.py`
  - `utest/test_capability_check.py`
  - `e2e/conftest.py`, `e2e/e2e_helpers.py`, `e2e/test_journeys.py`
  - `utest/test_web_visual.py`, `atest/web/xbrowser_vrt.robot`
  - `utest/test_ocr.py`
  - `utest/test_profiler.py`
  - the three utest modules that `import fitz`
- **Docs**: `docs/*-0.35.0.html` and the unversioned keyword docs. `docs/AI.html` and `docs/AI-0.28.0.html` are removed; the canonical `Ai*.html` files remain.
- **Tooling**: `scripts/audit_resolved_versions.py`, `scripts/compare_wheel_contents.py`, `tasks.py`, `.github/workflows/ci.yml`, `README.md`.
- **Users**:
  - Python 3.10 users must stay on the current release.
  - `[ai]` users pinned below pydantic-ai 1.40 must upgrade.
  - Users who relied on getting OpenCV 5 or PyMuPDF 2 through this library will get 4.x / 1.x instead.
  - `ocr_engine=east` results improve: lines that were previously dropped are now returned, fewer garbage tokens appear, and EAST runs about 2× faster. Text recognized from EAST regions can therefore differ from earlier releases.
- **Follow-up**: when scikit-image 0.27.0 final is released, replace the `>=0.27.0rc1` floor with `>=0.27.0`, so pip and uv stop preferring future pre-releases on 3.15.
