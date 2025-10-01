# OCRS Integration Plan

## 1. Current OCR usage and packaging constraints
- `DocTest/DocumentRepresentation.py` currently limits OCR engines to `"tesseract"` and `"east"`, calling `pytesseract.image_to_data` / `image_to_string` for both direct OCR and EAST post-processing workflows.【F:DocTest/DocumentRepresentation.py†L37-L108】【F:DocTest/DocumentRepresentation.py†L531-L569】
- `DocTest/Ocr.py` relies on `pytesseract` to recognize text within EAST-detected regions, again via `image_to_data` for structured outputs and `image_to_string` for refinements.【F:DocTest/Ocr.py†L5-L119】【F:DocTest/Ocr.py†L188-L205】
- The default OCR engine is configured as `"tesseract"` in `DocTest/config.py`, and the Python dependency list includes `pytesseract` in `pyproject.toml`.【F:DocTest/config.py†L1-L10】【F:pyproject.toml†L24-L40】
- Test coverage assumes Tesseract availability. Running `poetry run pytest` shows widespread `TesseractNotFoundError` failures plus barcode tests that expect Tesseract-provided text, confirming that replacing the OCR backend will require coordinated test updates.【783cff†L1-L144】

## 2. OCRS capabilities and distribution status
- The `ocrs` project provides a Rust library and CLI capable of text detection (word bounding boxes), layout analysis, and text recognition, delivering both raw text and structured geometry through the `OcrEngine` API.
- OCRS outputs characters with bounding boxes (`TextChar`, `TextWord`, `TextLine`) enabling reconstruction of word-level coordinates needed for mask generation.
- OCRS has no published Python wheel on PyPI (`pip index versions ocrs` returns no matches), so adopting it requires building a Rust extension or vendoring a CLI binary during packaging.【fd94ab†L1-L2】
- Model files are fetched on first use (into `~/.cache/ocrs`), meaning the runtime must ensure network availability or provide a prefetch hook when running offline.

## 3. Packaging strategy for Poetry + Rust
1. **Select build approach**: Evaluate integrating a Rust extension built with `maturin` (preferred for PEP 517 wheels) versus shelling out to the CLI.
   - `maturin` can coexist with Poetry by keeping Poetry for dependency resolution while using `maturin` to build the Rust extension as part of the wheel (requires switching `build-system` to `maturin` or using `maturin`'s `pep517` backend and invoking Poetry via plugin hooks).
   - If direct Rust integration proves too complex, fallback is vendoring the `ocrs-cli` binary and invoking it via `subprocess`, but this introduces platform-specific binaries and complicates wheel builds.
2. **Prototype build**:
   - Create a `rust/ocrs_py` crate wrapping minimal APIs (load models, run detection + recognition) using `pyo3`.
   - Configure `pyproject.toml` with a `[tool.maturin]` section pointing to the Rust crate and add `maturin` to the build requirements while keeping Poetry-managed dependencies for runtime (requires updating `build-system.requires` and CI build instructions).
   - Ensure the resulting wheel bundles the Rust extension (verify via `poetry build`). Document cross-platform compilation expectations (need Rust toolchain in CI, and possibly `manylinux`/`macOS` builds via `maturin`’s `--release` builds).
3. **Dependency updates**:
   - Add a Python shim package (e.g., `DocTest/ocrs_adapter.py`) to load the Rust extension, manage model downloads, and expose Pythonic methods returning data shaped like pytesseract outputs.
   - Mark `pytesseract` as optional or remove it once parity is confirmed; keep as fallback during migration to avoid breaking deployments lacking Rust tooling.

## 4. Application refactor plan
1. **Introduce OCR abstraction**:
   - Add a new internal interface (e.g., `DocTest/ocr_engines.py`) defining a base class for OCR engines returning dictionaries with `text`, `left`, `top`, `width`, `height`, `conf` arrays to match existing expectations.
   - Implement adapters for existing Tesseract logic and the new OCRS backend so `DocumentRepresentation.Page.apply_ocr` can delegate without engine-specific branches.【F:DocTest/DocumentRepresentation.py†L37-L108】
2. **OCRS engine implementation**:
   - Within the adapter, call the Rust extension to retrieve line and word data, convert bounding boxes (rotated rectangles) to axis-aligned boxes, and populate confidence scores (OCRS provides per-character probabilities; if unavailable, approximate or expose detection scores).
   - Implement optional configuration for allowed characters and recognition options mirroring existing `tesseract_config` features (e.g., digits-only mode) using OCRS `allowed_chars` and decode settings.
   - Handle model caching, exposing configuration to control cache directories and offline usage.
3. **Update EAST workflow**:
   - Replace direct pytesseract calls in `DocTest/Ocr.py` with the OCR abstraction so EAST text extraction uses the same backend for recognition and ensures consistent bounding box formatting.【F:DocTest/Ocr.py†L84-L119】
4. **Mask generation adjustments**:
   - Update `_process_pattern_ignore_area_from_ocr` and related helpers to rely on OCRS-provided tokens, ensuring they still populate `pixel_ignore_areas` correctly.【F:DocTest/DocumentRepresentation.py†L531-L569】
   - Validate the new boxes align with EAST outputs (may require resizing or additional geometry conversions from OCRS rotated rectangles).
5. **Configuration and defaults**:
   - Add `"ocrs"` as a valid engine constant, switch `OCR_ENGINE_DEFAULT` to `"ocrs"` once tests pass, and provide compatibility warnings for deprecated `"tesseract"` usage.【F:DocTest/config.py†L1-L10】
   - Extend public APIs (`VisualTest`, Robot Framework keywords) to document the new engine and maintain backwards compatibility options.【F:DocTest/VisualTest.py†L50-L227】
6. **Testing updates**:
   - Replace existing OCR unit tests to validate OCRS outputs against expected text/masks, accounting for potential differences in recognition quality.
   - Mock or adapt tests that previously expected `TesseractNotFoundError` to instead skip when OCRS models are unavailable, and add integration tests covering model download caching.
   - Update barcode tests if they depended on Tesseract-populated ignore areas; ensure ocRS-based ignore areas still allow barcode detection.

## 5. Wheel verification & distribution
1. **Build validation**: Run `poetry build` (or `maturin build`) to ensure the Rust extension is embedded in the wheel and inspect the `.whl` for the compiled shared library.
2. **Cross-platform CI**: Document requirements for building wheels on Linux, macOS, Windows (Rust toolchain + `maturin`). Ensure automation downloads models during tests or uses cached artifacts.
3. **Runtime verification**: Extend smoke tests (Robot Framework and Python) to run `VisualTest.get_text` and mask generation end-to-end using OCRS, confirming no reliance on system Tesseract installations.
4. **Fallback strategy**: Decide whether to keep `pytesseract` as an optional dependency for environments where Rust builds are infeasible, including configuration toggles and documentation.

## 6. Documentation & migration notes
- Update README and API docs to describe the new OCR backend, installation prerequisites (Rust extension, model caching), and environment variables for controlling OCRS behavior.
- Provide migration guidance for users who previously configured Tesseract options, mapping them to OCRS equivalents or documenting unsupported features.
- Highlight performance expectations and known limitations (Latin alphabet only, early preview status) with references to the OCRS upstream roadmap.

