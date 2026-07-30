# Tasks: library-level-template-detection

## 1. Library-level setting

- [x] 1.1 Add `TEMPLATE_DETECTION_METHODS = {"template", "orb", "sift"}`, `TEMPLATE_DETECTION_ALIASES = {"classic": "template"}` and `TEMPLATE_DETECTION_DEFAULT = "template"` class constants
- [x] 1.2 Add `template_detection: Optional[str] = None` to `VisualTest.__init__` (plain `Optional[str]`, never `Literal`), normalize + validate it, store as `self.template_detection`
- [x] 1.3 Add the `Set Template Detection` keyword mirroring `Set Movement Detection` (same validation and log line style)

## 2. Keyword resolution

- [x] 2.1 Change `image_should_contain_template`'s `detection` argument to `Optional[str] = None` and resolve at the top of the keyword: explicit (non-empty) argument → `self.template_detection` → `template`, with normalization, aliasing and validation
- [x] 2.2 Update the keyword docstring: document the new resolution, and note that `threshold` is not consulted under `sift`/`orb` and that those branches return no `confidence` key

## 3. Fix documented movement_detection alias

- [x] 3.1 Widen `movement_detection`'s `__init__` type hint from `Literal[...]` to `str` so `classic` reaches the alias map; confirm the raised error message for bad values is unchanged

## 4. Baselines and docs

- [x] 4.1 Regenerate `utest/keyword_signatures_baseline.json` and `scripts/keyword_surface_baseline.json`; diff both and confirm the only changes are the `detection` default and the new `Set Template Detection` keyword
- [x] 4.2 README: document `template_detection` as a library import argument with an example

## 5. Tests

- [x] 5.1 Unit tests: default resolution unchanged when nothing is set; import argument applied; `Set Template Detection` applied and re-settable; explicit call argument wins; `${EMPTY}` treated as not provided; case-insensitivity; `classic` alias; invalid value error message
- [x] 5.2 Unit test: `movement_detection=text` does NOT change the template detection method
- [x] 5.3 Unit test: `WebVisualTest(template_detection="sift")` actually reaches `VisualTest` (guards the silent `**kwargs` swallow)
- [x] 5.4 Robot acceptance test exercising import-argument type conversion for both `template_detection=sift` and `movement_detection=classic`

## 6. Verification

- [x] 6.1 `uv run pytest utest -q -n auto --timeout=300` green (incl. signature-snapshot and package-parity tests); `uvx ruff check DocTest` green; new acceptance suite passes
