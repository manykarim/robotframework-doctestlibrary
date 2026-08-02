# Proposal: library-level-template-detection

## Why

GitHub issue #127: `Image Should Contain Template` takes a per-call `detection` argument, but there is no way to set it once at library import — so suites that always want `sift` must repeat `detection=sift` in every call. The reporter expected the existing `movement_detection` import argument to cover this keyword too, and got no feedback when it didn't.

While designing this, a related pre-existing defect surfaced: `movement_detection=classic` is documented in the README but **rejected at import time**. The `Literal["template","orb","sift","text"]` type hint makes Robot Framework convert (and refuse) the value before library code runs, so `MOVEMENT_DETECTION_ALIASES = {"classic": "template"}` is dead code. Verified: `Library DocTest.VisualTest movement_detection=classic` fails with *"got value 'classic' that cannot be converted to 'template', 'orb', 'sift' or 'text'"*.

## What Changes

- New library import argument `template_detection` (values `template`, `orb`, `sift`, plus the `classic` alias) setting the default detection method for `Image Should Contain Template`.
- New keyword `Set Template Detection` to change it mid-suite, mirroring `Set Movement Detection`.
- `Image Should Contain Template`'s `detection` argument default becomes a `None` sentinel, resolving: explicit call argument → library `template_detection` → `template`. **Behavior for existing suites is unchanged** — anyone who does not set the new argument still gets `template` exactly as before. Only the libdoc-rendered default changes, so the reviewed signature baselines are regenerated.
- The `detection` value is now normalized (lower-cased, `classic` aliased) and validated up front with a clear error listing the supported values.
- **Fix `movement_detection=classic`**: widen the `__init__` type hint from `Literal[...]` to `str` so the documented alias reaches the alias map. Validation, error message and accepted values are unchanged (the library already validates and raises its own `ValueError`).
- `movement_detection` is deliberately **not** reused for template matching: it accepts `text`, which is meaningless for template matching, and reusing it would silently change behavior for existing `movement_detection` users.
- Docstring notes that under `sift`/`orb` the `threshold` argument is not consulted and the returned dict has no `confidence` key — relevant now that these can become a library-wide default.

## Capabilities

### New Capabilities
- `template-detection-config`: configuring the template-matching detection method at library level, its resolution order, validation, and aliasing.

## Impact

- Code: `DocTest/VisualTest.py` (`__init__`, new `Set Template Detection` keyword, `image_should_contain_template`).
- Baselines: `utest/keyword_signatures_baseline.json` and `scripts/keyword_surface_baseline.json` regenerated — the only expected diffs are the `detection` default and the added keyword.
- Docs: README gains a library-level template-detection example.
- Tests: new unit tests plus a Robot acceptance test (the only path that exercises Robot's import-argument type conversion, which is where the `classic` bug lives).
- `WebVisualTest` inherits the new setting through `**kwargs`; covered by an explicit test because unknown kwargs are silently swallowed.
