# Design: library-level-template-detection

## Context

`Image Should Contain Template` (DocTest/VisualTest.py:2069) hard-defaults `detection: str = "template"` and dispatches on it at :2165 (`template`) / :2221 (`sift` or `orb`) / :2298 (else → `ValueError`). The library already has the exact pattern to mirror for a library-level setting: `movement_detection` (validated in `__init__` at :191-198, changeable via `Set Movement Detection` at :1684, resolved per call in `compare_images` at :461-468 from an `Optional[str] = None` argument).

## Goals / Non-Goals

**Goals:** one place to configure the template detection method; unchanged behavior for suites that don't adopt it; documented aliases actually working.

**Non-Goals:**
- Making `movement_detection` drive template matching (see decision 3).
- Fixing that `detection=orb` actually runs SIFT (`:2221` routes both into `get_sift_keypoints_and_descriptors`). Pre-existing; documented, not changed here — a behavior change there deserves its own issue.
- Making `threshold` work under `sift`/`orb`, or unifying the two branches' return shapes. Documented as a caveat instead.
- Warning on unknown import kwargs (`__init__` ends in `**kwargs` and silently swallows typos — a real trap, but a separate change).

## Decisions

1. **A `None` sentinel is required, and it is behavior-preserving.** `"template"` is a legitimate explicit value, so a hard default makes "caller passed template" indistinguishable from "caller passed nothing" and no library-level setting could ever apply. Changing the default to `None` is the same pattern `compare_images` already uses for `movement_detection` (`Optional[str] = None`). Because resolution terminates at `"template"`, **no existing suite changes behavior** — only the libdoc default string changes, which is why the reviewed signature baselines are regenerated rather than the change being avoided.
2. **Plain `str`/`Optional[str]` hints, never `Literal`, for anything with aliases.** Robot converts import and keyword arguments from the type hint *before* library code runs, so a `Literal` hint rejects aliases like `classic` and makes the alias map unreachable. Verified live: `movement_detection=classic` raises *"cannot be converted to 'template', 'orb', 'sift' or 'text'"*. The new argument therefore uses `Optional[str]`, and `movement_detection`'s hint is widened to `str` to make the documented alias work. Validation quality does not regress — the library's own `ValueError` already lists supported values and is what `Set Movement Detection` has always produced.
3. **Do not reuse `movement_detection`.** It accepts `text`, which is not a template-matching method, so reuse would need a silent fallback for an invalid value; and any suite already setting `movement_detection=sift|orb` would silently switch template matching too. A dedicated setting satisfies the request in the issue's own title ("Add a detection library import argument") with zero blast radius.
4. **Separate validation helper, no shared refactor.** `__init__` and `Set Movement Detection` interpolate the *raw* argument into their error messages while `compare_images` interpolates the *normalized* one; a single shared helper cannot preserve all three messages byte-for-byte. The new setting gets its own small validator mirroring the movement one, and the movement code paths are left alone apart from the type hint.
5. **Empty string counts as "not provided."** `detection=${EMPTY}` is falsy, so `detection or self.template_detection or "template"` falls back rather than raising as it does today. That is the more useful behavior for a defaulted argument and is pinned by a test.
6. **Validate up front.** Resolution and validation happen at the top of the keyword, so a bad `detection` fails before file loading. This changes error *precedence* (previously a mismatched-crop or missing-file error surfaced first); no test depends on the old ordering.
7. **Acceptance test is mandatory, not optional.** Only a Robot suite exercises Robot's import-argument type conversion — the exact mechanism behind the `classic` bug. Python-level unit tests cannot catch a regression there.

## Risks / Trade-offs

- [Widening `movement_detection` to `str` loses libdoc's rendered enum of allowed values] → The docstring argument table already lists them, and validation still rejects bad values with a clearer, library-owned message. Worth it to make a documented feature work.
- [Both signature baselines must be regenerated, and the regeneration scripts rewrite the whole file, potentially absorbing unrelated drift] → Diff both JSONs and confirm the only changes are the `detection` default and the new keyword.
- [A user adopting `template_detection=sift` may be surprised that `threshold` is ignored and the result dict loses `confidence`] → Documented explicitly in the keyword docstring, since a library-wide default makes this easier to hit.
- [`WebVisualTest` forwards `**kwargs` to `VisualTest.__init__`, which silently swallows unknown keys, so a wiring regression would fail silently] → Explicit regression test asserting the setting arrives.

## Migration Plan

Single minor release; purely additive for users. Rollback = revert. No data or config migration.

## Open Questions

None. Naming (`template_detection`, parallel to `movement_detection`, rather than the issue's suggested `detection`) is a deliberate choice: the per-call argument stays `detection`, and a library-level argument literally named `detection` would read ambiguously next to it.
