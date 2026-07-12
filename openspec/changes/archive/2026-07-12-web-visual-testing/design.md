# Design: web-visual-testing

## Decisions

- **D1 Library shape**: `WebVisualTest(VisualTest)` — inherits `compare_images`,
  masks, thresholds, result sidecars, screenshots. Import params: those of
  `VisualTest` plus `baseline_directory` (default `visual_baselines`, resolved
  against `${EXECDIR}` when relative). `ROBOT_LIBRARY_SCOPE = 'TEST SUITE'` like
  the parent.
- **D2 Adapter protocol** (`DocTest/WebCapture.py`):
  `WebCaptureAdapter` with `capture_page(path, full_page)`, `capture_element(path,
  locator)`, `element_rects(locator) -> list[Rect css px]`, `device_pixel_ratio()`,
  `describe() -> dict` (browser/viewport/url — used by P3). Implementations
  `BrowserLibraryAdapter` / `SeleniumLibraryAdapter` call keywords exclusively via
  `BuiltIn().run_keyword` (`Browser.Take Screenshot`, `SeleniumLibrary.Capture Page
  Screenshot`, `Execute Javascript`, …) so DocTest never imports either package.
  `detect_adapter()` probes `get_library_instance("Browser")` then
  `("SeleniumLibrary")`; a clear error names both if neither is loaded. An explicit
  `web_library=` import parameter can force one (two libraries loaded at once).
- **D3 Capture defaults (from experiments)**: Browser page: `fullPage=True,
  scale=css, disableAnimations=True, log_screenshot=False, timeout=` library
  default; element: `selector=`, `scale=css`. Selenium: `Capture Page Screenshot` /
  `Capture Element Screenshot` into explicit paths. Captures land in
  `{OUTPUT_DIR}/doctest_web/{name}-{attempt}.png` — inside the robot output dir so
  log/report and dashboard uploads keep working.
- **D4 Baseline lifecycle**: baseline path = `baseline_directory / f"{name}.png"`;
  `name` is sanitized (`[^A-Za-z0-9._-]` → `_`, no path separators — traversal
  impossible). Missing baseline: capture is copied to the baseline path (dirs
  created), keyword logs WARN `Baseline created: <path>` and passes. `${REFERENCE_RUN}`
  truthy: capture overwrites the baseline and passes (same semantics as VisualTest).
  Otherwise: `self.compare_images(baseline, capture, **kwargs)` — every VisualTest
  option (masks, threshold, move_tolerance, name=, …) passes straight through; the
  sidecar label defaults to the baseline `name` for stable dashboard identity.
- **D5 Stabilization retry**: on `Failure` from `compare_images`, recapture +
  recompare while elapsed < `retry_timeout` (RF time string, default `3s`,
  `0` disables), sleeping `retry_interval` (default `500ms`) between attempts.
  The LAST failure is re-raised (its sidecar/screenshots are the ones that persist).
  Passing attempts short-circuit.
- **D6 Test strategy**: unit level — `FakeAdapter` (deterministic PNG generator)
  driving baseline-create/pass/retry/timeout paths + name-sanitization checks, no
  robot run needed. Acceptance — `atest/web/browser_vrt.robot` and
  `atest/web/selenium_vrt.robot` against a file:// deterministic page (no clock),
  executed by a new CI `web` job: `uv sync` + group `web-test` + `rfbrowser init
  chromium` + chrome-for-selenium (present on ubuntu-latest runners).
- **D7 Packaging**: `robotframework-browser` and `robotframework-seleniumlibrary`
  go into `[dependency-groups] web-test` only — never into project dependencies or
  extras; runtime adapters bind to whatever the user has installed.

## Risks

- [Browser/Selenium API drift] → adapters use stable public keywords only; the web
  CI job pins nothing so drift surfaces as a failing job, not a user bug.
- [Selenium full-page requests] → explicit error message pointing at viewport
  capture + documented limitation.
- [Retry masking real diffs] → retry only repeats the SAME comparison; a real diff
  fails every attempt and the final failure (with sidecar) is raised.
