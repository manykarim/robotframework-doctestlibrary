# Web-App Visual Testing (Browser Library & SeleniumLibrary)

`DocTest.WebVisualTest` turns the comparison engine into a full visual-regression
workflow for web applications: it captures pages/elements through the web library
your suite already uses, manages named baselines automatically, retries flaky
captures, and feeds the review dashboard like every other DocTest comparison.

## 1. Setup

```bash
pip install robotframework-doctestlibrary
# plus ONE of the web libraries (whichever you already test with):
pip install robotframework-browser && rfbrowser init
pip install robotframework-seleniumlibrary

# or as one-liners via the convenience extras:
pip install robotframework-doctestlibrary[browser] && rfbrowser init
pip install robotframework-doctestlibrary[selenium]
```

`WebVisualTest` has no dependency on either library — it drives whichever one is
imported in your suite.

## 2. Quickstart

```robotframework
*** Settings ***
Library    Browser
Library    DocTest.WebVisualTest    baseline_directory=${EXECDIR}/visual_baselines

*** Test Cases ***
Checkout Page Looks Right
    New Page    https://shop.example.com/checkout
    Compare Page To Baseline    checkout

Price Table Is Stable
    Compare Element To Baseline    id=price-table    checkout-prices
```

With SeleniumLibrary the same keywords work (viewport captures — Selenium cannot
render the full scrollable page):

```robotframework
*** Settings ***
Library    SeleniumLibrary
Library    DocTest.WebVisualTest

*** Test Cases ***
Dashboard Header
    Open Browser    https://app.example.com    headlesschrome
    Compare Page To Baseline       app-dashboard    full_page=False
    Compare Element To Baseline    id=header        app-header
```

## 3. Baseline lifecycle

- Baselines are plain PNGs at `{baseline_directory}/{name}.png` — commit them to
  version control.
- **First run**: no baseline exists → the capture becomes the baseline and the test
  passes with a `Baseline created` warning. Review the created files before
  committing.
- **Update flows**: run with `--variable REFERENCE_RUN:True` to overwrite all
  baselines with fresh captures, or accept individual failures in the
  [review dashboard](dashboard.md) — accepting promotes the candidate over the
  baseline file, identical to any other DocTest comparison.
- Names are sanitized to `[A-Za-z0-9._-]`; they cannot escape the baseline
  directory.

## 4. Fighting flakiness

- **Capture retry**: when a comparison fails, the page/element is recaptured and
  recompared until `retry_timeout` (default `3s`) expires — late-settling pages
  pass, real regressions still fail with the final diff. Tune with
  `retry_timeout=`/`retry_interval=` import parameters, disable with `0`.
- **Browser Library captures** are taken with `disableAnimations=True` and
  `scale=css`, so animations, transitions and device-pixel-ratio differences
  (retina laptop vs CI runner) do not produce false diffs.
- **Ignore dynamic elements by locator** — the flagship anti-flake tool. Live
  bounding boxes of every match become comparison-time masks (re-read on each
  retry, translated into element space for element comparisons, DPR-scaled for
  Selenium). Baselines stay clean — nothing is burned into the screenshot:

```robotframework
Compare Page To Baseline       news       ignore_elements=id=clock;css=.ad-banner
Compare Element To Baseline    id=card    card    ignore_elements=css=.avatar
```

  Locators that match nothing are skipped with a log entry (a vanished ad is not
  an error). The applied masks are recorded in the sidecar, so the dashboard and
  mask editor show exactly what was ignored.

- **All Compare Images options** pass through the keywords: mask dynamic regions
  (`mask=`, `placeholder_file=`), tolerate movement (`move_tolerance=`), relax with
  `threshold=`, label with `name=`:

```robotframework
Compare Page To Baseline    checkout    threshold=0.05
...    mask={"page": "all", "type": "coordinates", "unit": "px", "x": 0, "y": 80, "width": 1280, "height": 60}
```

- **DOM-assisted classification** (`dom_analysis=True`): a semantic DOM snapshot
  (visible elements, roles, labels, values, normalized text) is stored beside each
  baseline and diffed on every comparison — the verdict (`identical` / `changed` +
  change summary) lands in the sidecar and dashboard. With
  `accept_rendering_only=True`, a failing pixel comparison whose DOM is unchanged
  (color/style-only difference) **passes with a warning** while the failure stays
  recorded for dashboard review — accept non-important changes without losing the
  audit trail. Semantic changes (text, structure, values) are never auto-accepted.
  Not suitable for canvas/SVG-rendered content (the DOM cannot see it):

```robotframework
Library    DocTest.WebVisualTest    dom_analysis=True    accept_rendering_only=True
```

- **Cross-browser tolerance**: `ignore_antialiasing=True` classifies differing
  pixels that sit on rendering edges in *both* images as anti-aliasing and
  excludes them from the verdict (measured: 5065 of 5070 differing pixels in a
  chromium-vs-firefox pair are edge noise; added elements and changed text keep
  hundreds of non-edge pixels and still fail). Combine with a small budget:

```robotframework
Compare Page To Baseline    home    ignore_antialiasing=True    max_diff_pixels=100
```

- **Pixel budgets for anti-aliasing**: `max_diff_pixels=` / `max_diff_ratio=`
  accept a bounded number of meaningfully-changed pixels (per-pixel intensity
  gate `pixel_intensity_threshold=`, default 20/255) — the right tool when
  baselines and CI render fonts slightly differently. Keep budgets small
  (≤ 0.1%):

```robotframework
Compare Page To Baseline    home    max_diff_ratio=0.001
```

- Keep the environment deterministic: fixed viewport (`New Context
  viewport={'width': 1280, 'height': 720}`), fixed test data, same OS for baseline
  creation and CI (containers recommended).

## 5. Optional AI review

With the `[ai]` extra installed and `DOCTEST_LLM_*` configured (see the README's
LLM section), the model can judge whether a detected difference matters — it
receives the diff renderings **plus the capture context** (browser, viewport,
URL) and the DOM-analysis verdict when `dom_analysis` is on:

```robotframework
Compare Page To Baseline    home    llm=True                     # advisory: decision logged & recorded
Compare Page To Baseline    home    llm=True    llm_override=True    # approving decision passes the test
```

Without the extra or configuration the options degrade gracefully (a warning, the
normal comparison result stands). Combine consciously with `retry_timeout=0` if
LLM latency per attempt is a concern.

## 6. Cross-browser & responsive baselines

Every web comparison records its **capture context** — library, browser, viewport,
device pixel ratio, URL — in the result sidecar; the dashboard shows it as chips
on the comparison. To keep one baseline per configuration:

```robotframework
Library    DocTest.WebVisualTest    split_baselines_by=browser,viewport
```

Baselines then live at `visual_baselines/{browser}/{viewport}/{name}.png` (e.g.
`chromium/1280x720/home.png`), and dashboard history/flaky tracking stays separate
per configuration because the comparison identity is config-qualified.

## 7. Reviewing failures

Enable sidecars and every web comparison shows up in the dashboard with
side-by-side/overlay/blink/swipe/highlight viewers, similarity grouping, history
and one-click baseline promotion:

```robotframework
Library    DocTest.WebVisualTest    result_json=true
```

```bash
doctest-dashboard serve
doctest-dashboard ingest results/output.xml
```

## 8. Examples

`examples/web-visual/` ships a page gallery (app dashboard with dynamic widgets,
checkout form, responsive product grid, SVG+canvas charts, long typographic
article) and `web_visual_demo.robot` — a commented, runnable tour of the
features. The same pages are exercised end-to-end in CI
(`atest/web/examples_vrt.robot`), including the edge cases: canvas changes that
DOM analysis cannot see, per-viewport baseline splitting, and capture-stability
on reflowing pages.

```bash
cd examples/web-visual && robot web_visual_demo.robot   # run twice: create, then compare
```

## 9. Keyword reference

| Keyword | Purpose |
|---|---|
| `Compare Page To Baseline    name    [full_page=True]    [**options]` | Capture the page (Browser: full scrollable page by default; Selenium: viewport with `full_page=False`) and compare against baseline `name`. |
| `Compare Element To Baseline    locator    name    [**options]` | Capture one element and compare against baseline `name`. |
| `Set Baseline Directory    path` | Switch the baseline directory mid-suite. |

Import parameters: `baseline_directory`, `web_library` (`Browser`/`SeleniumLibrary`
to force one when both are loaded), `retry_timeout`, `retry_interval`,
`split_baselines_by`, plus every
`VisualTest` parameter (`threshold`, `result_json`, `ocr_engine`, …).
