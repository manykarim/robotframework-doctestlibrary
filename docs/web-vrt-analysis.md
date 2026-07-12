# Web-App Visual Testing with Browser Library & SeleniumLibrary — Exploration & Proposals

Goal: make robotframework-doctestlibrary the go-to solution for visual regression
testing (VRT) of web applications driven by Browser Library (Playwright) and
SeleniumLibrary — combining the library's comparison engine, mask system and
review dashboard with first-class web capture workflows.

## 1. What the library offers today (relevant to web VRT)

- `Compare Images` — SSIM/pixel comparison of any two image files with thresholds,
  move tolerance, partial-image detection, coordinate/area/text-pattern masks,
  watermark handling, optional LLM assist.
- Baselines are plain files; `REFERENCE_RUN` promotes candidates; the dashboard
  reviews failures (grouping, history, batch accept) and promotes baselines.
- `result_json` sidecars (v1.1: highlighted renderings, thumbnails, `name=` labels,
  facets, run manifest) feed the dashboard.
- **No web integration**: users must capture screenshots themselves, invent baseline
  path conventions, pre-create baselines manually, and there is no notion of
  browser/viewport/DPR context, element comparison, or capture retry.

## 2. Capture surface of the two target libraries (verified from source)

### Browser Library (`Take Screenshot`, keywords/browser_control.py)
- `selector=` (element screenshot), `fullPage=True`, `crop=` (BoundingBox dict),
  `disableAnimations=True` (CSS animations/transitions/web-animations stopped),
  `mask=`/`maskColor` (selector-based pink overlays burned into the image),
  `scale=css|device` (DPR normalization at capture!), `fileType`, `quality`,
  `return_as=path|bytes|base64`, `timeout`.
- `Get BoundingBox    selector` → `{x, y, width, height}` in CSS pixels.
- `Set Viewport Size`, `Get Viewport Size`, `Evaluate JavaScript` (DPR readable via
  `window.devicePixelRatio`).

### SeleniumLibrary (keywords/screenshot.py, element.py)
- `Capture Page Screenshot` (viewport only — no native full-page), `Capture Element
  Screenshot    locator`, EMBED/BASE64 modes.
- `Get Element Size`, `Get Horizontal/Vertical Position` (CSS pixels);
  `Execute Javascript` for DPR/scroll/rects.
- No masking, no animation control, no DPR scaling → the engine must do the work.

Implication: **capture-time masking (Browser's pink boxes) pollutes baselines and is
Playwright-only.** Engine-side masks from element rectangles work identically for
both libraries, keep baselines clean, and stay reviewable/editable in the dashboard
and mask editor — that is this library's structural advantage.

## 3. The industry feature bar

| Capability | Playwright `toHaveScreenshot` | Percy/Applitools | BackstopJS | WatchUI (RF) | doctest today |
|---|---|---|---|---|---|
| Auto baseline create on first run | ✔ | ✔ (cloud) | ✔ | ✖ | ✖ |
| Capture-retry until stable | ✔ | ✔ | ✖ | ✖ | ✖ |
| Mask dynamic elements by selector | ✔ | ✔ | ✔ | ✖ | (coords/pattern only) |
| Pixel-ratio tolerance (`maxDiffPixels`/ratio) | ✔ | perceptual AI | ✔ | threshold | SSIM threshold + regions |
| Per browser/viewport baselines | ✔ (auto-suffix) | ✔ | ✔ (viewports) | ✖ | ✖ |
| Element-level comparison | ✔ | ✔ | ✔ | partial | ✖ |
| Review UI with accept/reject | ✖ (CLI update) | ✔ (SaaS) | ✔ (report) | ✖ | ✔ (dashboard) |
| Cross-run history / flaky detection | ✖ | ✔ | ✖ | ✖ | ✔ (dashboard) |
| Similarity grouping of failures | ✖ | ✔ | ✖ | ✖ | ✔ (dashboard) |
| Works offline / self-hosted | ✔ | ✖ | ✔ | ✔ | ✔ |

Known flake sources the design must address (industry consensus): dynamic content
(dates/ads/avatars), animations/carets, font/AA rendering differences across OS,
DPR differences between local and CI, network/lazy-load timing.

## 4. Gap analysis → proposals

The dashboard (review, groups, history, flaky, mask editor, baselines-as-files)
already exceeds every open-source competitor. What is missing is the **capture and
baseline workflow between the web library and the engine**. Four changes, applied
sequentially:

### P1 `web-visual-testing` — capture-agnostic baseline keywords (core UX)
New keywords (working with whichever of Browser/SeleniumLibrary is imported,
auto-detected via `BuiltIn.get_library_instance`):
- `Compare Page To Baseline    name    [options]`
- `Compare Element To Baseline    locator    name    [options]`
Behavior: capture (Browser: `fullPage`, `disableAnimations` defaults; Selenium:
viewport/element) → resolve baseline path `{baseline_directory}/{name}.png` →
if baseline missing: save capture as baseline, PASS with WARN + sidecar marked
`baseline_created` (reviewable) → else `Compare Images` with all existing options
passed through (masks, thresholds, move tolerance, `name=`…).
Stabilization: on failure, recapture and recompare until `retry_timeout` expires
(Playwright-style auto-retry) — flake from timing dies here, real diffs remain.

### P2 `selector-ignore-masks` — mask dynamic elements by locator
`ignore_elements=<locator>[;<locator>…]` on the new keywords: element bounding
boxes (all matches per locator) are read from the active web library, scaled by
DPR to device pixels, converted to coordinate masks for the engine, and recorded
in the sidecar (visible in viewer/mask editor). Baselines stay clean — masks are
comparison-time, not capture-time.

### P3 `web-context-metadata` — configuration-aware baselines & dashboard context
Capture context `{browser, viewport, device_pixel_ratio, url}` recorded in the
sidecar (`context` field, additive schema). Optional `split_baselines_by=browser,viewport`
appends config segments to the baseline path (per-browser/viewport baselines).
Comparison identity uses `name::<label>` (existing) so history/flaky tracking works
per baseline name; dashboard shows context chips on the comparison view.

### P4 `pixel-diff-tolerance` — AA-tolerant acceptance options
`max_diff_pixels=` / `max_diff_ratio=` acceptance criteria on visual comparison
(counted on the thresholded diff), tuned for cross-OS anti-aliasing noise as an
alternative/complement to SSIM `threshold`.

Docs & examples (README section + docs/web-visual-testing.md + example suites)
are folded into each change; e2e coverage runs real Browser/Selenium suites
against a local static page served in tests.

## 5. Experiment results (verified locally, robot runs against a real page)

Browser Library 20.0.0, chromium headless, 1280×720 viewport, local HTML page:

| Capture | DPR=1 | DPR=2 (`deviceScaleFactor=2`) |
|---|---|---|
| viewport screenshot | 1280×720 | — |
| `fullPage=True` (default `scale=device`) | 1280×1694 | **2560×3388** |
| `fullPage=True scale=css` | 1280×1694 | **1280×1694** (normalized!) |
| `selector=id=clock` element shot | 216×35 | 432×70 (device px) |
| `Get BoundingBox id=clock` | `{x:0, y:119.875, w:216, h:34}` | identical (always CSS px, fractional) |

SeleniumLibrary 6.x, google-chrome headless (Selenium Manager auto-fetched the driver):
- `Capture Page Screenshot` → 1280×577: **`Set Window Size` includes browser chrome** — the
  real viewport must be read via `window.innerWidth/innerHeight`.
- `Capture Element Screenshot id=clock` → exactly 216×34 (CSS px at DPR=1).
- `getBoundingClientRect` via `Execute Javascript` returns fractional CSS px like Browser.

Conclusions baked into the designs:
1. Browser page captures use `scale=css` → DPR-independent baselines and masks usable
   in CSS px directly. Selenium captures are device px → rects must be scaled by
   `window.devicePixelRatio`.
2. Bounding boxes are fractional → masks round outward (floor origin, ceil extent).
3. Adapters must invoke keywords through `BuiltIn().run_keyword` — no imports of
   Browser/Selenium in DocTest code, hence no new hard dependencies.
4. Selenium Manager makes CI setup trivial where a chrome binary exists; Browser
   needs `rfbrowser init` (node) in the web CI job.

## 6. Risks / constraints

- robotframework-browser needs `rfbrowser init` (node runtime) in CI for tests —
  gate web-e2e tests to a dedicated CI job; unit-test capture adapters against
  fakes, acceptance-test against the real libraries.
- SeleniumLibrary full-page capture is not native — P1 scope: viewport + element
  for Selenium, full page only where the driver supports it (documented).
- DPR: Browser can capture with `scale=css`; Selenium cannot — engine-side rect
  scaling (P2) must read `window.devicePixelRatio` per session.
- Keep the new dependency optional: `[web]`-style extra or docs-only dependency on
  the user's own Browser/Selenium install (the adapters import lazily — no hard
  dependency in the base package).

## 7. Reliability exploration (round 2 — verified experiments)

Goal: easy & reliable web VRT — accept non-important changes, survive cross-browser
rendering, reduce false failures with DOM analysis and optional AI.

**Experiment A — semantic DOM snapshots** (JS walker over visible elements:
tag/role/aria-label/value/href/src/alt + normalized text; runs via
`Evaluate JavaScript` / `Execute Javascript`, so it works for Browser AND Selenium):
- CSS-only change (header color): visual diff, snapshot **identical** → classifiable
  as "rendering-only".
- Text change: snapshot differs → semantic change detected.
- chromium vs firefox on the same page: snapshots **byte-identical** across engines.

**Experiment B — cross-browser pixel noise** (demo page, 900×626, chromium vs
firefox, scale=css): 1.85% of pixels differ at all, 0.90% differ by >20 intensity —
far too much for raw pixel budgets without hiding real changes.

**Experiment C — anti-aliasing classifier** (pixelmatch-inspired: a differing pixel
is AA if it sits on a local 3×3 intensity edge (range > 60) in BOTH images):
| Pair | differing (>20) | classified AA | residual real |
|---|---|---|---|
| chromium vs firefox (noise only) | 5070 | 5065 | **5 (0.0009%)** |
| new solid element added | 12141 | 0 | 12141 → caught |
| "Hello World" → "Hollo Warld" | 1573 | 1134 | 439 → caught |

Conclusion: `ignore_antialiasing` + a tiny pixel budget passes cross-engine noise
while text/content changes retain hundreds of "real" pixels and still fail. DOM
analysis is the semantic backstop; both compose with the existing LLM assist.

### Round-2 proposals
1. `antialiasing-tolerance` — `ignore_antialiasing=` engine option; AA pixels do
   not count toward failure or pixel budgets; cross-browser acceptance test
   (chromium baseline vs firefox candidate must pass).
2. `dom-assisted-comparison` — adapter `dom_snapshot()`; `dom_analysis=` stores a
   `.dom.json` beside the baseline, classifies failures as rendering-only vs
   semantic (sidecar facet), `accept_rendering_only=` passes pure-rendering diffs.
3. `web-ai-review` — verified LLM passthrough for web keywords (optional [ai]),
   auto-enriched prompt with capture context + DOM verdict.
4. `web-vrt-examples` — example .html gallery (dynamic content, responsive,
   forms, SVG/canvas, typography, long pages) + end-to-end example suites
   exercising every reliability feature; living documentation.
