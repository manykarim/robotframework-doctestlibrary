# Proposal: web-context-metadata

## Why

A web visual failure is only reviewable with its configuration: which browser,
which viewport, which DPR, which URL. Today the sidecar carries none of that, all
browsers share one baseline file, and cross-browser runs of the same page would
overwrite each other's baselines and pollute history.

## What Changes

- **Sidecar context** (additive, schema v1): comparisons carry a
  `context` object — for web comparisons `{library, browser, viewport,
  device_pixel_ratio, url}` from the capture adapter's `describe()`. `Compare
  Images` accepts `context=` and `ResultWriter` stores it.
- **Split baselines**: `split_baselines_by=` import parameter (comma-separated
  subset of `browser`, `viewport`) appends sanitized config segments to the
  baseline path — `visual_baselines/chromium/1280x720/home.png` — giving each
  configuration its own baseline. The sidecar label becomes the config-qualified
  name (`chromium/1280x720/home`) so dashboard history and flaky detection track
  per configuration.
- **Dashboard**: the comparison view renders context chips (browser, viewport,
  DPR, URL) when a sidecar carries context; sidecar model gains the additive
  `context` field.
- Tests: unit (context in sidecar via FakeAdapter run, split-path building,
  label qualification), dashboard model/API passthrough, and a real-browser
  acceptance test asserting the written sidecar contains the context.

## Capabilities

### Modified Capabilities

- `web-visual-testing`: capture-context and split-baseline requirements.
- `result-sidecar`: additive `context` field.
- `dashboard-review`: context display requirement.

## Impact

`DocTest/ResultWriter.py`, `DocTest/VisualTest.py` (context kwarg),
`DocTest/WebVisualTest.py`, `doctest_dashboard/models/sidecar.py`,
`frontend/src/ComparisonView.tsx`; tests + docs.
