# Proposal: dom-assisted-comparison

## Why

Pixels alone cannot say whether a visual difference *means* anything. The verified
experiment shows a semantic DOM snapshot (visible elements: tag/role/aria-label/
value/href/src/alt + normalized text, collected via in-page JS in BOTH web
libraries) is invariant under CSS-only changes, changes when content changes, and
is byte-identical across chromium/firefox. That gives visual comparison a semantic
second opinion: a failing pixel comparison whose DOM is unchanged is a
rendering-only difference — reviewable, and optionally acceptable automatically.

## What Changes

- Adapters gain `dom_snapshot(locator=None)` (page or element scope) running the
  shared JS walker through `Evaluate JavaScript` / `Execute Javascript`.
- `dom_analysis=True` (import parameter or per-call option) on the web keywords:
  - a `{name}.dom.json` snapshot is stored beside the PNG baseline (created with
    it, refreshed on every visual PASS so dashboard-accepted baselines self-heal);
  - before comparing, the current snapshot is diffed against the stored one and
    the verdict (`identical` / `changed` + compact change summary via deepdiff /
    `missing-baseline`) is recorded in the sidecar context → visible in the
    dashboard.
- `accept_rendering_only=True`: a FAILING visual comparison whose DOM verdict is
  `identical` passes with a WARN — the sidecar still records the failure for
  dashboard review (accept non-important changes without losing the audit trail).
  Any DOM change keeps the failure.
- Tests: unit (FakeAdapter snapshot scripting: create/verdicts/accept/refresh),
  acceptance in browser_vrt.robot (CSS-only change accepted, text change not).

## Capabilities

### Modified Capabilities

- `web-visual-testing`: DOM-assisted classification and rendering-only acceptance.

## Impact

`DocTest/WebCapture.py` (walker + adapter methods), `DocTest/WebVisualTest.py`
(analysis flow), `utest/test_web_visual.py`, `atest/web/browser_vrt.robot`, docs.
