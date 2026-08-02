# Design: fix-oversized-log-screenshots

## Context

`add_screenshot_to_log` (DocTest/VisualTest.py:2390-2439) is the only producer of screenshot HTML in the codebase. It computes one style string and interpolates it into three emit branches: base64 JPEG, base64 PNG, and the default file-link branch. `width:50%` is the single occurrence of that literal anywhere in the repo — no test, doc, dashboard or frontend file asserts on it.

## Goals / Non-Goals

**Goals:** stop upscaling small screenshots; leave every other rendering byte-identical; lock the behavior with a test.

**Non-Goals:**
- Changing the `original_size=True` branch. It was introduced deliberately (commit 3150d8d, "Embed template screenshots in their original (small) size") so template screenshots keep natural size; bounding it would reverse that.
- Solving high-DPI crops that legitimately exceed half the column (a `dpi=600` crop can still be clamped). `max-width` is the fix for *upscaling*, not for downscaling large rasters — out of scope.
- Any change to comparison logic, sidecar output, or keyword signatures.

## Decisions

1. **`width:50%` → `max-width:50%`** at the single producer line. CSS semantics: `width:50%` always resolves to half the containing block; `max-width:50%` resolves to `min(intrinsic, 50%)`. For any image ≥ half the column the two are identical, so large screenshots are unaffected by construction.
2. **One-line change, no refactor.** All three emit branches share the `{img_style}` placeholder, so they change together with no risk of drift.
3. **Test via `robot_logger` patching, not `caplog`.** `robot_logger` is `robot.api.logger` (imported at VisualTest.py:11); outside a Robot run it routes to the Python logger named `RobotFramework`, not `DocTest.VisualTest`, so `caplog` on the module logger captures nothing. The test patches `DocTest.VisualTest.robot_logger` and asserts on the emitted HTML string.
4. **Cover the shipped default path.** Defaults are `screenshot_format="jpg"` and `embed_screenshots=False` (the file-link branch). The test exercises the default file-link path plus the embedded path, and pins the `original_size=True` branch as unchanged.

## Risks / Trade-offs

- [Robot's `log.html` renders messages inside an auto-layout table cell, where percentage `max-width` against a shrink-to-fit container is less crisply specified than percentage `width`] → The issue reporter empirically verified the improvement with before/after screenshots in the real log viewer, and the maintainer confirmed the annoyance; browsers resolve this consistently in practice for replaced elements with intrinsic dimensions.
- [A small image in the base64 branches has no click-through link, so it can no longer be opened full size] → It renders at natural size now, which is the point; the default (file-link) branch keeps its `target="_blank"` wrapper.
- [The no-upscaling invariant cannot be asserted from the style string alone, since both variants emit one shared string regardless of image size] → The test pins the exact style strings per branch, which is the only observable contract; the size-dependent rendering is the browser's job.

## Migration Plan

Single patch release. Rollback = revert the one-line change.

## Open Questions

None.
