# Design: web-context-metadata

## Decisions

- **D1 Sidecar field**: `ComparisonResult.context: dict | None` (pydantic model
  additive default `None`). `ResultWriter.write(..., context=None)` stores it
  verbatim; `VisualTest.compare_images` pops `context` from kwargs like `name`.
- **D2 Adapter context**: `describe()` already exists (best-effort, never raises).
  Web keywords call it once per keyword invocation and pass it through.
- **D3 Split baselines**: `split_baselines_by` import param, comma-separated,
  allowed keys `browser`, `viewport`. Path: `baseline_dir / seg1 / seg2 /
  name.png` where segments are `sanitize_baseline_name(context[key])`; a missing
  context value falls back to `"unknown-<key>"` (still deterministic). The
  sidecar label (and default dashboard identity) becomes
  `"/".join(segments + [name])`.
- **D4 Dashboard**: `sidecar.py` model gains `context`; `ComparisonView` renders
  chips from `detail.sidecar_json.context` (browser, viewport, dpr as `@2x`,
  url with title tooltip). No DB/ingest changes — sidecar_json is already stored
  verbatim.

## Risks

- [describe() varies between libraries] → all fields optional; chips render only
  what exists.
- [Split key typo] → validated at import: unknown keys raise immediately with the
  allowed list.
