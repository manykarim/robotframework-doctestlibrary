# Proposal: api-surface-compat-gate

## Why

The dashboard and web-testing work touched `VisualTest`, `PdfTest` and
`ResultWriter`. The 635-test core suite proves current behavior, but nothing
mechanically guarantees that a future refactor cannot silently drop or break a
public keyword or argument that existing DocTestLibrary users depend on.

## What Changes

- A recorded **public keyword surface baseline**
  (`scripts/keyword_surface_baseline.json`): keyword names, argument names,
  defaults and argument kinds for `VisualTest`, `PdfTest`, `PrintJobTests` and
  `WebVisualTest`, generated via robot's own LibraryDocumentation.
- A unit test asserting every baselined keyword still exists and every baselined
  argument keeps its name and default; **additions are allowed** (new keywords,
  new defaulted arguments), removals and signature breaks fail the build.
- A regeneration script (`scripts/update_keyword_surface.py`) for intentional,
  reviewed surface changes.

## Capabilities

### New Capabilities

- `api-compatibility`: mechanical backwards-compatibility gate over the public
  keyword surface.

## Impact

`scripts/` (baseline + generator), `utest/test_keyword_surface.py` (new). No
library code changes.
