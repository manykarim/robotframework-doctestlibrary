# Design: api-surface-compat-gate

## Decisions

- **D1 Source of truth**: `robot.libdocpkg.LibraryDocumentation` — the same
  machinery users' editors and docs use, so what we gate is exactly what users
  see. Snapshot per library: `{keyword: [{name, default_repr, kind}]}`.
- **D2 Compat rule**: baseline keyword missing → fail; baseline argument missing
  from a keyword → fail; a baseline argument's default changed → fail; anything
  new → allowed. Defaults compared by repr string.
- **D3 Regeneration**: `uv run python scripts/update_keyword_surface.py`
  rewrites the baseline — the diff then shows up in review.

## Risks

- [Default reprs differ across Python versions] → reprs normalized to str; CI
  runs the gate on every matrix version, keeping the baseline honest.
