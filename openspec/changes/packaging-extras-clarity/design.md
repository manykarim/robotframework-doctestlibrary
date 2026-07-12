# Design: packaging-extras-clarity

## Decisions

- **D1 Extras semantics**: `[browser]`/`[selenium]` are *convenience* extras —
  unpinned single-package pulls, documented as equivalent to installing the web
  library yourself. `WebVisualTest` never imports them; adapters bind at runtime.
- **D2 `[all]` unchanged** (ai+dashboard): pulling a ~100MB node-backed browser
  stack implicitly would hurt more users than it helps; the matrix documents it.
- **D3 Gates**: the parity comparator validates effective base dependencies —
  extras additions must appear in the baseline metadata (extended, reviewed);
  a metadata test asserts Provides-Extra = {ai, dashboard, all, browser, selenium}
  and that base Requires-Dist has no fastapi/uvicorn/pydantic-ai/browser/selenium
  entries without an extra marker.

## Risks

- [Comparator rejects new extras] → baseline update is part of the change and
  reviewed via the diff.
