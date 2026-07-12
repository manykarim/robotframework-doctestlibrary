# Design: web-vrt-examples

## Decisions

- **D1 One gallery, two consumers**: pages live in `examples/web-visual/pages/`;
  the user-facing demo suite and the CI acceptance suite both load them via
  `file://` — no server needed, no duplication.
- **D2 Determinism**: pages render fixed content by default; dynamic behavior
  (clock ticking, ad rotation, canvas mutation, validation states) is triggered
  explicitly through exposed JS hooks (`window.demo.*`) so tests control every
  state transition.
- **D3 Edge cases as tests**: the canvas page proves the DOM blind spot — a
  canvas-only change keeps the DOM verdict `identical`, so the acceptance test
  asserts pixels still fail it (and documentation warns against
  accept_rendering_only there).
- **D4 Example suite doubles as documentation**: heavily commented, mirroring the
  guide's sections.

- **D5 Stable capture**: every capture step takes two shots and repeats (≤3
  rounds) until two consecutive shots are byte-identical, returning the settled
  one — the Playwright approach. Found via the article page: first fullPage
  capture 778px tall, all later ones 780px (scrollbar reflow), which poisoned the
  baseline forever within the session.

## Risks

- [Gallery drift from docs] → the acceptance suite runs in CI; broken examples
  fail the build.
