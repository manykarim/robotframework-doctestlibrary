# Design: keyword-docs-and-examples

## Decisions

- **D1 Docstring format**: same libdoc conventions as VisualTest (argument
  tables, `|` example rows) — consistent rendering on the published site.
- **D2 Verification**: a unit test builds the libdoc model
  (`LibraryDocumentation("DocTest.WebVisualTest")`) and asserts every public
  keyword has non-empty documentation containing at least one example row —
  documentation cannot silently rot.
- **D3 Examples README**: task-oriented (run → inspect baselines → break →
  review), mirroring the demo suite's order.

## Risks

- [Docs drift from behavior] → the demo suite runs in CI (previous change);
  the libdoc test pins presence, CI pins truth.
