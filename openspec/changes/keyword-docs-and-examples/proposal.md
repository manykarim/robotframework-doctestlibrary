# Proposal: keyword-docs-and-examples

## Why

New sublibraries and keywords are only real for users once they appear in the
published keyword documentation with examples and a getting-started path.
`WebVisualTest` is missing from the libdoc pipeline and the README's keyword-doc
list; the example gallery has no README of its own.

## What Changes

- **Libdoc pipeline**: `WebVisualTest` added to the `invoke libdoc` targets
  (versioned + unversioned HTML like the other libraries); generation verified
  in the test suite (libdoc runs cleanly, all public keywords present with
  documentation and examples in the output).
- **Library-level documentation**: `WebVisualTest` class docstring becomes a
  proper libdoc landing page — intro, requirements, quickstart example table,
  baseline lifecycle, links to the full guide.
- **README**: keyword-documentation list gains Web Visual Tests; the web section
  links the examples.
- **Examples README** (`examples/web-visual/README.md`): what the gallery shows,
  how to run the demo suite, where baselines land, how to review in the
  dashboard.

## Capabilities

### Modified Capabilities

- `web-visual-testing`: documentation completeness requirement.

## Impact

`tasks.py`, `DocTest/WebVisualTest.py` (docstrings only), README,
`examples/web-visual/README.md` (new), `utest/test_web_visual.py` (libdoc test).
