# Spec delta: web-visual-testing (keyword-docs-and-examples)

## ADDED Requirements

### Requirement: Documentation completeness
Every public WebVisualTest keyword SHALL carry libdoc documentation with an
example; the library SHALL be part of the keyword-documentation build and the
README's documentation index; the example gallery SHALL include run
instructions.

#### Scenario: Keyword docs published
- **WHEN** `invoke libdoc` runs
- **THEN** WebVisualTest.html is generated alongside the other libraries

#### Scenario: Documentation cannot rot silently
- **WHEN** a public keyword loses its documentation or example
- **THEN** the test suite fails
