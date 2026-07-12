# Spec delta: api-compatibility (api-surface-compat-gate)

## ADDED Requirements

### Requirement: Keyword surface stability
The test suite SHALL fail when a baselined public keyword or argument
disappears or changes its default. The public keyword surface (keyword names,
argument names and defaults) of the shipped Robot Framework libraries is
recorded in a reviewed baseline; additive changes pass.

#### Scenario: Accidental keyword removal is caught
- **WHEN** a refactor removes or renames a public keyword or argument
- **THEN** the surface test fails naming the missing item

#### Scenario: Additive evolution passes
- **WHEN** a new keyword or a new defaulted argument is added
- **THEN** the surface test passes without baseline changes
