# Spec delta: unified-packaging (packaging-extras-clarity)

## ADDED Requirements

### Requirement: Web convenience extras and base purity
The package SHALL provide `browser` and `selenium` convenience extras installing
the respective web automation library, and the base installation SHALL contain
no dashboard, AI or web-automation dependencies.

#### Scenario: Base install stays lean
- **WHEN** the wheel metadata is inspected
- **THEN** no dashboard/AI/web packages appear as unconditional dependencies

#### Scenario: One-liner web setup
- **WHEN** a user installs `robotframework-doctestlibrary[browser]`
- **THEN** robotframework-browser is installed alongside the library
