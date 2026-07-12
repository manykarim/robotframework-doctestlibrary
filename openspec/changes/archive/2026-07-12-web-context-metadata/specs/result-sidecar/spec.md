# Spec delta: result-sidecar (web-context-metadata)

## ADDED Requirements

### Requirement: Capture context field
The sidecar schema SHALL carry an optional additive `context` object describing
how the candidate was captured; absent for non-web comparisons.

#### Scenario: Additive compatibility
- **WHEN** a consumer reads a sidecar without `context`
- **THEN** parsing succeeds exactly as before
