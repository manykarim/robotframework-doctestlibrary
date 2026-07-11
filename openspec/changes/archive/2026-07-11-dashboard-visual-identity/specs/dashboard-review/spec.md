# Spec delta: dashboard-review (dashboard-visual-identity)

## ADDED Requirements

### Requirement: Design token layer
All dashboard chrome colors, radii and shadows SHALL be defined as CSS custom properties in a single token layer; component styles SHALL reference tokens only.

#### Scenario: One place to restyle
- **WHEN** a token value changes
- **THEN** every component using that semantic role follows without per-component edits

### Requirement: Dark mode
The dashboard SHALL support a dark theme: initial value from the OS preference, a persistent user toggle in the top bar, and document renderings kept on neutral-light surfaces in both themes.

#### Scenario: Toggle persists
- **WHEN** a reviewer switches to dark mode and reloads the page
- **THEN** the dashboard renders dark without flashing light first

### Requirement: Empty states
A dashboard without ingested runs SHALL show an onboarding empty state naming the ingestion paths, and the mask editor without a loaded document SHALL point to Browse/Upload instead of rendering non-functional controls.

#### Scenario: Fresh install
- **WHEN** the runs page loads with zero runs
- **THEN** an onboarding card explains how to ingest results instead of an empty table
