# Spec delta: dashboard-review (web-context-metadata)

## ADDED Requirements

### Requirement: Capture context display
The comparison view SHALL display capture context (browser, viewport, device
pixel ratio, URL) when the sidecar provides it.

#### Scenario: Web comparison shows its configuration
- **WHEN** a comparison with sidecar context is opened
- **THEN** browser and viewport chips are visible in the header area
