# Spec delta: web-visual-testing (web-context-metadata)

## ADDED Requirements

### Requirement: Capture context in results
Web comparisons SHALL record the capture context (library, browser, viewport,
device pixel ratio, URL — best effort) in the result sidecar.

#### Scenario: Context written
- **WHEN** a web comparison runs with `result_json` enabled
- **THEN** the sidecar's `context` object names the capturing library and configuration

### Requirement: Split baselines by configuration
`split_baselines_by=browser,viewport` SHALL give each configuration its own
baseline path segment and a config-qualified sidecar label; unknown split keys
SHALL fail at import time.

#### Scenario: Per-browser baselines
- **WHEN** the same comparison runs under two browsers with `split_baselines_by=browser`
- **THEN** each browser reads and writes its own baseline file and history stays separate
