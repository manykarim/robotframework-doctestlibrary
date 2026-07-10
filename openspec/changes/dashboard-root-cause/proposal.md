# Proposal: dashboard-root-cause

## Why

Reviewers see *where* documents differ but not *what* changed — Percy's DOM/CSS diff shows the cause, and our document-testing analog is text: the engine can already extract and compare text inside any region (`_compare_text_content_in_area_with`), and PdfTest already computes facet-level differences that the sidecar currently flattens into unreadable note strings (analysis P1.2/P1.3, U7).

## What Changes

- **Text-in-region**: `POST /api/comparisons/{id}/region-text` runs the engine's region text comparison (reference vs candidate, at the comparison's DPI) for a selected diff region; the viewer gains an "explain region" panel showing both texts with a same/different verdict.
- **Structured PdfTest facets**: the PdfTest sidecar gains an additive `facets` list (facet, description, details) instead of only flattened notes; the comparison detail view renders facet sections with formatted payloads for text/metadata/structure differences.
- Feature flag `root-cause` for the version-skew contract.

## Capabilities

### New Capabilities

- `dashboard-root-cause`: region text explanation and structured PdfTest facet presentation.

## Impact

`doctest_dashboard/engine.py` (+ worker job), engine router, viewer UI panel; `DocTest/PdfTest.py` + `ResultWriter` (additive `facets` field, schema stays v1 with additive semantics), sidecar model. Tests: engine/endpoint units, core sidecar facet test, one e2e journey.
