# Design: dashboard-root-cause

## Context

`Page._compare_text_content_in_area_with(cand_page, rect, force_ocr)` returns `(same, ref_text, cand_text)` and powers the library's `check_text_content` option — reusing it server-side gives exact engine parity. PdfTest collects `{facet, description, details}` diffs (`llm_differences`) that `_write_pdf_result` currently flattens into note strings.

## Goals / Non-Goals

**Goals:** on-demand region text explanation (explicit button — OCR is expensive), structured facets in sidecar + UI, both cached/tested.
**Non-Goals:** word-level inline diff rendering (pre-formatted facet payloads suffice for v1), region text for degraded records (no paths).

## Decisions

- **D1 Region text job**: `_region_text_job(reference, candidate, page_no, region, dpi, force_ocr)` in the engine worker — builds both `DocumentRepresentation`s at the comparison's sidecar DPI (region pixels are DPI-relative), calls the Page comparison, returns `{same, reference_text, candidate_text}`. Engine-level LRU cache on (file fingerprints, page, region, dpi).
- **D2 Endpoint**: `POST /api/comparisons/{id}/region-text {page_no, region}` — resolves paths + DPI from the stored sidecar, 409 for degraded records, roots-confined, 400 for out-of-range pages.
- **D3 Viewer UI**: when a diff region is selected, an *Explain region* button fetches and renders a two-column panel (reference text / candidate text) with a same/different badge. Explicit fetch keeps OCR cost user-controlled.
- **D4 Facets**: `ComparisonResultWriter.write` gains optional `facets`; `_write_pdf_result` passes its differences (keeping the flattened notes for backward-looking readers). Sidecar model gains `facets: List[dict]` (schema stays v1 — additive field, model already tolerates extras). Detail view renders one collapsible section per facet with a monospace payload.

## Risks / Trade-offs

- [Region text on image-only comparisons needs OCR] → the endpoint reports OCR-unavailable as 409 with the same messaging as mask preview; PDFs use embedded text and work everywhere.
- [DPI mismatch → wrong region] → DPI always taken from the stored sidecar, never client-supplied.

## Migration Plan

Additive on both library and dashboard; old sidecars simply lack `facets` (UI hides the section).
