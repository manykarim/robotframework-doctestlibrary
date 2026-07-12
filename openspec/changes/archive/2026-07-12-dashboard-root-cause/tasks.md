# Tasks: dashboard-root-cause

- [x] 1.1 Engine `_region_text_job` + `EngineService.region_text` (sidecar-DPI, LRU-cached) and `POST /api/comparisons/{id}/region-text` (degraded 409, roots 403, page-range 400) + `root-cause` feature flag; endpoint tests (changed-text region, degraded, 403)
- [x] 1.2 Library: `ResultWriter.write(facets=…)` + `_write_pdf_result` passes structured differences; sidecar model `facets` field; core unit test (PdfTest sidecar carries facets)
- [x] 1.3 Viewer UI: *Explain region* panel (two-column texts + verdict badge); detail view facet sections for PdfTest comparisons; e2e journey (select region → explain → texts shown; PdfTest run → facet sections visible)
- [x] 1.4 Docs + full dashboard/e2e verification
