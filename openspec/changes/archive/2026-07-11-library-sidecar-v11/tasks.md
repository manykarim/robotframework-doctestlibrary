# Tasks: library-sidecar-v11

- [x] 1.1 Highlighted candidate + thumb: clean-copy at loop start, rectangle drawing + 240px thumb in ResultWriter, wired for failing pages; unit tests (files exist, differ from clean candidate, thumb width)
- [x] 1.2 Run manifest `ensure_run_manifest` (atomic, once) called on first write; unit test (single file, version fields)
- [x] 1.3 `regions_text` for PDF-text pages (no OCR trigger); unit tests (pdf failure carries texts, image failure stays fast/absent)
- [x] 1.4 PdfTest facet `data` payloads (DeepDiff→dict, json-safe); `name=` kwarg on both compare keywords → sidecar `name`; unit tests
- [x] 1.5 Dashboard sidecar model additive fields; contract + full core/dashboard suites green
