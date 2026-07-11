# Design: library-sidecar-v11

## Context

At failure time `compare_images` holds the clean page images (saved before label mutation when `result_json` is on), the diff rectangles, and — for PDFs — extracted text; PdfTest holds structured diff payloads before flattening them. Everything below is serialization of already-computed data.

## Decisions

- **D1 Highlighted candidate**: keep a copy of the clean candidate image at loop start (only when `result_json` is on); on failure draw the page's diff rectangles (2px red, no fill) on the copy → `candidate_with_diff`; downscale the same image to 240px width → `thumb`. Reference stays clean (highlights on the candidate are what reviewers scan).
- **D2 Run manifest**: `ResultWriter.ensure_run_manifest(output_dir)` writes `doctest_results/run.json` once (existence check): schema_version, started, DocTest/robot/python versions, platform, tesseract version via `pytesseract.get_tesseract_version()` guarded. Called on first sidecar write per process.
- **D3 Region text**: only when the page pair has PDF text (`pdf_text_words`) — zero extra cost; OCR-only images skip it (the dashboard's on-demand endpoint remains the fallback). Uses the same `_compare_text_content_in_area_with` the engine exposes.
- **D4 Facet data**: `_record_diff` keeps `details` (pformat) and adds `data` — the raw payload; DeepDiff objects pass through `dict(diff)`/`to_dict()` when available, else the writer's `_json_safe` stringifies. Additive.
- **D5 `name=`**: popped from kwargs before DocumentRepresentation sees them; stored as top-level `name` in the sidecar. The dashboard may prefer it for identity/display (consumed in dashboard-visual-polish, ingester: identity `name::<label>` when present).

## Risks / Trade-offs

- [Memory: one extra page copy while comparing] → only under `result_json`, one page at a time (streaming preserved).
- [Manifest concurrency under pabot] → existence-check + atomic write (temp+rename); duplicate content across processes is identical anyway.

## Migration Plan

Schema stays v1 (additive). Old dashboards ignore new fields; new dashboard falls back when fields are absent.
