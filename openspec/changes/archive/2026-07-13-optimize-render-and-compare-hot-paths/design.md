# Design: optimize-render-and-compare-hot-paths

## Context

Measured hot-path waste (proposal §Why). Constraint: outputs must be provably identical — this is a performance-only change gated by the full unit suite plus new equivalence tests, on top of the correctness fixes already merged on this branch.

## Goals / Non-Goals

**Goals:**
- ~3.9× faster PDF page loading (render + skip unneeded text extraction), identical outputs.
- Lower peak memory on the default (non-LLM) path.
- A benchmark harness that locks the gains in.

**Non-Goals:**
- SSIM fast-reject (§4.5) — deliberately deferred; it touches comparison semantics and deserves its own change.
- Any keyword or option changes.

## Decisions

1. **Pixmap conversion** in `_load_pdf` and `_render_pdf_page`:
   `np.frombuffer(pix.samples, np.uint8)` reshaped via `pix.height/width/n`, then `cv2.cvtColor` (RGB2BGR, or RGBA2BGR when `pix.n == 4`; grayscale `n==1` via GRAY2BGR). Guard non-contiguous strides (`pix.stride != pix.width * pix.n`) by reshaping on the stride and slicing columns. A dedicated equivalence test asserts array-equality against the old PNG round-trip for a real document. Rationale: measured 3.2×, verified bit-exact; PNG encode work is pure overhead.
2. **Lazy text extraction via cached instance attributes with property accessors.** `Page` gains private `_pdf_text_{data,dict,words,blocks}` slots plus a `_pdf_page_ref` (the fitz page) and properties that extract-on-first-access then cache. Setters keep working (existing code and tests assign these attributes) — assignment simply pre-populates the cache, preserving today's semantics. `pdf_text_data` remains a real extracted value (not aliased to `words`) to keep observable values byte-identical. For the non-streaming loader the fitz document closes at load end, so `Page` extraction must happen while the doc is open **only if accessed later fails** — therefore: on document close (`with` exit / `close()`), any page whose text was never accessed extracts nothing; accessing afterwards must still work. Resolution: keep a reference to the fitz *document* per DocumentRepresentation with deterministic close, and have lazy access extract from a re-opened page only as a fallback. Simpler and chosen: **lazy within the load loop is limited to the expensive formats** — `words` stays eager (2.8 ms, needed by the common move-tolerance path and truthiness checks), while `text`, `dict`, `blocks` are extracted on first access with a fallback that re-opens the document by path and page number. Re-opening costs ~ms and happens only on the rare late access after close.
3. **`count_real_difference_pixels`**: `cv2.subtract(cv2.dilate(gray, kernel), cv2.erode(gray, kernel))` on uint8 — identical values to the int64 dilate−erode (dilate ≥ erode pointwise, so no clamping occurs); measured 2.6×.
4. **Conditional diff copies**: gate the two `.copy()` calls on `llm_requested` (known before the page loop). The LLM path behavior is unchanged; the default path stops allocating unused full-page copies.
5. **Benchmarks** live in `utest/benchmarks/test_benchmarks.py`, excluded from the default run via `--benchmark-disable` compatibility: use plain `pytest-benchmark` with `benchmark` fixtures; the suite runs fine under normal pytest (benchmarks execute once). Canonical cases: load `sample.pdf` at 200 DPI; compare `sample_1_page.pdf` vs `sample_1_page_moved.pdf`.

## Risks / Trade-offs

- [Stride/alpha edge cases in pixmap conversion] → explicit `n`-channel handling + stride guard + bit-exactness test on real documents; full suite re-run.
- [Late text access after document close (lazy fallback)] → fallback re-opens by path; unit test covers access-after-close equality.
- [Tests that *assign* `pdf_text_*` (streaming path does today)] → properties keep setters; assignment bypasses extraction exactly as before.
- [pytest-benchmark adds dev-only dependency] → dev group only; no user impact.

## Migration Plan

Single release, no migration. Rollback = revert commit. Gate: equivalence tests + full `uv run pytest utest` + benchmark harness demonstrating the improvement.

## Open Questions

None.
