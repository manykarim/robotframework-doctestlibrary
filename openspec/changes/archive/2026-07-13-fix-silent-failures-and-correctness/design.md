# Design: fix-silent-failures-and-correctness

## Context

Nine verified defects from docs/ultradeep-analysis-solution-proposal.md §2 and §3.1. Three can produce false PASSes (LLM string-truthiness, dead `contains_barcodes`, block-SSIM `abs()`); the rest are crashes-instead-of-errors, silent metadata corruption, resource leaks, and download hardening. The `openspec/specs/api-compatibility` baseline freezes keyword names, argument names, and defaults — every fix here stays inside that contract.

## Goals / Non-Goals

**Goals:**
- Eliminate all false-PASS paths; make documented behavior real behavior.
- One regression test per fix in `utest/`, runnable via `uv run pytest utest`.
- Zero changes to keyword signatures, defaults, import paths, or passing-suite outcomes (except where the current outcome is itself the bug).

**Non-Goals:**
- Performance work, dependency changes, CI changes (separate changes on this branch).
- The §5.2 architecture refactor; fixes are made in place with minimal structural churn.
- SSRF DNS/IP-level blocking (RFC1918 resolution checks) — scheme/host re-validation on redirects only; full egress policy is out of scope.

## Decisions

1. **LLM flags through `_as_bool`** (PdfTest.py:251-256). Reuse the module's existing `_as_bool` helper on the popped values — smallest diff, consistent with sibling options. Alternative (annotating params) rejected: they arrive via `**kwargs`, and adding named params would change the signature surface.
2. **`contains_barcodes` forwarded, then applied as ignore areas + content check** (VisualTest.py). Pass `contains_barcodes=contains_barcodes` into both `DocumentRepresentation` constructions (they already accept it and run `identify_barcodes()` at load: DocumentRepresentation.py:989/1081). Exclude detected barcode rects from pixel diff by adding them to the page's ignore areas; compare decoded values pairwise and fail with a clear message on mismatch. Content check runs only when both docs decoded at least one barcode; mismatched counts are a failure. This matches the docstring ("excludes those areas ... barcode data will be checked instead").
3. **Block SSIM: `min()` without `abs()`; clamp reported score** (DocumentRepresentation.py:557). `lowest_score = min(lowest_score, block_score)`. For the score reported upward from `compare_with` (:482), clamp to `max(block_score, 0.0)` before the `1.0 - score` inversion so downstream percentage displays stay in range; the pass/fail guard uses the raw min.
4. **Print jobs: explicit `else: raise ValueError`; missing property = difference.** In `compare_properties`, when the counterpart item is `None`, append a difference entry naming the property and `continue`. Keep the final `AssertionError('The compared print jobs are different.')` behavior.
5. **EAST `_resize_image` return order fixed at the source** (Ocr.py:135) rather than re-swapping at the call site — one line, and both call sites become correct.
6. **Watermark loops rename `mask` → `wm`** (VisualTest.py:636, 678). Pure rename; sidecar write at :1135 then sees the user's argument.
7. **fitz closes:** `_load_pdf` wraps in `with fitz.open(...) as doc:` (all pixmaps/text are copied out before exit). The three PdfTest text keywords use `try/finally: doc.close()` (smaller diff than restructuring their long bodies).
8. **PCL/PS page numbers:** `page_number=index + 1` (int) in `_load_pcl`/`_load_ps`, matching PDF/image loaders. No normalization inside `Page.__init__` — keeps the change surface minimal.
9. **Downloader:** replace `urlretrieve` with `urllib.request.build_opener` + a `HTTPRedirectHandler` subclass that validates each hop's scheme against `ALLOWED_SCHEMES` (now `{"http", "https"}`); stream `resp.read(chunk)` to the temp file, aborting and unlinking once `max_size` is exceeded. `ftp` removal is the one deliberate default tightening — docstring-noted; anyone needing ftp can fetch out-of-band. Alternatives (requests/httpx) rejected: no new runtime dependency for stdlib-capable work.

## Risks / Trade-offs

- [Barcode content check turns a previously always-ignored flag into active behavior] → Default stays `False`; behavior only changes for callers who explicitly passed the flag expecting the documented semantics. Release notes flag it.
- [Block-SSIM fix makes previously-passing inverted-content comparisons fail] → That is the fix; opt-in mode only. Release notes flag it.
- [`with fitz.open()` in `_load_pdf` could break if later code lazily reads from `doc`] → Verified: everything needed is materialized into `Page` objects inside the loop; unit suite + acceptance suites gate.
- [Downloader redirect handler diverges from urlretrieve semantics (e.g., ftp removal)] → Scheme set is explicit and documented; tests cover redirect rejection and size abort with a local http server.
- [String-boolean coercion changes behavior for anyone passing `llm=False` strings today] → Today that string *enables* the LLM against the user's intent; the change restores intent. No plausible reliance on the inverted meaning.

## Migration Plan

Single release. No data or config migration. Rollback = revert the commit. Each fix lands with its regression test; full `uv run pytest utest` and the acceptance-relevant suites must pass before archive.

## Open Questions

None blocking — all decisions above are settled against the analysis document.
