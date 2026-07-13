# Ultradeep Codebase Analysis & Solution Proposal

**Project:** robotframework-doctestlibrary v0.34.0
**Date:** 2026-07-12
**Scope:** Full library analysis — correctness, security, performance/algorithms, maintainability/architecture, toolstack/dependencies/CI
**Constraint:** Every proposed change is backwards compatible for library users. Keyword names, argument names, defaults, and import paths (`from DocTest.VisualTest import VisualTest`, etc.) remain unchanged. Anything that would alter install behavior gets an explicit deprecation path.

**Methodology:** Five parallel deep-analysis passes over the full source, with claims verified empirically — micro-benchmarks and repro scripts were run against real test data (`uv run python`), the unit-test suite was executed, and every headline finding was re-confirmed against the current source before inclusion. Findings below marked **[verified]** were confirmed by experiment or direct source inspection.

---

## 1. Executive Summary

The library is functionally healthy (761 unit tests; a representative 135-test subset runs green) and the two newest subsystems (WebVisualTest, dashboard live-recompare) already follow the right architecture — facades over `VisualTest.compare_images`. The most important problems found:

| Rank | Finding | Class | Impact |
|---|---|---|---|
| 1 | `llm=False` / `llm_override=False` strings are truthy in `Compare Pdf Documents` | Correctness | **False PASS** possible; unintended LLM invocation + document egress |
| 2 | `contains_barcodes` argument of `Compare Images` is a documented no-op | Correctness | Documented feature silently does nothing |
| 3 | `abs()` in block-based SSIM defeats detection of inverted regions | Correctness | **False PASS** (verified numerically) |
| 4 | PDF rendering does a PNG encode/decode round-trip | Performance | 276 → 85 ms/page (3.2×), bit-exact fix |
| 5 | Eager 4-format PDF text extraction on pure pixel compares | Performance | ~75 ms/page wasted; 3.9× page-load with #4 |
| 6 | `VisualTest.compare_images` has cyclomatic complexity 138 (~900 lines) | Maintainability | Central liability; every feature is a branch in one scope |
| 7 | No lint/format/type-check gate anywhere in CI | Toolstack | Cheapest high-leverage quality gap |
| 8 | `Wand` is a dead dependency; `imutils` is unmaintained for one 10-line function | Toolstack | Install footprint and supply-chain surface |

The recommended sequence: **fix the silent false-PASS bugs first** (they undermine the library's core promise), then land the two measured, bit-exact performance wins, then adopt the quality gates, then begin the incremental engine extraction.

---

## 2. Correctness Defects

### 2.1 HIGH — RF string-truthiness enables the LLM and can override real failures **[verified]**

`DocTest/PdfTest.py:251-256`

```python
llm_requested = bool(
    kwargs.pop("llm", False)
    or kwargs.pop("llm_enabled", False)
    or kwargs.pop("use_llm", False)
)
llm_override_result = bool(kwargs.pop("llm_override", False))
```

These flags arrive via `**kwargs`, which Robot Framework does **not** type-convert. In a `.robot` file, `llm=False` is the *string* `"False"`, and `bool("False") is True`. Every other bool kwarg in this keyword goes through `_as_bool`; only the LLM flags bypass it.

**Failure scenario:** `Compare Pdf Documents  ref.pdf  cand.pdf  compare=text  llm=False  llm_override=False` — the user explicitly disables the LLM, yet both flags become `True`. The LLM is invoked (cost + both PDFs egress as attachments), and because `llm_override_result` is truthy, an approving verdict (`PdfTest.py:629`) sets `differences_detected=False` → **a genuinely different PDF passes**.

**Fix (BC):** Route all four flags through `_as_bool(...)`. Real booleans and `${True}`/`${False}` behave identically; only the broken string interpretation changes — in the direction users already intend.

### 2.2 HIGH — `contains_barcodes` in `Compare Images` is a dead parameter **[verified]**

`DocTest/VisualTest.py:262` (parameter), `465-484` (document construction)

`compare_images` declares `contains_barcodes: bool = False`, documents it (lines 289, 315: *"Identified barcodes in documents and excludes those areas from visual comparison"*), but never forwards it to either `DocumentRepresentation` and never triggers `identify_barcodes()`. Because it's a named parameter (not `**kwargs`), it's swallowed with no error. The only functional use of barcode detection is the unrelated `Get Barcodes` keyword (line 1994).

**Failure scenario:** `Compare Images  ref.pdf  cand.pdf  contains_barcodes=${True}` — user expects barcode regions masked and their decoded content compared instead. Neither happens; 1-px barcode render jitter fails the comparison and content is never checked.

**Fix (BC):** Forward the flag into both `DocumentRepresentation(...)` constructions (which already accept `contains_barcodes`) and apply barcode areas as ignore regions + content check, matching the documented behavior. Default `False` preserves current behavior exactly.

### 2.3 MEDIUM-HIGH — `abs()` defeats block-based SSIM for inverted content **[verified numerically]**

`DocTest/DocumentRepresentation.py:557`

```python
lowest_score = abs(min(lowest_score, block_score))
```

SSIM ranges over [-1, 1]. A photometrically inverted block (selected/highlighted field, dark-mode fragment, inverted logo) scores near −1 — maximally dissimilar — but `abs()` converts it to near +1, so the guard `lowest_score < (1.0 - threshold)` never trips. Reproduced: raw block SSIM −0.9456 → reported 0.9456 → `fail=False` at thresholds 0.3/0.5/0.6 where it must fail. This defeats the exact purpose of block-based mode ("catch differences in smaller areas").

**Fix (BC):** `lowest_score = min(lowest_score, block_score)`. Also revisit the `1.0 - block_based_ssim_score` returned in `compare_with` (`:482`) so reported scores stay sane for negative SSIM. Only affects the opt-in `block_based_ssim=True` mode, and only in the direction of catching real differences.

### 2.4 MEDIUM — `compare_print_jobs` raises `UnboundLocalError` for `afp`/unknown types **[verified]**

`DocTest/PrintJobTests.py:354-362` — for `type='afp'` (named in docstrings) or any typo, `reference_print_job`/`test_print_job` are never bound; the trailing `compare_properties(...)` raises `UnboundLocalError` instead of a clear error.

**Fix (BC):** `else: raise ValueError(f"Unsupported print job type '{type}'")`; implement or explicitly reject `afp`.

### 2.5 MEDIUM — `compare_properties` dereferences `None` on asymmetric property sets **[verified]**

`DocTest/PrintJobTests.py:375-376` — `next(..., None)` can return `None`; `test_property_item['value']` then raises `TypeError`, masking the real "missing property" difference with an opaque crash.

**Fix (BC):** treat a missing property as a difference, record it, `continue`.

### 2.6 MEDIUM — EAST OCR `_resize_image` returns width/height swapped **[verified]**

`DocTest/Ocr.py:135` returns `(resized_image, height, width, ...)` while the caller (`:43`) unpacks `image, width, height, ...`. The transposed values drive `blobFromImage` and the results mask. Hidden today only because both call sites default to square 480×480; any non-square EAST target (`width=320, height=640`) produces transposed, wrong/empty detections.

**Fix (BC):** return `(resized_image, width, height, ratio_width, ratio_height)`.

### 2.7 MEDIUM — user's `mask` argument shadowed by watermark loops, corrupting the JSON sidecar **[verified]**

`DocTest/VisualTest.py:636, 678` rebind the local `mask` (the user's ignore-area definition, parameter at `:271`) to watermark image arrays; the sidecar write at `:1135` (`"mask": mask`) then records a stringified NumPy array instead of the mask definition. Silent metadata corruption for dashboard/tooling consumers whenever `result_json` + `watermark_file` + a differing page coincide.

**Fix (BC):** rename loop variables (`for wm in watermarks:`).

### 2.8 MEDIUM — resource leaks: `fitz` documents never explicitly closed **[verified by inspection]**

- `DocumentRepresentation._load_pdf` (`DocumentRepresentation.py:1019-1046`): `doc = fitz.open(...)` never closed on the non-streaming path (contrast: streaming path closes in `close()`).
- `DocTest/PdfTest.py:817/832, 862/887/890, 920/942/945`: `check_text_content`, `PDF Should Contain Strings`, `PDF Should Not Contain Strings` release via `doc = None`, never `doc.close()`, including on error paths.

Handles accumulate until GC — file-handle pressure over long suites; on Windows can block deletion/overwrite of references (relevant to reference-run promotion).

**Fix (BC):** `with fitz.open(...) as doc:` / `try/finally: doc.close()`.

### 2.9 MEDIUM-LOW — PCL/PS pages use string page numbers, so page-scoped masks never match **[verified by inspection]**

`DocumentRepresentation.py:1210, 1267` set `page_number=str(index+1)` while every other loader uses `int` and mask pages are cast to `int` (`:829-830`). `int == str` is always `False` → any `{"page": N, ...}` ignore area is silently skipped for `.pcl`/`.ps` documents.

**Fix (BC):** `page_number=index + 1` (int), matching the other loaders.

### 2.10 LOW — minor items

- `check_pdf_text` kwarg in `PdfTest.py:287,295` is parsed, coerced, logged — and never used. Wire it up or deprecate with a warning.
- `EastTextExtractor._get_east` (`Ocr.py:159-173`): with a custom relative `east=` path, the model downloads to a package-relative path but `readNet` loads the original path → guaranteed load failure; fallback model fetched without integrity check.
- `VisualTest.py:1311,1319`: text-movement screenshot labels double-increment an already 1-based page number (cosmetic).
- `VisualTest.py:866-871`: dead local `check_pdf_content = False` and a no-op pre-block in `get_pdf_content`.

### 2.11 Verified non-issues (checked, sound)

Text-normalization ordering (`TextNormalization`/`_sanitize_span_text`); character replacements applied consistently across snapshot/structure/contain-strings paths; WebVisualTest bool handling (annotated `full_page`, `_truthy` for kwargs); `sanitize_baseline_name` path-traversal guard; `compare_document_text_only` replace-pairing and tolerance logic; Downloader URL-derived filenames are not path-traversable.

---

## 3. Security Findings

### 3.1 Downloader hardening (`DocTest/Downloader.py:59-118`) — LOW-MEDIUM

Impact is bounded (URLs originate from the test author), but for CI environments fetching references from semi-trusted locations:

1. **SSRF via redirects:** `urllib.request.urlretrieve` follows 3xx; `ALLOWED_SCHEMES` vets only the initial URL, so an allowed `https://` URL can bounce to internal hosts (`169.254.169.254`, intranet).
2. **`ftp://` allowed** — rarely needed; drop from the default allowlist (with a documented opt-in escape hatch to stay BC).
3. **`max_size` enforced post-download** (`:104-107`) — a huge/slow endpoint transfers fully before rejection. Stream and abort at the byte limit.
4. **Temp-file accumulation:** downloads land in `tempfile.gettempdir()` and are never cleaned.

**Fix (BC):** custom opener that re-validates scheme/host on each redirect hop (optionally blocking RFC1918/link-local resolutions), streamed size enforcement, and cleanup (or documented ownership) of temp files.

### 3.2 Supply chain

- No dependabot, no `pip-audit`/CodeQL in CI (see §6).
- EAST model fallback downloaded via plain `urlretrieve` with no checksum (§2.10) — add a pinned SHA-256.
- PyPI publish uses a long-lived API token; migrate to OIDC trusted publishing (workflow action is already SHA-pinned — good).

---

## 4. Performance & Algorithmic Improvements (all measured)

Baseline: end-to-end `compare_images` on a 1-page PDF pair with `move_tolerance=20` ≈ **1545 ms**, dominated by SSIM (~44%), PDF→PNG→numpy rendering (~22%), text-dict extraction (~8%), PyMuPDF marshalling (~10%).

### 4.1 Eliminate the PNG encode/decode round-trip — **3.2× render, bit-exact** [S effort]

`DocumentRepresentation.py:1030-1031` (`_load_pdf`) and `_render_pdf_page`:

```python
img_data = pix.tobytes("png")                      # ~139 ms/page PNG encode
image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)  # ~33 ms decode
```

PyMuPDF already holds raw RGB in `pix.samples`. Measured (sample.pdf @200 DPI, 40 iters): **276.3 → 85.0 ms/page**, output verified bit-exact (`np.array_equal == True`).

```python
pix = page.get_pixmap(dpi=self.dpi)
buf = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
image = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
```

Guard `pix.stride != pix.width * pix.n` (reshape on stride, slice) and alpha. Applies to both streaming and non-streaming paths. Zero behavior change.

### 4.2 Lazy PDF text extraction — up to ~75 ms/page on pixel-only compares [M effort]

`DocumentRepresentation.py:1034-1037, 1069-1072` unconditionally run `get_text("text"/"dict"/"words"/"blocks")` per page. Measured costs: dict **66.7**, text 8.7, words 2.8, blocks 0.6 ms/page. Usage analysis: `pdf_text_dict` is needed only for pattern ignore areas / structure comparison / mask filtering; `pdf_text_data` is used *only* as a truthiness check (`:792`, `:1356`) — replace with `bool(self.pdf_text_words)`; `pdf_text_blocks` only in `apply_pixel_masks_to_pdf_text`; only `pdf_text_words` is needed for the common move-tolerance path.

**Fix (BC):** convert the four `pdf_text_*` fields to lazily-cached properties. Careful with the streaming `close()` path: extract before releasing the underlying page, or keep the page alive inside the cache window. Combined with §4.1, per-page PDF load drops **~342 → ~88 ms (~3.9×)** with no output change.

### 4.3 `count_real_difference_pixels`: uint8 morphology instead of int64 — 2.6× [S effort]

`VisualTest.py:80-81`: `.astype(int)` promotes full-page dilate/erode to int64. `cv2.subtract(cv2.dilate(gray,k), cv2.erode(gray,k))` on uint8 is identical in result: measured **14.7 → 5.7 ms** (1191×1080). The caller (`:570-573`) also re-runs two `cvtColor`s that `Page.compare_with` already computed — cache/pass the grays.

### 4.4 Skip diff-image copies when no LLM will consume them [S effort]

`VisualTest.py:978-981` unconditionally `.copy()`s `absolute_diff` and `combined_diff` (~11 MB/page at 200 DPI color) into `detected_differences`, but they are only read on the LLM path (`:1234-1240`). Copy only when `llm_requested` — meaningful peak-memory reduction on multi-page documents in the default configuration.

### 4.5 SSIM cost — optional fast-reject (informational) [M effort]

Full-res `structural_similarity(..., full=True)` (`DocumentRepresentation.py:471`) is the inherent wallclock floor (~680 ms/compare measured). A backwards-compatible acceleration: half-scale SSIM fast-reject first, full-res only near the threshold (~4× on clearly-equal/clearly-different pages), gated so default semantics are unchanged. The existing `np.array_equal` short-circuit (`:457`) is good — keep it. Benchmarked non-issue: per-block `cvtColor` in block-based SSIM is not a hotspot (21.4 vs 22.8 ms).

### 4.6 Regression harness

`.benchmarks/` exists but `pytest-benchmark` is not set up. Add it (dev group) with 3–4 canonical comparisons so §4.1–4.4 gains are locked in and future regressions surface in CI.

---

## 5. Maintainability & Architecture

### 5.1 Diagnosis

1. **`VisualTest.compare_images` is a god-method** — radon CC **138 (F)**, ~900 lines (`VisualTest.py:254-1153`): ~25-option kwarg unpacking, URL download, reference-run promotion, doc loading, a per-page loop inlining dimension check / SSIM / anti-aliasing budget / watermark heuristics / multi-watermark mask merging / text acceptance / movement dispatch / diff collection / result assembly / LLM dispatch / failure raise.
2. **`VisualTest` is a god-class** — 4104 lines, ~70 instance attributes, 5 concerns; includes ~1600 lines of pure CV feature-matching (SIFT CC 41, ORB CC 41, matchtemplate CC 33, template containment CC 34; lines ~2037-3660) with zero Robot dependency.
3. **`PdfTest.compare_pdf_documents` (CC 77) duplicates cross-cutting machinery** — `_coerce_label_value`/`_decision_equals_flag` are byte-identical copies (`VisualTest.py:36,41` vs `PdfTest.py:45,50`); parallel LLM handlers; reference-run detection, output-dir reading, sidecar wiring, char-replacement parsing each re-coded.
4. **`Page` is overloaded** (`DocumentRepresentation.py:29-947`): image + OCR + EAST + four near-parallel mask processors + SSIM + two barcode engines + three area-text backends.
5. **Two logging systems used interchangeably** (stdlib `logging` in 8 files, `robot.api.logger` in 5, both in the same methods), stray `print()` in 3 files, 77 broad `except Exception` package-wide (32 in VisualTest.py alone), many silently swallowing.
6. **`**kwargs` threading** through 9 VisualTest / 8 PdfTest / 6 WebVisualTest / 3 DocumentRepresentation signatures; unknown keys pass silently into constructors — typos are undetectable.

**Key strength to build on:** WebVisualTest already subclasses VisualTest and delegates to `compare_images` (`WebVisualTest.py:44, 394`), and the dashboard's live-recompare calls `compare_images` + reads the JSON sidecar. Two consumers already treat the comparison logic as an engine behind a facade — the work is hoisting that engine out of the 900-line keyword, not inventing architecture. The `openspec/specs/api-compatibility` baseline is both the constraint and the safety net.

### 5.2 Incremental refactoring plan — each phase shippable, each backwards compatible

Invariant for every phase: keyword names, argument names, defaults, and import paths unchanged; api-compatibility spec is the merge gate.

- **Phase 0 — Lock the contract.** Libdoc-driven keyword-signature snapshot test (names + arg names + defaults) plus a public-import smoke test. Makes all later phases provably safe. *Small; do first.*
- **Phase 1 — Extract shared cross-cutting helpers.** Deduplicate `_coerce_label_value`/`_decision_equals_flag`, char-replacement parsing, robot-variable/output-dir reading, reference-run promotion into internal modules (reuse `ReferencePromotion.py`, `ResultWriter.py`). Pure code motion; facades keep one-line delegations.
- **Phase 2 — Extract the CV algorithm layer.** Move SIFT/ORB/template/movement detection (~1600 lines) into `DocTest/matching/` as typed free functions/strategies — they already take images, not self-state. Removes 5 of the top-10 complexity offenders and makes the algorithms directly unit-testable.
- **Phase 3 — Introduce a `ComparisonEngine` core.** Typed `ComparisonOptions` + two `DocumentRepresentation`s → structured `ComparisonResult`. Decompose the per-page loop into named strategy steps (dimension / SSIM / pixel-budget / watermark / text / movement). `compare_images` and `compare_pdf_documents` become parse → options → engine → render (log + sidecar + raise). Reporting layer shared. Complexity F → low.
- **Phase 4 — Typed options replace `**kwargs` threading.** Parse and validate kwargs once at the keyword boundary; keep `**kwargs` in keyword signatures (required for RF compatibility; spec forbids removal) but reject unknown keys with a clear error — after one release of warning-only, to stay BC for existing suites that may carry typos today.
- **Phase 5 — Split `Page` responsibilities.** Page = image + geometry; OCR through `Ocr.py`; the four `_process_*_ignore_area` behind `IgnoreAreaManager` (the dashboard already treats it as mask authority); barcode and comparison separated. Largest phase; do after the engine boundary is stable.
- **Phase 6 — Converge logging/error conventions.** Stdlib `logging` for internal diagnostics; `robot.api.logger` only in facade/reporting; delete `print()`; narrow the broad excepts (each either handles specifically, logs at warning with context, or propagates).

### 5.3 Quick wins (independent, low-risk)

1. Deduplicate the identical LLM-label helpers (4 functions → 2).
2. Remove stray `print()` in PdfTest/PrintJobTests/CapabilityCheck.
3. Land the Phase-0 signature snapshot test immediately.
4. Extract the anti-aliasing budget block (`VisualTest.py:558-594`) and watermark-merge block (`:631-760`) into named, tested private methods.
5. Type-annotate the CV helper signatures ahead of Phase 2.
6. Normalize module logger naming (`LOG` vs `logger`).
7. Generate the public keyword list via libdoc into the api-compatibility baseline.

---

## 6. Toolstack, Dependencies, Testing, CI/CD

### 6.1 Quality gates (adopt now; zero user impact)

1. **ruff (check + format) + lenient mypy** over `DocTest/` as a CI job — currently there is no lint/format/type gate at all (`pyproject.toml` has no `[tool.*]` beyond hatch). Biggest quality-per-effort win, and the guard rail the §5 refactor needs.
2. **pytest-xdist + pytest-timeout** in the dev group; `pytest -n auto --timeout=300`. Measured: a 135-test subset takes 130 s serially on CPU-bound image work; no per-test timeout means a hung OCR/subprocess blocks CI until job timeout.
3. **pre-commit** (ruff, ruff-format, whitespace).
4. **Coverage floor** — coverage is already computed (`tasks.py`) but never enforced; add `--fail-under` and optionally Codecov.
5. **pytest-benchmark** harness (see §4.6).

### 6.2 CI/CD

6. **Take live-LLM tests off the required path.** The `test` job across all 4 Pythons sets `DOCTEST_LLM_ENABLED: true` against openrouter.ai (ci.yml:66-108) — an external paid service on every PR × 4 interpreters. Run LLM tests on one interpreter as a non-required (or scheduled) job; keep `smoke-no-ai` required.
7. **Fix or drop `docker-image.yml`:** it uses EOL action majors (checkout@v2, login-action@v1) and the Dockerfile installs the library **from PyPI** (Dockerfile:10), so the image never tests the PR's code. Build from the checkout (`pip install .[ai]`) and bump actions.
8. **dependabot.yml** (pip/uv + GitHub Actions) and a `pip-audit` step.
9. **OIDC trusted publishing** for PyPI (drop `PYPI_API_TOKEN`).

Existing strengths to preserve: clean uv+hatchling migration with committed `uv.lock`; the wheel-parity gate (`compare_wheel_contents.py`) and `audit_resolved_versions.py`; thorough dashboard/web CI (frontend build, Playwright e2e, Browser+Selenium acceptance).

### 6.3 Dependency actions

| Dependency | Finding **[verified]** | Action | User impact |
|---|---|---|---|
| **Wand** | Imported nowhere in the codebase (PDF/PS rendering uses PyMuPDF + `gs` subprocess) | Drop from `dependencies`; update audit-script PRESENCE list | None (dead weight removal) |
| **imutils** | Unmaintained (2021); single call `grab_contours` at `VisualTest.py:2317` (~10 trivial lines) | Inline the function; drop the dep | None |
| **scipy** | Not directly imported; transitive via scikit-image | Drop direct entry (or keep only as documented floor) | None |
| 7 unpinned deps (opencv, pytesseract, parsimonious, pylibdmtx, pyzbar, …) | No version floors — PyPI consumers get whatever resolves | Add tested lower bounds | None (floors only) |
| **pylibdmtx / pyzbar / parsimonious** | Niche features (barcodes, PCL/PJL) requiring native system libs (libdmtx0b, libzbar0) | **Deprecation path only:** add `[barcode]`/`[printjobs]` extras now, keep in base deps for ≥1 minor release, emit clear ImportError pointing at the extra (lazy imports already in place), remove from base only in a documented minor/major bump | Announced, staged |

Python matrix 3.10–3.13 matches classifiers and CI; plan the 3.10 EOL (~Oct 2026) drop as a normal minor-version note.

### 6.4 Test-suite health (measured)

761 unit tests collected (2.3 s collection); 8-file representative subset: **133 passed, 2 skipped, 0 failed** in 129.9 s. `import DocTest.VisualTest` ≈ 0.54 s. Robot acceptance: 8 suites via `invoke atests`; Playwright e2e for the dashboard. Verdict: green on correctness, amber on speed/isolation (serial, no timeouts) — addressed by §6.1.2.

---

## 7. Backwards-Compatibility Strategy

1. **Contract:** keyword names, argument names, argument defaults, and Python import paths are frozen. The `openspec/specs/api-compatibility` spec is the definition; the Phase-0 libdoc snapshot test is the enforcement.
2. **Behavior changes are bug-direction-only:** every §2 fix changes behavior only where the current behavior contradicts the documented behavior (false PASSes, no-op flags, crashes replacing clear errors). Documented-and-relied-upon behavior is never altered.
3. **Performance fixes are output-identical:** §4.1 is verified bit-exact; §4.2–4.4 change no outputs; §4.5's fast-reject is designed to be semantics-preserving under existing thresholds.
4. **Install-behavior changes are staged:** extras introduced additively first, base deps retained ≥1 release, actionable ImportErrors, removal only with release-notes announcement.
5. **`**kwargs` stays in keyword signatures** (RF suites pass arbitrary named args); unknown-key rejection ships warning-first for one release.

---

## 8. Prioritized Roadmap

### Milestone 1 — Correctness (1 release, ~days)
- §2.1 LLM flag coercion (**the** critical fix), §2.2 `contains_barcodes`, §2.3 block-SSIM `abs()`, §2.4/§2.5 print-job errors, §2.6 EAST resize swap, §2.7 mask shadowing, §2.8 fitz closes, §2.9 PCL page numbers — each with a regression test.
- Quick wins §5.3.1–3.

### Milestone 2 — Performance + guard rails (1 release)
- §4.1 pixmap direct conversion, §4.3 uint8 morphology, §4.4 conditional copies (all S).
- §4.6/§6.1.5 benchmark harness first, so gains are measured and locked.
- §6.1.1–4 ruff/mypy/xdist/timeout/pre-commit/coverage floor.

### Milestone 3 — Deps + CI hygiene (1 release)
- Drop Wand, inline imutils, drop direct scipy, add version floors (§6.3).
- CI: LLM off the required path, Docker fix, dependabot + pip-audit, OIDC publish (§6.2).
- §4.2 lazy text extraction (M; behind the benchmark harness).
- Introduce `[barcode]`/`[printjobs]`/`[all]` extras additively (deprecation clock starts).

### Milestone 4+ — Architecture (rolling, one phase per release)
- §5.2 Phases 0→6, gated by the signature snapshot test and the acceptance suites.
- §3.1 Downloader hardening and §4.5 SSIM fast-reject as opt-in improvements along the way.

---

*Benchmark and repro scripts referenced above were executed with `uv run python` against `testdata/` and `utest/testdata/`; profile: SSIM ~44% / render ~22% / text-dict ~8% / marshalling ~10% of a 1545 ms single-page comparison.*
