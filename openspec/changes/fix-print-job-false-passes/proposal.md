## Why

Print-job comparisons can report **PASS for documents that differ**. Local experiments against Ghostscript and GhostPCL 10.08.0 reproduced five independent false-pass paths:

- When a renderer fails, its exit code is ignored and the zero rendered pages compare as equal. Seen with a broken GhostPCL install (missing DLL) and with an empty PCL file.
- PCL XL (PCL 6) jobs produce no properties at all.
- PostScript page counts and header comments are never compared, so 1 page vs 2 pages, and A4 vs Letter, both pass.
- PCL page numbering breaks on form-feed bytes inside soft-font data.
- The documented A4 paper default never reaches GhostPCL.

On top of that, 7 of 12 common PostScript shapes crash `Compare Print Jobs`. A regression-testing library must not pass silently, so these paths are fixed before the planned print-job improvements (job ticket, content comparison, print-job alignment) build on them.

## What Changes

- **Renderer failures**
  - Rendering a `.pcl` or `.ps` document for visual comparison fails with a clear error when the renderer exits non-zero, reports an interpreter error, or produces no pages.
  - A rendered page count that contradicts the structurally determined page count also fails.
  - Each render uses its own temporary directory, so parallel workers cannot delete each other's output.
- **Paper default (BREAKING)**
  - The configured paper default (`printing_papersize`, default `a4`) now reaches GhostPCL as PJL `DEFAULT PAPER`. Until now the argument was malformed and silently ignored.
  - A PCL job that sets no page size of its own now renders on the configured paper (A4 by default) instead of Letter. Baselines rendered from such jobs must be regenerated.
- **PCL property extraction** (`Get Pcl Print Job`, `Compare Print Jobs pcl`)
  - Reads the file as bytes and tokenizes PCL 5 escape sequences per the PCL 5 grammar, skipping counted binary payloads, so form feeds inside downloaded fonts or raster data no longer advance the page.
  - Accepts the full numeric range of every extracted command, so e.g. paper source 7 is no longer dropped.
  - Adds a `page_count` property.
  - Decodes PCL XL streams: page count, plus `BeginPage` paper size, tray, orientation and duplex mode and `EndPage` copies, in the same shape as PCL 5.
  - Reports the job language so PCL 5 and PCL XL jobs are never silently equal.
- **PostScript property extraction** (`Get Postscript Print Job`, `Compare Print Jobs ps`)
  - Replaces the strict grammar with a tolerant DSC scanner that handles:
    - PJL wrapping, CRLF or CR line endings;
    - missing prolog or setup sections;
    - `EPSF` and `PS-Adobe-2.0` headers;
    - `(atend)` values, nested `%%BeginDocument`;
    - features in `%%BeginSetup` and `%%BeginPageSetup`;
    - a missing final newline.
  - Compares the documented `header`, `features` and `page_count` properties, which were documented but never returned, alongside `pjl_commands`, `pages` and `trailer`.
  - A job without DSC page structure gets its page count from Ghostscript, or fails with a clear error when Ghostscript is unavailable. It never passes silently.
- **Ignore lists for comparisons**
  - Fields that change on every run are excluded by default: DSC `CreationDate`, `Title`, `For`, `Routing`, `Invocation`; PJL `JOB NAME`/`JOBNAME`, `USERNAME`, user/host/job-ID and date/time values.
  - A new optional `ignore_properties` argument on `Compare Print Jobs` lets users exclude more. The change to the keyword surface is additive.
- **Behavior change (bug direction):** comparisons that passed only because a difference was invisible will now fail. Release notes call this out.
- **Dependencies:** `parsimonious` is no longer used and is removed from the runtime dependencies. The `printjobs` extra is kept, with no dependencies, so existing install commands keep working.
- **Internal classes:** `PclVisitor` and `PostscriptVisitor` are removed. They were undocumented, not Robot keywords, and specific to the grammar.

## Capabilities

### New Capabilities
- `print-job-comparison`: what `Get Pcl Print Job`, `Get Postscript Print Job`, `Compare Print Jobs` and `Check Print Job Property` extract and compare for PCL 5, PCL XL and PostScript jobs: page counts, per-page settings, PostScript header and features, PJL, and the default and user-supplied ignore lists.
- `print-job-rendering`: how PCL and PostScript documents are turned into pages for visual comparison: failure detection, page-count cross-checking, paper defaults and isolated temporary output.

### Modified Capabilities
- `toolchain-quality-gates`:
  - "Dependency hygiene" no longer lists `parsimonious`; this release is the documented removal, and the `printjobs` extra remains as a dependency-free alias.
  - "First-party code is free of interpreter-level deprecations" no longer refers to the print-job grammars, which this change removes.

## Impact

- **Code:**
  - `DocTest/PrintJobTests.py`: tokenizers and scanner replace the grammars and visitors.
  - `DocTest/DocumentRepresentation.py`: `_load_pcl` and `_load_ps` get the render guards, PJL paper and temporary directories.
- **Tests:**
  - `utest/test_printjobtests.py`: the visitor-specific tests are replaced.
  - `utest/test_source_hygiene.py`: the grammar-build test is replaced.
  - New fixtures for the false-pass cases.
  - `atest/PrintJobs.robot` gets more cases.
- **Public surface:**
  - `Compare Print Jobs` gains an optional `ignore_properties` argument; the keyword-surface baseline is regenerated.
  - Property lists returned by the `Get … Print Job` keywords gain entries.
- **Packaging:** `pyproject.toml` (dependencies and the `printjobs` extra), `uv.lock`, and the wheel-parity allow-list for the removed dependency.
- **Users:**
  - Visible behavior changes: real differences that were hidden now fail.
  - PCL jobs without their own page size render on A4 by default.
  - Unrenderable documents raise an error instead of passing.
- **Not in scope:**
  - normalized cross-format job ticket and PJL attribute comparison;
  - text/content comparison;
  - renderer version pinning;
  - print-job splitting and page alignment.

  These are planned as separate changes.
