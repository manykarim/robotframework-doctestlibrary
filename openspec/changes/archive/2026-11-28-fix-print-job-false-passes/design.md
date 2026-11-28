# Design

## Context

See proposal.md (Why) for the false-pass paths. Current state:

- **`DocTest/PrintJobTests.py`** parses with two `parsimonious` grammars.
  - The PCL grammar reads the file as UTF-8 with `errors="ignore"`, which drops every byte ≥ 0x80. It recognizes only `ESC&l` page commands, using hard-coded value ranges, and counts every 0x0C byte as a page.
  - The PostScript grammar requires a fixed DSC section order and returns only `pjl_commands`, `pages` and `trailer`.
  - `compare_properties()` diffs lists of `{property, value}` dicts.
- **`DocumentRepresentation._load_pcl` / `_load_ps`** run GhostPCL or Ghostscript with output sent to `DEVNULL` and the exit code ignored. They count PNG files in a directory named `<stem><random 100–999>` (removed with `rmtree`) and accept zero pages. The PCL paper default is passed as `-J"@PJL SET PAPER = a4"`, and GhostPCL discards that because the line does not start with `@PJL`.
- **Verified behavior of the renderers** (Ghostscript and GhostPCL 10.08.0, Windows; GhostPCL 9.56.1 matches for PCL 5):
  - GhostPCL exits 0 for an empty file (0 pages) and for a corrupt file (20 garbage pages, nothing on stderr).
  - Ghostscript exits 1 on PostScript errors.
  - `-J@PJL DEFAULT PAPER=A4` takes effect, and a page size set by the job overrides it.
  - PCL copies (`ESC&l#X`) do not multiply rendered pages. PostScript `NumCopies` does.
  - `ESC E` flushes a page that has content, and `ESC&l0H` ejects a page.
- **Constraints:**
  - The keyword names and existing arguments must not change (`api-compatibility` gate).
  - Ghostscript and GhostPCL are AGPL. They may be invoked as subprocesses, but their code, including `pcl/tools/pxldis.py`, must not be copied.

## Goals / Non-Goals

**Goals:**
- Read the bytes of PCL and PostScript jobs with byte-accurate tokenizers that follow the published grammars (HP PCL 5 Technical Reference, HP PCL XL Feature Reference 2.0/2.1/3.0, Adobe DSC 3.0).
- Keep the existing return shape (`PrintJob(jobtype, properties)` with `[{property, value}]`), so suites that use `Check Print Job Property` keep working, and add entries to it.
- Make every rendering failure loud, while keeping each render's own temporary state.

**Non-Goals:**
- Normalizing values across languages. A PCL XL MediaSize enum is reported as-is, not translated to PCL 5 codes or names; that belongs to the job-ticket change.
- Comparing PJL in PCL jobs.
- Text extraction.
- Pinning or recording the renderer version.
- A page-count cross-check for PostScript renders, because DSC and `NumCopies` make it unreliable.

## Decisions

### D1: Hand-written byte tokenizers replace the PEG grammars
A PEG over decoded text cannot respect PCL's counted binary payloads, and it makes DSC section order mandatory. Instead:
- **PCL 5:** a scanner over the raw bytes finds `ESC` and parses parameterized sequences: parameter char 0x21–0x2F, group char 0x60–0x7E, then repeated value and parameter char, terminated by 0x40–0x5E.
  - Commands with a counted payload skip exactly `#` bytes: `ESC*b#W/V`, `ESC(s#W`, `ESC)s#W`, `ESC(f#W`, `ESC*c#W`, `ESC*v#W`, `ESC&n#W`, `ESC&p#X`, `ESC*m#W`, `ESC*l#W`, `ESC*i#W`, `ESC*g#W`, `ESC&b#W`.
  - HP-GL/2 blocks (`ESC%#B` … `ESC%#A`) are skipped to the next `ESC%`.
- **Speed:** finding the next `ESC` uses `bytes.find`/`re`, not a Python loop per byte, so large print jobs stay fast. Target: ≥ 20 MB/s on CI.
- **Alternative rejected:** keeping parsimonious with a better grammar. It still decodes text, so binary payloads stay ambiguous, and it gives worse errors.

### D2: PCL page counting
A page ends at:
- a form feed in the text stream;
- `ESC&l0H` (eject);
- `ESC E`, a UEL or the end of the job, when the page has marks.

Marks are printable text, raster or rectangle-fill commands, HP-GL/2 drawing, or a macro call since the last page end. Page-setup commands are attributed to the current page, so the reported `page` numbers match the rendered pages. This page count drives both the `page_count` property and the render cross-check (D7).

### D3: PCL XL decoder written from the HP Feature Reference
The job is segmented at UEL (`ESC%-12345X`). A segment whose PJL says `ENTER LANGUAGE=PCLXL`, or whose data starts with a stream header `) HP-PCL XL;…` (little-endian) or `( HP-PCL XL;…` (big-endian), is decoded as PCL XL:
- **Data types:** tags 0xC0–0xE5 (scalars, arrays, xy, box).
- **Attributes:** 0xF8 and 0xF9 introduce attribute IDs; 0xFA and 0xFB introduce embedded data, which is skipped by length.
- **Operators:** every other byte is an operator, which consumes the pending attributes.
- **Extracted:** BeginPage gives `Orientation` (0x28), `MediaSize` (0x25, enum or name string), `MediaSource` (0x26), `MediaType` (0x27), and `SimplexPageMode` (0x34) or `DuplexPageMode` (0x35). EndPage gives `PageCopies` (0x31).
- **Reported under the PCL 5 property names** (`paper_format`, `paper_source`, `page_orientation`, `duplex`, `copies`), with the stream's native values. Duplex is `simplex`, `duplex_vertical` or `duplex_horizontal`.
- **ASCII binding** (`` ` `` header) raises an explicit "unsupported PCL XL binding" error instead of returning nothing.
- **Licensing:** the tables are written from the HP specification. `pxldis.py` is AGPL and is not used.
- **Validation:** the decoder already decoded `testdata/invoice_diff_format.pcl` correctly in the exploration: 1 page, Orientation 0, MediaSize 2, MediaSource 1, Simplex.

### D4: Language is a property
`language` holds one entry per job segment (`PCL5`, `PCLXL`). A print job that mixes languages lists each segment, so comparing PCL 5 with PCL XL fails on `language` rather than on confusing value differences.

### D5: Tolerant PostScript DSC scanner
The scanner works on lines (bytes decoded as Latin-1, split on CRLF, CR or LF):
- **Before `%!`:** a leading PJL block is skipped and its `@PJL` lines are collected.
- **Header:** comment lines up to `%%EndComments`, or up to the first line that is not a comment. `%%+` continuations are joined.
- **Nesting:** `%%BeginDocument`/`%%EndDocument` depth is tracked, and nested `%%Page` lines are not counted.
- **Binary data:** `%%BeginData:`/`%%BeginBinary:` blocks are skipped by their declared byte or line count.
- **Features:** `%%BeginFeature:`/`%%IncludeFeature:` lines are attributed to the document (setup) or to the current page.
- **Per page:** `%%PageMedia`, `%%PageOrientation`, `%%PageBoundingBox` and `%%PageRequirements`.
- **Trailer:** comments after `%%Trailer` resolve `(atend)` header values.
- **Page count:** the number of top-level `%%Page:` comments. When there are none, it falls back to D6. The declared `%%Pages` stays in `header`, so a header that contradicts the page markers shows up as a difference.
- **Returned shape:** `pjl_commands`, `header`, `features`, `pages` and `trailer` keep the `{property, value}` / `{page, property, value}` shapes. `page_count` is added.

### D6: Non-DSC page count through Ghostscript
The page count comes from running `gs -q -dSAFER -dNODISPLAY -dNOPAUSE -dBATCH` with an `EndPage` procedure that prints one marker line per real page (reason ≠ 2), then counting those lines. This works under `-dSAFER`, as verified in the exploration with a `setpagedevice` hook. If Ghostscript is missing or fails, reading the job raises an error that names the missing page structure and the Ghostscript requirement.

### D7: Render guards
- `subprocess.run(..., capture_output=True)`.
- The render fails when any of these hold:
  - the exit code is non-zero, or the executable cannot start (`OSError`, or Windows status codes such as 0xC0000135);
  - the output contains an interpreter error marker: Ghostscript's `Unrecoverable error` or GhostPCL's `Warning interpreter exited with error code`. The generic `Error:` text is not a marker, because a job may print it itself and Ghostscript already exits non-zero on real errors;
  - zero PNGs were produced;
  - for PCL, the number of PNGs minus the pages dropped by `ignore_printfactory_envelope` differs from the D2 page count.
- Each guard raises the same `AssertionError` type the loaders raise today ("No pcl6 executable found…"), with the document path, both counts and the trimmed renderer output. Existing `Run Keyword And Expect Error` patterns stay recognizable.
- **Alternative rejected:** a new exception class. That would break callers that catch `AssertionError`.

### D8: Paper default for GhostPCL
`printing_papersize` is upper-cased and passed as the single argument `-J@PJL DEFAULT PAPER=<SIZE>`, with no quotes. `DEFAULT` rather than `SET` because SET applies only to the first job of a multi-job file. Ghostscript keeps `-sPAPERSIZE`, which already reaches it, and the job's own `PageSize` keeps precedence.

### D9: Isolated temporary output
`tempfile.mkdtemp(prefix=f"{stem}-", dir=self.tmp_directory)` replaces the random 3-digit name and its "if it exists, rmtree it" logic. The directory is removed in a `finally` block, so an error path also cleans up.

### D10: Default ignore list and `ignore_properties`
A module constant holds the case-insensitive names to ignore by default:
- DSC comments: `CreationDate`, `Title`, `For`, `Routing`, `Invocation`.
- PJL variables and `key=value` attributes inside `@PJL` lines: `NAME` (of `@PJL JOB`), `JOBNAME`, `USERNAME`, `USERID`, `HOSTID`, `HOSTNAME`, `LHOSTID`, `LHOSTJOBID`, `UPN`, `DATE`, `TIME`.

PJL lines are filtered by removing those `key=value` pairs, not by dropping the whole line, so the job's other settings on the same line are still compared.

`Compare Print Jobs` gets a final optional argument `ignore_properties=None`, which accepts a list or a comma-separated string. Its names are matched against property groups (e.g. `trailer`), comment names, page property names and PJL keys. It extends the defaults and never replaces them.

### D11: Dependency removal
- `parsimonious` is removed from `[project.dependencies]`, and `printjobs = []` stays as an empty extra.
- `uv.lock` is regenerated, and the wheel-parity allow-list records the intentional removal.
- `utest/test_source_hygiene.py::test_print_job_grammars_build_without_syntax_warnings` becomes an import check of `DocTest.PrintJobTests` with SyntaxWarnings as errors.

## Risks / Trade-offs

- **Suites start failing on real differences.** The default ignore list removes run-specific noise, `ignore_properties` is the escape hatch, and the release notes describe the change.
- **PCL jobs without a page size render A4 instead of Letter.** This is called out as BREAKING. Users who relied on Letter set `printing_papersize=letter`.
- **The structural page count may be wrong for unusual PCL**, such as macros that eject pages or overlay macros, and then a correct render fails. Mitigations:
  - the mismatch error states both counts;
  - macro execution (`ESC&f#X` with 2 or 3) is counted as marking but never as ejecting;
  - jobs with an active overlay macro (`ESC&f4X`) skip the cross-check;
  - unit fixtures cover these cases.
- **A corrupt PCL file can still render plausibly** without changing the page count. Detection is best effort; the guards remove the zero-page and renderer-crash false passes, which are the silent ones.
- **The pure-Python tokenizers are slower than C on large print jobs.** Scanning uses `find` and regular expressions (D1), and a benchmark task guards the throughput.
- **PostScript `NumCopies` multiplies rendered pages.** Visual comparison of PS jobs that request copies is unchanged by this change, which neither cross-checks nor collapses copies. It is noted for the content change.
- **Removing `PclVisitor`/`PostscriptVisitor` breaks anyone who imported them.** They were undocumented and not Robot keywords. They are mentioned in the release notes.

## Migration Plan

1. Release as a minor version, with release notes for: the behavior change, the A4 default for PCL jobs without a page size, the `ignore_properties` argument, the removal of `parsimonious`, and the removed visitor classes.
2. Users whose suites start failing either fix the real difference, add the field to `ignore_properties`, or regenerate baselines for PCL jobs without a page size.
3. Rollback means pinning the previous version. There is no data migration.

## Open Questions

- What exact error markers GhostPCL 9.56.1 prints for a PXL interpreter error. CI still uses that version; the markers are confirmed against CI output during implementation, and only the marker list may change.
