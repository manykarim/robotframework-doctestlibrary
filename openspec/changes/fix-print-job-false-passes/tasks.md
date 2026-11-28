# Tasks

## 1. Test fixtures

- [x] 1.1 Add small committed fixtures under `utest/testdata/printjobs/`:
  - PCL 5: 1-page and 2-page jobs with identical page setup; a job with a soft-font download whose data contains 0x0C bytes and landscape on page 2; a tray-7 job; a job without its own page size; a Letter (`ESC&l2A`) job; an empty file; a byte-scrambled copy of `testdata/invoice_diff_format.pcl`.
  - PostScript: the 12 shape variants from the exploration (PJL-wrapped, CRLF, no prolog, setup and page-setup features, EPSF, PS-Adobe-2.0, `(atend)`, nested `%%BeginDocument`, no final newline, 2-page, no DSC); A4/Letter and `*Duplex` pairs; a pair differing only in `%%CreationDate`, `%%Title` and PJL `JOBNAME`/`USERNAME`; a pair differing only in `%%Creator`.
  - Verify: the fixtures exist and `git status` shows no personal data in them (no real user, host or e-mail values).

## 2. PCL 5 tokenizer and page model

- [x] 2.1 Implement the PCL 5 byte tokenizer (design D1): parameterized and combined sequences, the counted-payload table, HP-GL/2 skipping, and UEL/PJL segmentation with language detection. Unit tests cover combined sequences (`ESC&l1o26a1S`), every payload command, and a payload containing `ESC` and 0x0C bytes.
- [x] 2.2 Implement page counting and attribution (design D2): form feed, `ESC&l0H`, and `ESC E`/UEL/end of job with marks. The soft-font fixture must report 2 pages with landscape on page 2, the no-form-feed job 1 page, and the 2-page fixture 2 pages.
- [x] 2.3 Rebuild `get_pcl_print_job` on the tokenizer:
  - read bytes;
  - report `pcl_commands` with unrestricted values;
  - add `page_count` and `language`;
  - keep the `{page, property, value}` shape.

  Verify the tray-7 fixture and that the existing `testdata/invoice*.pcl` expectations in `atest/PrintJobs.robot` still hold.
- [x] 2.4 Benchmark the tokenizer on a ≥ 20 MB print job of concatenated fixtures and verify a throughput of ≥ 20 MB/s locally. Record the figure in the test as a lenient lower bound.

## 3. PCL XL decoder

- [x] 3.1 Implement the PCL XL decoder from the HP Feature Reference (design D3):
  - little- and big-endian bindings, data-type tags, attribute IDs, embedded data skipping;
  - an explicit error for ASCII binding.

  Verify `testdata/invoice_diff_format.pcl` decodes to 1 page with Orientation 0, MediaSize 2, MediaSource 1 and simplex.
- [x] 3.2 Map BeginPage/EndPage attributes onto `pcl_commands` (D3) and set `language` to `PCLXL`. Verify:
  - the scrambled-copy fixture and an attribute-modified copy both fail `Compare Print Jobs`;
  - a PCL 5 vs PCL XL comparison fails on `language` (design D4).

## 4. PostScript DSC scanner

- [x] 4.1 Implement the tolerant DSC scanner (design D5):
  - PJL prefix, line endings, header with `%%+`, `(atend)` resolution;
  - document nesting, binary-data skipping;
  - document and page features, page comments, trailer.

  Verify every variant fixture reads without error and reports the expected page count and comments.
- [x] 4.2 Implement the Ghostscript page count for jobs without DSC pages (design D6), with an explicit error when Ghostscript is unavailable. Verify:
  - with Ghostscript: the no-DSC 2-page fixture reports 2;
  - with Ghostscript mocked away: a clear error names the missing page structure.
- [x] 4.3 Rebuild `get_postscript_print_job` to return `pjl_commands`, `header`, `features`, `pages`, `trailer` and `page_count`. Verify:
  - `Check Print Job Property` with `header` passes on a contained comment;
  - the existing `invoice.ps` expectations still hold.

## 5. Comparison semantics

- [x] 5.1 Implement the default ignore list and PJL `key=value` filtering (design D10) in `compare_properties`. Verify:
  - the CreationDate/Title/JOBNAME/USERNAME pair passes;
  - the A4/Letter, `*Duplex`, 1-vs-2-page (PS and PCL) pairs fail and name the differing property.
- [x] 5.2 Add `ignore_properties=None` (list or comma-separated string) to `Compare Print Jobs`. Verify:
  - the Creator pair passes with `ignore_properties=Creator` and fails without it;
  - group names such as `trailer` are honored.
- [x] 5.3 Update the keyword docs of `Compare Print Jobs`, `Check Print Job Property` and the `Get … Print Job` keywords:
  - property lists per type, PCL XL support, default ignore list, `ignore_properties`.

  Regenerate the keyword-surface and signature baselines, and verify `utest/test_keyword_surface.py` and `utest/test_keyword_signatures.py` pass with only additive changes.
- [x] 5.4 Remove `PclVisitor`/`PostscriptVisitor` and replace their tests in `utest/test_printjobtests.py` with tokenizer, scanner and comparison tests. Verify the print-job unit tests pass and coverage of `DocTest/PrintJobTests.py` does not drop.

## 6. Render guards

- [x] 6.1 Capture renderer output and fail on:
  - a non-zero exit, or an executable that cannot start;
  - interpreter error markers;
  - zero pages (design D7).

  Verify with unit tests that mock `subprocess.run` (exit 1, Windows 0xC0000135, error marker, no PNGs), and that `Compare Images` on the empty PCL fixture fails with a rendering error.
- [x] 6.2 Cross-check the PCL rendered page count against the D2 count, accounting for `ignore_printfactory_envelope` and skipping jobs with an active overlay macro. Verify:
  - a mocked render with an extra page fails with both counts in the message;
  - the existing PCL acceptance tests still pass.
- [x] 6.3 Pass the paper default as `-J@PJL DEFAULT PAPER=<SIZE>` (design D8). Verify with GhostPCL that the no-page-size fixture renders at A4 dimensions by default and at Letter with `printing_papersize=letter`, and that the `ESC&l2A` fixture renders at Letter.
- [x] 6.4 Use `tempfile.mkdtemp` and `finally` cleanup for render output (design D9). Verify:
  - a unit test runs two renders of same-named files concurrently and both get complete pages;
  - no temporary directory remains after a failed render.

## 7. Dependencies and quality gates

- [x] 7.1 Remove `parsimonious` from the runtime dependencies, keep `printjobs = []`, and regenerate `uv.lock`. Verify `uv lock --check` and a fresh install of `robotframework-doctestlibrary[printjobs]` succeed.
- [x] 7.2 Teach `scripts/compare_wheel_contents.py` that `parsimonious` is an intentional removal (the existing mechanism only covers tightenings). Verify the parity gate passes on a fresh `uv build`.
- [x] 7.3 Replace `test_print_job_grammars_build_without_syntax_warnings` with an import check of `DocTest.PrintJobTests` under `-W error::SyntaxWarning`. Verify `utest/test_source_hygiene.py` passes on Python 3.11 and 3.15.

## 8. Acceptance and documentation

- [ ] 8.1 Extend `atest/PrintJobs.robot`:
  - PCL XL comparison;
  - 1-vs-2-page PCL and PS;
  - A4/Letter PS;
  - run-specific-field pair passing;
  - `ignore_properties`;
  - an empty-PCL visual comparison expecting the rendering error.

  Verify the suite passes locally with Ghostscript and GhostPCL 10.08.0 and on CI with 9.56.1.
- [ ] 8.2 Confirm the GhostPCL 9.56.1 interpreter-error markers from CI output and adjust the marker list if needed (design Open Questions). Verify the CI acceptance job passes.
- [x] 8.3 Update the README print-job section (supported languages, compared properties, ignore list, A4 default for PCL jobs without a page size) and the release notes (behavior change, BREAKING paper default, `ignore_properties`, `parsimonious` removal, removed visitor classes). Verify the README example keywords run as written in a Robot dry run.

## Workflow follow-up

- Open a PR from `fix/print-job-false-passes` after implementation and CI pass.
- Archive the change after merge (`/opsx:archive`), syncing the new `print-job-comparison` and `print-job-rendering` specs.
