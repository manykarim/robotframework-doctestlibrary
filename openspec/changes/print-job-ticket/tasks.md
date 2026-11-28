# Tasks

## 1. Fixtures

- [x] 1.1 Extend `utest/testdata/printjobs/make_fixtures.py` with ticket fixtures:
  - PJL `PAPER=LETTER` + `ESC&l26A`; PJL `PAPER=LETTER` with an A4 page, `ESC E`, then a second page;
  - a 2-page PCL 5 job with `ESC&l1S` before page 1 only; PCL 5 trays 2, 4, 6 and 7; `ESC&l2G` bin; `ESC&n#Wd` media name;
  - PCL XL with DuplexPageMode vertical and PageCopies 2;
  - PJL `DUPLEX=ON BINDING=SHORTEDGE`;
  - PostScript with `*InputSlot Tray2`; PostScript whose DSC claims `*PageSize Letter` but requests `PageSize [595 842]`; PostScript `PageSize [600 800]`; PostScript `Duplex true Tumble false`;
  - two PCL jobs differing only in PJL `JOBNAME`/`USERNAME`.

  Verify the fixture byte-equality test passes with the regenerated set.

## 2. Normalization tables

- [x] 2.1 Check the paper-size, duplex, orientation, tray, bin and media tables of design D4 against the primary documents (HP PCL 5 Technical Reference, PCL XL Feature Reference 2.0/2.1/3.0, PJL Technical Reference, PLRM 3), including PCL XL vertical binding → long_edge. Record the document and section per table in the module docstring, and correct any entry that differs from D4 (and D4 itself).
- [x] 2.2 Implement the tables in `DocTest/PrintJobTicket.py` with one parametrized unit test per table entry, plus PostScript size matching (±2 pt, both orientations, `<w>x<h>mm` fallback). Verify `[595 842]`, `[842 595]` → A4 and `[600 800]` → `212x282mm`.

## 3. Reader extensions

- [x] 3.1 Add the additive `payload_offset` field to `PclCommand` and read `&n#Wd` media names. Verify with a unit test, and verify that the existing print-job tests and `Get Pcl Print Job` output are unchanged.
- [x] 3.2 Read the PCL XL MediaDestination attribute (0x24) in the decoder without changing `read_pclxl_pages` output. Verify with a big-endian and a little-endian unit test.
- [x] 3.3 Implement the Ghostscript `setpagedevice` capture and output parser (design D6):
  - Verify with Ghostscript: the mid-job-changes fixture reports the requested keys per page.
  - Verify with Ghostscript mocked missing or failing: DSC-only data with a recorded reason, and no error.

## 4. Ticket model and effective values

- [x] 4.1 Implement `Setting`/`PageTicket`/`JobTicket` and their dictionary form (design D2). Verify a round-trip unit test and that every page entry carries `sources` and `raw`.
- [x] 4.2 Implement the PJL state (SET/DEFAULT, UEL inheritance, normalized and raw variables). Verify the `RESOLUTION=600` scenario and that DEFAULT survives a UEL while SET does not.
- [x] 4.3 Implement the PCL 5 state machine (D3): language defaults, PJL fallback, carry-forward, `ESC E` reset, snapshots at the page ejects of `read_pcl5_pages`. Verify:
  - `testdata/invoice.pcl` gives A4 / portrait / auto / 1 copy;
  - PJL paper is overridden by `ESC&l26A`;
  - duplex carries forward;
  - the reset restores Letter from PJL;
  - a job without page size reports `unspecified`.
- [x] 4.4 Implement the PCL XL state (BeginPage attributes with PJL fallback, PageCopies). Verify `testdata/invoice_diff_format.pcl` gives A4 / portrait / auto / simplex and the vertical-binding fixture gives long_edge with 2 copies.
- [x] 4.5 Implement the PostScript state (PJL, DSC document and page features, `%%Orientation`, Ghostscript requests taking precedence, `settings_source`). Verify:
  - with Ghostscript, the DSC-contradiction fixture reports A4 from the page device;
  - without Ghostscript, it reports Letter from DSC with a DSC-only `settings_source`;
  - `*InputSlot Tray2` gives `ps:Tray2`.

## 5. Mapping

- [x] 5.1 Implement the mapping input (D8): a dictionary or JSON file; unknown sections rejected; mapped names reported in the ticket. Verify the `drawer2` scenario (PCL 5 source 4 and `ps:Tray2`), the JSON-file variant and the unknown-section error.

## 6. Comparison and keywords

- [x] 6.1 Implement the ticket comparison (D7): job, pjl and page scopes; missing pages; message format and 20-entry cap; ignore lists reused from `PrintJobTests._ignored_names`. Verify:
  - `invoice.pcl` vs `invoice_difftray.pcl` message contains `page 1: input_tray auto → pcl5:4` and no duplex difference (both print simplex);
  - the JOBNAME/USERNAME pair passes;
  - `ignore_properties=input_tray` passes a tray-only difference.
- [x] 6.2 Add `Get Print Job Ticket`, `Compare Print Job Tickets` and `Print Job Ticket Should Contain` to `DocTest/PrintJobTests.py`, with keyword docs including the cross-format `ignore_properties=pjl,resolution,color_mode` pattern. Verify:
  - the keyword scenarios of the spec as unit tests: the cross-format invoice pair passes; all-pages duplex fails naming page 2; the single-page check passes;
  - the keyword-surface and signature baselines regenerate with additions only.

## 7. Packaging, acceptance and documentation

- [x] 7.1 Add `DocTest/PrintJobTicket.py` to the wheel and sdist baselines. Verify the parity gate passes on a fresh `uv build` and that `uvx ruff check DocTest` and mypy are clean.
- [x] 7.2 Add acceptance tests to `atest/PrintJobs.robot`:
  - ticket of the PCL 5 invoice;
  - tray difference message;
  - cross-format pass;
  - `Print Job Ticket Should Contain` pass and fail;
  - a mapping case.

  Verify the suite passes locally with Ghostscript/GhostPCL 10.08.0 and on CI.
- [x] 7.3 Benchmark the PostScript capture on a ≥ 5 MB PostScript job and record the time in a test docstring, with a lenient upper bound. Verify the test passes locally.
- [x] 7.4 Add a README section "Assert and compare print settings (job tickets)" with the vocabulary, `unspecified`, mapping and cross-format example. Verify the example runs as written in a Robot dry run.

## Workflow follow-up

- Open a PR after implementation and local verification; mark 7.2's CI part after CI passes.
- Archive the change after merge (`/opsx:archive`), creating the `print-job-ticket` main spec.
