## Why

QA teams that test statement and invoice output need to assert how a job *prints*: paper size, duplex binding, tray, output bin, copies and media. Two things stand in the way:
- **Settings are scattered** across PJL job defaults and page-description commands that override them.
- **Value systems differ per language.** PCL 5 uses `ESC&l26A` for A4, PCL XL uses `MediaSize 2`, PostScript uses `PageSize [595 842]` or `*PageSize A4`.

`Compare Print Jobs` deliberately compares these raw, per-language values (see `print-job-comparison`). So it cannot tell users:
- what a page will effectively print with;
- whether a PCL job and its PostScript re-implementation request the same thing;
- which human-readable setting changed ("page 3: tray 2 → tray 1").

The exploration found no open-source tool, and almost no commercial one, that offers a normalized, per-page print-settings comparison. Building on the byte-level readers from `fix-print-job-false-passes`, this is the library's main differentiator for print-output testing.

## What Changes

- **New `Get Print Job Ticket` keyword.**
  - It returns a normalized ticket for a PCL 5, PCL XL or PostScript job: languages, page count, job-level settings and every PJL variable, plus one entry per page.
  - Each page entry holds the *effective* paper size, orientation, duplex mode, copies, input tray, output bin and media type, each with its source.
  - "Effective" means PJL defaults overridden by page-description commands, carried forward page by page, with resets honored.
- **New `Compare Print Job Tickets` keyword.**
  - It compares two tickets field by field and page by page, also across languages.
  - It fails with human-readable per-page differences such as `page 3: input_tray tray2 → tray1`.
  - It supports the same default ignore list and `ignore_properties` as `Compare Print Jobs`.
- **New `Print Job Ticket Should Contain` keyword.** It asserts expected settings on one page or all pages without a reference file, e.g. `duplex=long_edge  paper_size=A4`.
- **Normalized vocabulary.**
  - Standardized settings are compared across languages: named paper sizes (or millimetre dimensions for custom sizes), orientation, `simplex`/`long_edge`/`short_edge`, and copies.
  - Printer-specific settings (trays, bins, numeric media types) use generic names only where HP or Adobe define them (`auto`, `manual`, …). Otherwise they keep a language-qualified raw value.
  - An optional user-supplied mapping translates raw tray, bin and media values into the user's own names, which makes cross-language comparison of those values possible.
  - A setting a job leaves to the printer is reported as `unspecified`, not as a guessed default. It is compared like any other value, so it is never silently equal to a concrete setting.
- **PostScript settings.**
  - Settings come from DSC feature comments.
  - When Ghostscript is installed they also come from the settings the job actually requests (`setpagedevice`), which take precedence.
  - The ticket records which sources were used.
- **No behavior change** for `Compare Print Jobs`, `Get Pcl Print Job`, `Get Postscript Print Job` or `Check Print Job Property`.

## Capabilities

### New Capabilities
- `print-job-ticket`: the normalized, per-page print-settings model of a print job (PJL, PCL 5, PCL XL, PostScript), the effective-value rules, the normalized vocabulary and mapping, and the `Get Print Job Ticket`, `Compare Print Job Tickets` and `Print Job Ticket Should Contain` keywords.

### Modified Capabilities
(none)

## Impact

- **Code:**
  - New `DocTest/PrintJobTicket.py`: the ticket model, normalization tables, effective-value resolution and comparison.
  - `DocTest/PrintJobParsing.py` gets small additive extensions: command payload offsets for `ESC&n#Wd` media names; the PCL XL MediaDestination attribute; a Ghostscript `setpagedevice` capture.
  - The new keywords are exported from `DocTest/PrintJobTests.py`.
- **Public surface:** three new keywords. The keyword-surface and signature baselines change additively only.
- **Packaging:** the wheel-parity baseline gains the new module. No new dependencies.
- **Tests and docs:** unit tests per language and for the normalization tables, new fixtures, acceptance tests in `atest/PrintJobs.robot`, keyword docs, and a README section.
- **Depends on** `fix-print-job-false-passes` (PR #168), which must be merged and archived first.
- **Not in scope:**
  - text or content comparison;
  - finishing semantics beyond raw PJL values (staple and punch are available as raw PJL variables);
  - colour or ink checks;
  - print-job splitting and page alignment.
