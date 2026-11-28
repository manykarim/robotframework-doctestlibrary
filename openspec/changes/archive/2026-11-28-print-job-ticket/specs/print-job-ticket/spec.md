# Spec Delta

## Purpose

Provides a normalized, per-page view of how a print job prints (paper, orientation, duplex, copies, tray, bin, media) across PJL, PCL 5, PCL XL and PostScript, so print settings can be asserted and compared, also between jobs in different printer languages, with readable per-page differences.

## ADDED Requirements

### Requirement: Ticket content
`Get Print Job Ticket` SHALL return:
- the job's languages and page count;
- job-level settings: resolution, color mode and every PJL variable;
- one entry per page with paper size, orientation, duplex, copies, input tray, output bin and media type.

Each setting SHALL carry its value and the source it was taken from.

#### Scenario: PCL 5 invoice
- **WHEN** a ticket is read from `testdata/invoice.pcl`
- **THEN** it reports language PCL 5 and one page with paper size `A4` from the PCL commands, orientation `portrait`, input tray `auto` and copies `1`

#### Scenario: PJL variables are available
- **WHEN** a ticket is read from a job with `@PJL SET RESOLUTION=600`
- **THEN** the job-level resolution is `600` and the PJL variables include `RESOLUTION`

### Requirement: Page-description commands override PJL defaults
A setting given by a page-description command (PCL 5, PCL XL or PostScript) SHALL override the PJL value for the same setting from that point on.

#### Scenario: PCL page size overrides PJL paper
- **WHEN** a job sets `@PJL SET PAPER=LETTER` and then selects A4 with `ESC&l26A`
- **THEN** the page's paper size is `A4` with source PCL 5

### Requirement: Settings carry forward across pages
A setting that remains in effect until it is changed SHALL be reported on every following page until a later command changes it or a printer reset (`ESC E`) restores the job defaults.

#### Scenario: Duplex set once
- **WHEN** a 2-page PCL 5 job selects long-edge duplex before page 1 only
- **THEN** both pages report duplex `long_edge`

#### Scenario: Reset restores PJL defaults
- **WHEN** a PCL 5 job with `@PJL SET PAPER=LETTER` prints page 1 on A4 and issues `ESC E` before page 2
- **THEN** page 2 reports paper size `Letter` with source PJL

### Requirement: Unspecified settings are explicit
A setting that neither the job nor its language defines SHALL be reported as `unspecified` and compared like any other value. A setting whose default the language defines (PCL copies 1, simplex, portrait) SHALL be reported with that default and source `default`.

#### Scenario: PCL job without page size or PJL paper
- **WHEN** a ticket is read from a PCL 5 job that sets no page size and has no PJL `PAPER`
- **THEN** the paper size is `unspecified`, not a guessed printer default

#### Scenario: Unspecified differs from a concrete value
- **WHEN** a job with paper size `unspecified` is compared with a job printing on `A4`
- **THEN** `Compare Print Job Tickets` reports the paper size difference

### Requirement: Paper sizes are normalized
Paper sizes SHALL be reported as names (e.g. `A4`, `A3`, `A5`, `Letter`, `Legal`, `Executive`, `Ledger`, `Com10`, `DL`, `C5`, `Monarch`), whether they come from PCL 5 codes, PCL XL enumerations or names, PJL `PAPER` values or PostScript dimensions within 2 points of a named size. Other sizes SHALL be reported as width × height in millimetres.

#### Scenario: Same size in every language
- **WHEN** tickets are read for PCL 5 `ESC&l26A`, PCL XL MediaSize 2, PJL `PAPER=A4` and PostScript `PageSize [595 842]`
- **THEN** each reports paper size `A4`

#### Scenario: Custom PostScript size
- **WHEN** a PostScript job requests `PageSize [600 800]`
- **THEN** the paper size is reported as `212x282mm`

### Requirement: Duplex is normalized
Duplex SHALL be reported as `simplex`, `long_edge` or `short_edge` of the physical sheet. Sources are PCL 5 `ESC&l#S`, PCL XL `SimplexPageMode`/`DuplexPageMode`, PJL `DUPLEX` with `BINDING`, PostScript `Duplex` with `Tumble`, and DSC `*Duplex` options. PCL XL bindings and PostScript `Tumble` are relative to the page's orientation; they SHALL be converted to the sheet edge of the page they apply to.

#### Scenario: Long-edge binding in every language
- **WHEN** tickets are read for PCL 5 `ESC&l1S`, PCL XL vertical binding on a portrait page, PJL `DUPLEX=ON` with `BINDING=LONGEDGE`, PostScript `Duplex true Tumble false` on a portrait page size, and DSC `*Duplex DuplexNoTumble`
- **THEN** each reports duplex `long_edge`

#### Scenario: Page-relative binding on landscape pages
- **WHEN** a PCL XL landscape page requests vertical binding, or a PostScript job requests `Duplex true Tumble false` with a landscape page size
- **THEN** the duplex is reported as `short_edge`

### Requirement: Orientation is normalized
Orientation SHALL be reported as `portrait`, `landscape`, `reverse_portrait` or `reverse_landscape`. For PostScript it SHALL be taken from DSC `%%PageOrientation`/`%%Orientation` comments. Without such a comment, a requested page size SHALL determine it (width greater than height is `landscape`, otherwise `portrait`). Without either it SHALL be `unspecified`.

#### Scenario: PCL landscape
- **WHEN** a PCL 5 job selects `ESC&l1O`
- **THEN** the orientation is `landscape`

#### Scenario: PostScript landscape page size
- **WHEN** a PostScript job without orientation comments requests `PageSize [842 595]`
- **THEN** the paper size is `A4` and the orientation is `landscape`

### Requirement: Printer-specific values keep their raw identity
Input tray, output bin and media type SHALL be reported with a generic name only where the language specification defines one, such as `auto` and `manual` trays. Otherwise they SHALL be reported as a language-qualified raw value (e.g. `pcl5:4`, `ps:Tray2`). Raw values of different languages SHALL NOT be considered equal.

#### Scenario: Generic tray name
- **WHEN** a PCL 5 job selects paper source 7 and a PCL XL job selects MediaSource auto-select
- **THEN** both report input tray `auto`

#### Scenario: Model-specific tray
- **WHEN** a PCL 5 job selects paper source 4
- **THEN** the input tray is `pcl5:4`

### Requirement: User mapping translates printer-specific values
`Get Print Job Ticket`, `Compare Print Job Tickets` and `Print Job Ticket Should Contain` SHALL accept an optional mapping, given as a dictionary or a JSON file, from language-qualified raw tray, bin and media values to user-chosen names. Mapped values SHALL be reported and compared by their mapped name.

#### Scenario: Same drawer in PCL and PostScript
- **WHEN** a PCL 5 job selects paper source 4, a PostScript job selects `*InputSlot Tray2`, and the mapping maps `pcl5:4` and `ps:Tray2` to `drawer2`
- **THEN** both tickets report input tray `drawer2` and the comparison of that setting passes

### Requirement: PostScript settings use the job's own requests
For PostScript, settings SHALL be taken from DSC feature and page comments. When Ghostscript is available, the page-device settings the job requests while it runs SHALL also be read and SHALL take precedence over DSC comments. The ticket SHALL state whether interpreted requests were used.

#### Scenario: DSC comment contradicts the job
- **WHEN** a PostScript job declares `*PageSize Letter` in a feature comment but requests `PageSize [595 842]`, and Ghostscript is installed
- **THEN** the paper size is `A4` with source PostScript page device

#### Scenario: Without Ghostscript
- **WHEN** the same job is read and Ghostscript is not installed
- **THEN** the paper size is `Letter` with source DSC and the ticket states that only DSC comments were used

### Requirement: Ticket comparison reports readable per-page differences
`Compare Print Job Tickets` SHALL compare page count, job-level settings, PJL variables and every page setting of two jobs, also across languages. It SHALL fail with a message listing each difference as `page <n>: <setting> <reference value> → <candidate value>`, or as `job: …` for job-level settings, and SHALL pass when no difference remains.

#### Scenario: Tray changed on page 1
- **WHEN** `testdata/invoice.pcl` is compared with `testdata/invoice_difftray.pcl`
- **THEN** the keyword fails and the message contains `page 1: input_tray auto → pcl5:4`, and no duplex difference is reported because both pages print simplex (one by default, one by `ESC&l0S`)

#### Scenario: Same document in PCL 5 and PCL XL
- **WHEN** `testdata/invoice.pcl` is compared with its PCL XL rendition `testdata/invoice_diff_format.pcl` with `ignore_properties=pjl,resolution,color_mode`
- **THEN** the keyword passes

### Requirement: Ignore lists apply to tickets
`Compare Print Job Tickets` SHALL apply the default list of run-specific values used by `Compare Print Jobs` to PJL variables, and SHALL accept `ignore_properties` naming settings (e.g. `input_tray`), the group `pjl`, or individual PJL variables to exclude.

#### Scenario: Run-specific PJL values
- **WHEN** two jobs differ only in PJL `JOBNAME` and `USERNAME`
- **THEN** `Compare Print Job Tickets` passes

#### Scenario: Ignoring a setting
- **WHEN** two jobs differ only in input tray and `ignore_properties=input_tray` is given
- **THEN** `Compare Print Job Tickets` passes

### Requirement: Expected settings can be asserted without a reference
`Print Job Ticket Should Contain` SHALL check named expected settings (e.g. `duplex=long_edge`, `paper_size=A4`) on all pages, or on the page given by `page`. It SHALL fail listing every page and setting whose value differs.

#### Scenario: All pages duplex
- **WHEN** `Print Job Ticket Should Contain` is called with `duplex=long_edge` on a job whose page 2 is simplex
- **THEN** the keyword fails and names page 2 and its actual duplex value

#### Scenario: Single page
- **WHEN** it is called with `page=1  paper_size=A4` on a job whose page 1 is A4
- **THEN** the keyword passes
