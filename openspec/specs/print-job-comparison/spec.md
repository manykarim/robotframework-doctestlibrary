# print-job-comparison Specification

## Purpose

Defines what the print-job keywords extract from PCL 5, PCL XL and PostScript jobs and how `Compare Print Jobs` decides two jobs are equal, so that a material difference in page count, page setup or job structure is never reported as a pass.

## Requirements

### Requirement: Page count is always compared
`Get Pcl Print Job` and `Get Postscript Print Job` SHALL return a `page_count` property for every job they accept, and `Compare Print Jobs` SHALL fail when the reference and candidate page counts differ.

#### Scenario: PostScript jobs with different page counts
- **WHEN** a 1-page and a 2-page DSC-conforming PostScript job with otherwise identical comments are compared
- **THEN** the keyword fails and the logged differences name `page_count` with both values

#### Scenario: PCL jobs with different page counts
- **WHEN** a 1-page and a 2-page PCL 5 job whose page setup commands are identical are compared
- **THEN** the keyword fails and the logged differences name `page_count`

### Requirement: PCL page boundaries ignore binary payloads
PCL 5 page attribution SHALL advance only on page ejects in the command stream, never on bytes inside the counted binary payload of a command such as a soft-font, character or raster data transfer.

#### Scenario: Form feed bytes inside a soft-font download
- **WHEN** a 2-page PCL 5 job downloads a character whose data contains form-feed bytes and selects landscape orientation on page 2
- **THEN** `page_count` is 2 and the landscape orientation is reported for page 2

### Requirement: PCL command values are not range-restricted
Extracted PCL 5 page-setup commands (paper source, paper format, orientation, duplex, copies) SHALL be reported with whatever numeric value the job contains, so a value outside a hard-coded range is never dropped.

#### Scenario: Tray value above 6
- **WHEN** a PCL 5 job selects paper source 7 (`ESC&l7H`)
- **THEN** the job's properties include paper source 7 for that page

### Requirement: PCL XL jobs are decoded
`Get Pcl Print Job` SHALL decode PCL XL (PCL 6) streams. It SHALL report the page count and, per page, paper size, paper source, orientation, duplex mode and copies, using the same property names as for PCL 5 jobs.

#### Scenario: PCL XL page attributes
- **WHEN** the repository's PCL XL invoice (`testdata/invoice_diff_format.pcl`) is read
- **THEN** the properties report one page with its paper size, paper source, orientation and simplex/duplex mode

#### Scenario: Modified PCL XL job is not equal
- **WHEN** a PCL XL job is compared with a copy whose page attributes or page count differ
- **THEN** `Compare Print Jobs` fails

### Requirement: Job language is part of the comparison
The PCL job properties SHALL include the page-description language of each job (PCL 5 or PCL XL), and `Compare Print Jobs` SHALL report a language difference between reference and candidate.

#### Scenario: PCL 5 versus PCL XL rendition of the same document
- **WHEN** a PCL 5 job is compared with a PCL XL job
- **THEN** the keyword fails and the logged differences include the language

### Requirement: Common PostScript job shapes are accepted
`Get Postscript Print Job` SHALL accept PostScript jobs in the shapes printer drivers and tools commonly emit, and SHALL NOT fail with a parse error on them. The accepted shapes are listed in the scenarios.

#### Scenario: Driver and tool variants parse
- **WHEN** jobs are read that are PJL-wrapped, use CRLF or CR line endings, omit the prolog or setup section, use an `EPSF` or `PS-Adobe-2.0` header line, defer `%%Pages` to the trailer with `(atend)`, contain a nested `%%BeginDocument`, carry `%%BeginFeature` blocks inside `%%BeginSetup` or `%%BeginPageSetup`, or lack a final newline
- **THEN** each job is read without error and its page count and comments are reported

#### Scenario: Deferred page count is resolved
- **WHEN** a job declares `%%Pages: (atend)` and states the count in its trailer
- **THEN** the reported `page_count` is the trailer value

### Requirement: PostScript header and features are compared
`Compare Print Jobs` SHALL compare the PostScript `header` comments, the document and page `features`, and the per-page comments (media, orientation, bounding box), in addition to the PJL commands and trailer it already compares.

#### Scenario: Different bounding box
- **WHEN** two PostScript jobs differ only in `%%BoundingBox` (A4 versus Letter)
- **THEN** the keyword fails and names the `BoundingBox` difference

#### Scenario: Different requested feature
- **WHEN** two PostScript jobs differ only in a `%%BeginFeature: *Duplex` option
- **THEN** the keyword fails and names the feature difference

### Requirement: PostScript jobs without page structure are not silently equal
When a PostScript job has no DSC page structure, the page count SHALL be determined by interpreting the job. If it cannot be determined, reading the job SHALL fail with an error that names the missing page structure, instead of returning an empty page list.

#### Scenario: Non-DSC job with the interpreter available
- **WHEN** a 2-page PostScript job without `%%Page` comments is read and Ghostscript is installed
- **THEN** `page_count` is 2

#### Scenario: Non-DSC job without the interpreter
- **WHEN** the same job is read and Ghostscript is not installed
- **THEN** an error explains that the page count cannot be determined and that Ghostscript is required

### Requirement: Run-specific fields are ignored by default
`Compare Print Jobs` SHALL ignore, by default, fields whose values change on every run of the same job:
- DSC `CreationDate`, `Title`, `For`, `Routing` and `Invocation`;
- PJL job name, user name, host, job identifier, date and time values.

#### Scenario: Same document printed twice
- **WHEN** two PostScript jobs differ only in `%%CreationDate`, `%%Title` and the PJL `JOBNAME`/`USERNAME` values
- **THEN** the keyword passes

### Requirement: Callers can exclude further properties
`Compare Print Jobs` SHALL accept an optional `ignore_properties` argument listing additional property or comment names to exclude from the comparison. Omitting it SHALL keep the default ignore list only.

#### Scenario: Excluding a known-volatile comment
- **WHEN** two PostScript jobs differ only in `%%Creator` and `ignore_properties` contains `Creator`
- **THEN** the keyword passes

#### Scenario: Default behavior without the argument
- **WHEN** the same two jobs are compared without `ignore_properties`
- **THEN** the keyword fails and names the `Creator` difference

### Requirement: Documented properties exist
Every property name that the keyword documentation of `Compare Print Jobs` and `Check Print Job Property` lists for a job type SHALL be present in the properties returned for that job type.

#### Scenario: PostScript header property
- **WHEN** `Check Print Job Property` is called on a PostScript job for property `header` with a comment the job contains
- **THEN** the check passes instead of reporting that the property does not exist
