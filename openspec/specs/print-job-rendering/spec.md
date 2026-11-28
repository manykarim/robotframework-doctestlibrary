# print-job-rendering Specification

## Purpose

Defines how PCL and PostScript documents are rendered into pages for visual comparison, so that a rendering failure, a missing renderer or an incomplete render always surfaces as an error instead of a passing comparison.

## Requirements

### Requirement: Renderer failures are errors
Rendering a `.pcl` or `.ps` document SHALL fail with an error naming the document when the renderer exits with a non-zero status, cannot be started, or reports an interpreter error in its output. The error SHALL include the renderer's diagnostic output.

#### Scenario: Renderer cannot start
- **WHEN** the renderer executable is found but fails to start (for example a missing runtime library)
- **THEN** `Compare Images` fails with a rendering error, not a pass

#### Scenario: Corrupt PostScript
- **WHEN** a `.ps` document triggers an unrecoverable interpreter error
- **THEN** the comparison fails with an error that contains the interpreter's message

### Requirement: A render without pages is an error
Rendering a `.pcl` or `.ps` document that produces no pages SHALL fail with an error instead of yielding an empty document.

#### Scenario: Empty PCL file
- **WHEN** an empty `.pcl` file is compared with any document
- **THEN** the comparison fails with an error stating that no pages were rendered

#### Scenario: Two unrenderable documents
- **WHEN** both reference and candidate fail to render
- **THEN** the comparison fails instead of reporting them as equal

### Requirement: Rendered page count matches the job structure
When the page count of a PCL job can be determined from its command stream, the number of rendered pages SHALL equal it. Pages deliberately skipped by `ignore_printfactory_envelope` SHALL be accounted for. A mismatch SHALL fail with an error that states both counts.

#### Scenario: Render produces extra pages
- **WHEN** a PCL job whose command stream contains one page renders to more than one page
- **THEN** the comparison fails and reports the structural and rendered page counts

### Requirement: Paper default reaches the PCL renderer
The configured printing paper size (`printing_papersize`, default `a4`) SHALL be applied as the default paper for PCL rendering. A page size set by the job itself SHALL take precedence.

#### Scenario: Job without its own page size
- **WHEN** a PCL job that sets no page size is rendered with the default settings
- **THEN** the rendered page has A4 dimensions

#### Scenario: Job with its own page size
- **WHEN** a PCL job that selects Letter (`ESC&l2A`) is rendered with the default settings
- **THEN** the rendered page has Letter dimensions

### Requirement: Concurrent renders are isolated
Each render SHALL write its page images to a directory that no other render uses, and SHALL remove only that directory afterwards. Parallel comparisons of documents with the same file name SHALL therefore not affect each other's pages.

#### Scenario: Same file rendered in parallel
- **WHEN** two workers render documents with the same file name at the same time
- **THEN** each obtains its own complete set of pages
