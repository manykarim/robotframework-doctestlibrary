# Spec Delta

## MODIFIED Requirements

### Requirement: Dependency hygiene
The runtime dependency set SHALL NOT contain packages that are never imported. Every runtime dependency SHALL declare a tested lower version bound. Barcode (`pylibdmtx`, `pyzbar`) dependencies SHALL additionally be exposed as the optional extra `barcode` while remaining in the base set until a documented future release removes them. The print-job parser no longer needs `parsimonious`; it is removed from the runtime dependencies in this documented release, and the `printjobs` extra SHALL remain installable without dependencies so existing install commands keep working.

#### Scenario: Extras install the niche dependencies
- **WHEN** a user installs `robotframework-doctestlibrary[barcode]`
- **THEN** pylibdmtx and pyzbar are installed

#### Scenario: Legacy print-job extra still installs
- **WHEN** a user installs `robotframework-doctestlibrary[printjobs]`
- **THEN** the installation succeeds and the print-job keywords work without `parsimonious`

#### Scenario: No dead dependencies
- **WHEN** the dependency audit runs
- **THEN** every base dependency is imported somewhere in the shipped packages or explicitly documented as a system-integration requirement

### Requirement: First-party code is free of interpreter-level deprecations
On every supported Python version, the shipped packages SHALL import and run without SyntaxWarnings (e.g. invalid escape sequences in string or regular-expression literals) and without deprecation warnings caused by first-party use of a deprecated third-party API for which a replacement is available.

#### Scenario: No SyntaxWarning on the newest interpreter
- **WHEN** the print-job keywords module is imported and every module is compiled under Python 3.15 with SyntaxWarnings treated as errors
- **THEN** no error is raised

#### Scenario: Current PDF API is used
- **WHEN** the library is imported with PyMuPDF 1.28 or newer
- **THEN** no "`fitz` API is deprecated" warning is emitted
