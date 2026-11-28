# environment-check Specification

## Purpose

Defines the `Check DocTest Environment` report: which external tools and Python packages DocTest features need, whether they are available, and how each tool is detected and its version probed.

## Requirements

### Requirement: Only genuine tools are reported as available
A fallback executable name SHALL only count as the capability when it is not a known, unrelated operating-system utility of the same name. On Windows, `convert.exe` from the system directory (a disk-conversion tool) SHALL NOT be reported as ImageMagick.

#### Scenario: Windows system convert.exe is ignored
- **WHEN** the environment check runs on Windows, where ImageMagick is not installed and `convert` resolves to `C:\Windows\System32\convert.exe`
- **THEN** ImageMagick is reported as missing, with its install hint

#### Scenario: Real ImageMagick fallback is accepted
- **WHEN** `magick` is absent but an ImageMagick `convert` outside the system directory is on the PATH
- **THEN** ImageMagick is reported as available, with a note that the fallback command is used

### Requirement: Version probing tolerates any output encoding
Probing a tool's version SHALL succeed or quietly fall back to no version, whatever encoding the tool writes. It SHALL NOT raise or emit decoding errors, including under Python 3.15's default UTF-8 mode.

#### Scenario: Localized non-UTF-8 output
- **WHEN** a probed binary prints `Ungültige Laufwerkangabe.` in code page 1252
- **THEN** the report is produced, and no `UnicodeDecodeError` is raised or reported from a background thread
