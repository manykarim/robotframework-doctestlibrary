# Spec Delta

## ADDED Requirements

### Requirement: Capture arguments reach the web library verbatim
Screenshot paths, element locators and injected scripts that the web keywords pass to Browser Library or SeleniumLibrary SHALL arrive exactly as given on every platform. Robot Framework escape sequences and variable syntax inside them SHALL NOT be interpreted.

#### Scenario: Windows output directory
- **WHEN** `Compare Page To Baseline` runs on Windows with an output directory such as `C:\work\new\results`
- **THEN** the screenshot is written to that directory, and no path segment is altered by escape processing (e.g. `\n` turning into a newline)

#### Scenario: Locator with backslash and equals sign
- **WHEN** an element is captured with a locator containing `=` and a backslash
- **THEN** the web library receives the identical locator string

#### Scenario: DOM snapshot text is not altered
- **WHEN** a DOM snapshot is stored for a comparison
- **THEN** the walker script runs with its whitespace-collapsing pattern intact, and the stored text keeps every letter "s"
