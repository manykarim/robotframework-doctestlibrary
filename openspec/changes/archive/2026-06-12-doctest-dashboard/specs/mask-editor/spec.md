# Spec: mask-editor

## ADDED Requirements

### Requirement: Schema-exact masks.json round-trip
The editor SHALL read and write the exact mask schema consumed by `IgnoreAreaManager`/`DocumentRepresentation` — types `coordinates` (x/y/width/height, unit px/mm/cm/pt), `area` (location/percent), `pattern`, `line_pattern`, `word_pattern` (regex, optional xoffset/yoffset), each with `page` ("all", int, or numeric string) and optional `name` — introducing no proprietary fields. Loading a file and saving it without edits SHALL be content-preserving (stable key order, pretty-printed). Imports of the shorthand string form (`top:10;bottom:10`) SHALL be normalized to area masks; exports SHALL always be JSON.

#### Scenario: Round-trip preserves library compatibility
- **WHEN** an existing `masks.json` from the library's test data is loaded and saved unchanged
- **THEN** the saved file parses identically through `IgnoreAreaManager` and a comparison using it behaves identically

#### Scenario: Shorthand import
- **WHEN** a user imports the string `top:10;bottom:5`
- **THEN** the editor shows two area masks (top 10%, bottom 5%) and exports them as JSON area entries

### Requirement: Coordinate mask editing with unit/DPI fidelity
Coordinate masks SHALL be drawable by drag, adjustable via handles and synced numeric fields, with a unit selector. The stored value SHALL remain in the user's chosen unit; conversion SHALL happen for display only, using the rendering DPI from the comparison sidecar, which SHALL be displayed prominently.

#### Scenario: Millimetre mask displays correctly
- **WHEN** a user creates a 25.4 mm wide mask on a page whose sidecar DPI is 200
- **THEN** the canvas rectangle is 200 px wide, and the saved JSON stores `width: 25.4, unit: "mm"`

### Requirement: Area mask configuration with preview
Area masks SHALL be configured from a panel (location selector, percent slider, page scope toggle) and previewed as a translucent band on the page rendering.

#### Scenario: Percent slider preview
- **WHEN** a user sets a top area mask to 15 percent
- **THEN** a band covering the top 15% of the displayed page appears before saving

### Requirement: Pattern mask live preview
Pattern, line_pattern, and word_pattern masks SHALL offer a live match preview: the backend resolves the regex through the library's own text-extraction path for the displayed page and returns the bounding boxes that would be masked, which the editor highlights. Preview requests SHALL be debounced and results cached per (file, page, engine, pattern). When OCR capability is unavailable, pattern preview SHALL be disabled with an explanation while other mask types remain editable.

#### Scenario: Date pattern preview
- **WHEN** a user types a date regex matching visible text on the displayed page
- **THEN** the matched line/word bounding boxes are highlighted within the debounce-plus-processing interval

### Requirement: Create mask from diff region
In the diff viewer, each detected diff region SHALL offer a one-click "add ignore mask" action that opens the editor pre-seeded with a `coordinates` mask of that region plus configurable padding, targeting a masks.json chosen via a picker rooted at the configured data roots.

#### Scenario: One-click mask from diff
- **WHEN** a reviewer clicks "add ignore mask" on a diff region at (410, 36, 150, 42)
- **THEN** the editor opens with a coordinates mask covering at least that rectangle, ready to save to the selected masks.json

### Requirement: Safe file writes
Saving SHALL be atomic (write temp, rename), create a `.bak` of the previous content, and refuse paths outside configured roots.

#### Scenario: Backup on save
- **WHEN** a user saves changes to an existing masks.json
- **THEN** the new content is in place and the prior content exists alongside as a backup

### Requirement: End-to-end editor journey
The journey — open editor from a diff, draw/adjust masks of each type, live-preview a pattern, save, and verify a robot re-run with the saved file passes — SHALL be covered by automated browser tests using the real backend and library.

#### Scenario: Editor journey test passes
- **WHEN** the Playwright suite runs the mask-editing journey on the `birthday_1080` test pair
- **THEN** the saved masks.json makes the previously failing comparison pass in an actual library run
