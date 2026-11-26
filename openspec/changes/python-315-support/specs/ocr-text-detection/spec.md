# Spec Delta

## Purpose

Covers the EAST deep-learning text detector used for OCR and text masking: how its model file is provisioned and which image-processing runtime it requires.

## ADDED Requirements

### Requirement: Concurrency-safe EAST model provisioning
When the EAST model file is missing, the library SHALL download it so that the model path only ever contains a complete file. Concurrent processes provisioning the model at the same time (e.g. pytest-xdist workers or pabot) SHALL each end up loading a valid model.

#### Scenario: Parallel first use
- **WHEN** several processes use EAST text detection at the same time and no model file exists yet
- **THEN** every process loads the model successfully, and none fails with a model parse error

#### Scenario: Interrupted download leaves no corrupt model
- **WHEN** a model download is interrupted before completion
- **THEN** no file exists at the model path, and the next use downloads the model again

#### Scenario: Existing model is reused
- **WHEN** a complete model file already exists at the model path
- **THEN** no download is attempted

### Requirement: Image-processing runtime supports the EAST model format
The declared OpenCV dependency range SHALL be limited to releases that can load the shipped EAST model format (TensorFlow frozen graph) and that produce the template-matching results the movement-detection tests verify.

#### Scenario: EAST loads on a fresh install
- **WHEN** the library is installed into a fresh environment on any supported Python version and EAST text detection is used
- **THEN** the model loads without a "TensorFlow importer has been removed" error

#### Scenario: Movement detection results are stable
- **WHEN** the movement-tolerance tests run on a fresh install
- **THEN** template-based movement detection reports the same results as with the previously locked OpenCV version

### Requirement: Model download works in a clean interpreter
The automatic EAST model download SHALL NOT depend on any other module having imported networking helpers beforehand.

#### Scenario: First EAST use in a fresh process
- **WHEN** a fresh Python process with no model file present triggers EAST provisioning before any other library code has run
- **THEN** the download is attempted, with no `AttributeError` about `urllib.request`

### Requirement: Only recognized words are returned
EAST recognition results SHALL contain only words Tesseract actually recognized in a region. Tesseract's structural rows (page, block, paragraph and line entries without text) SHALL be excluded, and no OCR work SHALL be spent on them.

#### Scenario: Structural rows are excluded
- **WHEN** EAST text extraction runs on any image
- **THEN** every returned token has non-empty text, and no token comes from an OCR row Tesseract marks with confidence -1

### Requirement: Multi-line regions are fully recognized
When adjacent text lines merge into one detected region, every line in that region SHALL be recognized, not only one.

#### Scenario: Stacked lines of text
- **WHEN** EAST OCR runs on `text_small.png` or `text_big.png` (uppercase, lowercase, digits and symbol lines stacked closely)
- **THEN** the extracted text contains `ABCDEFGHI`, `abcdefghi` and `1234567890` with Tesseract 5.3 and 5.5

#### Scenario: No fragments from neighbouring lines
- **WHEN** a region containing several lines is recognized
- **THEN** its vertical margin is proportional to a single line's height and does not include partial glyphs of the next line

### Requirement: Small labels in busy images are recognized
When OCR of a detected region yields no alphanumeric word, the region SHALL be retried once with a tighter crop around the detected text, and the better result kept.

#### Scenario: Small sign on a coloured background
- **WHEN** EAST OCR runs on `Beach_date.png`
- **THEN** the extracted text contains `01-Jan-2021`, `123456789` and `SOUVENIR` (case-insensitive) with Tesseract 5.3 and 5.5
