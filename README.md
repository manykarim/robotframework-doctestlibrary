# robotframework-doctestlibrary
----

[Robot Framework](https://robotframework.org) DocTest library.
Simple Automated Visual Document Testing.

Powered by
- Open CV
- scikit-image
- ImageMagick
- Ghostscript
- PyWand
- Tesseract OCR
- pdfminer
- parsimonious
- The knowledge of stackoverflow.com

See [keyword documentation](https://github.com/manykarim/robotframework-doctestlibrary/DocTest.html).

# Installation instructions

Only Python 3.X or newer is supported.

1. Clone the robotframework-doctestlibrary `git clone https://github.com/manykarim/robotframework-doctestlibrary.git`
2. Install robotframework-doctestlibrary from the commandline: `python setup.py install`
3. Install Tesseract, Ghoscript and GhostPCL binaries

Or use the [docker images](https://github.com/manykarim/robotframework-doctestlibrary/packages).

# Examples

### Testing with [Robot Framework](https://robotframework.org)
```RobotFramework
*** Settings ***
Library    DocTest.VisualTest

*** Test Cases ***
Compare two Images and highlight differences
    Compare Images    Reference.jpg    Candidate.jpg
```

### Use masks/placeholders to exclude parts from visual comparison

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest

*** Test Cases ***
Compare two Images and ignore parts by using masks
    Compare Images    Reference.jpg    Candidate.jpg    placeholder_file=masks.json

Compare two PDF Docments and ignore parts by using masks
    Compare Images    Reference.jpg    Candidate.jpg    placeholder_file=masks.json
```
#### Different Mask Types to ignore parts from comparison
##### Areas, Coordinates, Text Patterns
```python
[
    {
		"page": "all",
		"name": "Date Pattern",
		"type": "pattern",
		"pattern": ".*[0-9]{2}-[a-zA-Z]{3}-[0-9]{4}.*"
    },
    {
		"page": "1",
		"name": "Top Border",
		"type": "area",
        "location": "top",
        "percent":  5
    }
    {
		"page": "1",
		"name": "Left Border",
		"type": "area",
        "location": "left",
        "percent":  5
    },
    {
		"page": 1,
		"name": "Top Rectangle",
		"type": "coordinates",
		"x": 0,
		"y": 0,
		"height": 10,
		"width": 210,
		"unit": "mm"
	}
]
```
### Accept visual different by checking move distance or text content

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest

*** Test Cases ***
Accept if parts are moved up to 20 pixels by pure visual check
    Compare Images    Reference.jpg    Candidate.jpg    move_tolerance=20

Accept if parts are moved up to 20 pixels by reading PDF Data
    Compare Images    Reference.pdf    Candidate.pdf    move_tolerance=20    get_pdf_content=${true}

Accept differences if text content is the same via OCR
    Compare Images    Reference.jpg    Candidate.jpg    check_text_content=${true}

Accept differences if text content is the same from PDF Data
    Compare Images    Reference.pdf    Candidate.pdf    check_text_content=${true}    get_pdf_content=${true}
```
### Options for taking additional screenshots, screenshot format and render resolution

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest   take_screenshots=${true}    screenshot_format=png
```

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest   show_diff=${true}    DPI=300
```

### Check content of PDF files

```RobotFramework
*** Settings ***
Library    DocTest.PdfTest

*** Test Cases ***
Check if list of strings exists in PDF File
    @{strings}=    Create List    First String    Second String
    Check Text Content    ${strings}    Candidate.pdf
```


# Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.

## Core team

In order of appearance.

  * Many Kasiriha

## Contributors

This project is community driven and becomes a reality only through the work of all the people who contribute.