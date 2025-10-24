"""
# robotframework-doctestlibrary
----
[Robot Framework](https://robotframework.org) DocTest library.  
Simple Automated Visual Document Testing.

See **keyword documentation** for

- [Visual Document Tests](https://manykarim.github.io/robotframework-doctestlibrary/VisualTest.html)
- [Print Job Tests](https://manykarim.github.io/robotframework-doctestlibrary/PrintJobTest.html)
- [Pdf Tests (very basic)](https://manykarim.github.io/robotframework-doctestlibrary/PdfTest.html)
- [Ai Keywords (optional)](https://manykarim.github.io/robotframework-doctestlibrary/Ai.html)

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest

*** Test Cases ***
Compare two Images and highlight differences
    Compare Images    Reference.jpg    Candidate.jpg
```

[Optional LLM for image comparison](https://manykarim.github.io/robotframework-doctestlibrary/Ai.html)

```RobotFramework
*** Settings ***
Library    DocTest.Ai
Library    DocTest.VisualTest
Library    DocTest.PdfTest

*** Test Cases ***
Review Visual Differences With LLM
    Compare Images With LLM    Reference.pdf    Candidate.pdf    llm_override=${True}

Extract Text From Document With LLM
    ${text}=    Get Text With LLM    Candidate.pdf    prompt=Return text and table contents
    Log    ${text}

Image Should Contain Object With LLM
    Image Should Contain    Candidate.png    Missing product logo

Count Items With LLM
    ${count}=    Get Item Count From Image    Candidate.png    item_description=number of pallets
    Should Be True    ${count} >= 0
```

[![DocTest Library presentation at robocon.io 2021](https://img.youtube.com/vi/qmpwlQoJ-nE/0.jpg)](https://youtu.be/qmpwlQoJ-nE "DocTest Library presentation at robocon.io 2021")

# Installation instructions

`pip install --upgrade robotframework-doctestlibrary`

## Optional LLM-Assisted Comparisons

You can optionally rely on a large language model to review detected differences and
decide whether a comparison should pass. This path is fully opt-in; nothing changes
for users who skip these dependencies.

1. Install the optional extra only when needed:
   ```bash
   pip install "robotframework-doctestlibrary[ai]"
   ```
2. Create a `.env` file at the repository root (values here override existing environment variables):
   ```ini
   # OpenAI-compatible endpoints
   OPENAI_API_KEY=sk-...
   DOCTEST_LLM_MODEL=gpt-5,gpt-4o
   DOCTEST_LLM_VISION_MODEL=gpt-5-mini,gpt-4o-mini
   ```
   Azure OpenAI deployments can be configured with:
   ```ini
   AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_OPENAI_API_KEY=...
   AZURE_OPENAI_DEPLOYMENT=gpt-4o
   AZURE_OPENAI_API_VERSION=2024-06-01
   DOCTEST_LLM_PROVIDER=azure
   ```
   See `.env.example` for a combined template covering both providers.
3. Use the dedicated keywords (or pass `llm_enabled=${True}` to existing ones):

   ```RobotFramework
   *** Test Cases ***
   Review Visual Differences With LLM
       Compare Images With LLM    Reference.pdf    Candidate.pdf    llm_override=${True}

   Review Pdf Structure With LLM
       Compare Pdf Documents With LLM    reference.pdf    candidate.pdf    compare=structure
   ```

Set `llm_override=${True}` when an LLM approval should override SSIM/DeepDiff failures.
Without the override flag the AI feedback is logged for investigation while the original
assertion result is preserved.

Pass `llm_prompt=` (or the specialty variants `llm_visual_prompt=` / `llm_pdf_prompt=`) to
customise the prompt sent to the model for a particular comparison.

Only Python 3.X or newer is supported. Tested with Python 3.8/3.11/3.12

## Install robotframework-doctestlibrary

### Installation via `pip` from PyPI (recommended)

* `pip install --upgrade robotframework-doctestlibrary`

### Installation via `pip` from GitHub

* `pip install git+https://github.com/manykarim/robotframework-doctestlibrary.git`  

or

* `git clone https://github.com/manykarim/robotframework-doctestlibrary.git`
* `cd robotframework-doctestlibrary`
* `pip install -e .`

## Install dependencies

Install Tesseract, Ghostscript, GhostPCL, ImageMagick binaries and barcode libraries (libdmtx, zbar) on your system.
<br>Hint: Since `0.2.0` Ghostscript, GhostPCL and ImageMagick are only needed for rendering `.ps` and `.pcl`files.
<br> Rendering and content parsing of `.pdf` is done via `MuPDF`
<br>In the future there might be a separate pypi package for `.pcl` and `.ps` files to get rid of those dependencies.

Linux
```bash
apt-get install imagemagick tesseract-ocr ghostscript libdmtx0b libzbar0
```

Windows
 * https://github.com/UB-Mannheim/tesseract/wiki
 * https://ghostscript.com/releases/gsdnld.html
 * https://ghostscript.com/releases/gpcldnld.html
 * https://imagemagick.org/script/download.php


## Some special instructions for Windows 

### Rename executable for GhostPCL to pcl6.exe (only needed for `.pcl` support)
The executable for GhostPCL `gpcl6win64.exe` needs to be renamed to `pcl6.exe`

Otherwise it will not be possible to render .pcl files successfully for visual comparison.

### Add tesseract, ghostscript and imagemagick to system path in windows (only needed for OCR, `.pcl` and `.ps` support)
* C:\\Program Files\\ImageMagick-7.0.10-Q16-HDRI
* C:\\Program Files\\Tesseract-OCR
* C:\\Program Files\\gs\\gs9.53.1\\bin
* C:\\Program Files\\gs\\ghostpcl-9.53.1-win64

(The folder names and versions on your system might be different)

That means: When you open the CMD shell you can run the commands
* `magick.exe`
* `tesseract.exe`
* `gswin64.exe`
* `pcl6.exe`

successfully from any folder/location

### Windows error message regarding pylibdmtx

[How to solve ImportError for pylibdmtx](https://github.com/NaturalHistoryMuseum/pylibdmtx/#windows-error-message)

If you see an ugly `ImportError` when importing `pylibdmtx` on
Windows you will most likely need the [Visual C++ Redistributable Packages for
Visual Studio 2013](https://www.microsoft.com/en-US/download/details.aspx?id=40784). Install `vcredist_x64.exe` if using 64-bit Python, `vcredist_x86.exe` if using 32-bit Python.

## ImageMagick

The library might return the error `File could not be converted by ImageMagick to OpenCV Image: <path to the file>` when comparing PDF files.
This is due to ImageMagick permissions. Verify this as follows with the `sample.pdf` in the `testdata` directory:
```bash
convert sample.pdf sample.jpg 
convert-im6.q16: attempt to perform an operation not allowed by the security policy
```

Solution is to copy the `policy.xml` from the repository to the ImageMagick installation directory.

## Docker

You can also use the [docker images](https://github.com/manykarim/robotframework-doctestlibrary/packages) or create your own Docker Image
`docker build -t robotframework-doctest .`
Afterwards you can, e.g., start the container and run the povided examples like this:
* Windows
  * `docker run -t -v "%cd%":/opt/test -w /opt/test robotframework-doctest robot atest/Compare.robot`
* Linux
  * `docker run -t -v $PWD:/opt/test -w /opt/test robotframework-doctest robot atest/Compare.robot`

## Gitpod.io
[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/manykarim/robotframework-doctestlibrary)  
Try out the library using [Gitpod](https://gitpod.io/#https://github.com/manykarim/robotframework-doctestlibrary)

# Examples

Have a look at  
* [Visual Comparison Tests](./atest/Compare.robot)
* [PDF Content Tests](./atest/PdfContent.robot)
* [Print Job Tests](./atest/PrintJobs.robot)
* [AI-Assisted Tests](./atest/LLM.robot)

for more examples.

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

Compare two Farm images with date pattern
    Compare Images    Reference.jpg    Candidate.jpg    placeholder_file=testdata/pattern_mask.json

Compare two Farm images with area mask as list
    ${top_mask}    Create Dictionary    page=1    type=area    location=top    percent=10
    ${bottom_mask}    Create Dictionary    page=all    type=area    location=bottom    percent=10
    ${masks}    Create List    ${top_mask}    ${bottom_mask}
    Compare Images    Reference.jpg    Candidate.jpg    mask=${masks}

Compare two Farm images with area mask as string
    Compare Images    Reference.jpg    Candidate.jpg    mask=top:10;bottom:10

```
#### Different Mask Types to Ignore Parts When Comparing
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
    },
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

#### Different options to detect moved parts/objects

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest   movement_detection=orb

*** Test Cases ***
Accept if parts are moved up to 20 pixels by pure visual check
    Compare Images    Reference.jpg    Candidate.jpg    move_tolerance=20
```	

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest   movement_detection=template

*** Test Cases ***
Accept if parts are moved up to 20 pixels by pure visual check
    Compare Images    Reference.jpg    Candidate.jpg    move_tolerance=20
```	

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest   movement_detection=classic

*** Test Cases ***
Accept if parts are moved up to 20 pixels by pure visual check
    Compare Images    Reference.jpg    Candidate.jpg    move_tolerance=20
```	

### Options for taking additional screenshots, screenshot format and render resolution

Take additional screenshots or reference and candidate file.

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest   take_screenshots=${true}    screenshot_format=png
```
Take diff screenshots to highlight differences

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest   show_diff=${true}    DPI=300
```

### Experimental usage of Open CV East Text Detection to improve OCR

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest

*** Test Cases ***
Compare two Farm images with date pattern and east detection
    Compare Images    Reference.jpg    Candidate.jpg    placeholder_file=masks.json    ocr_engine=east
```

### Check content of PDF files

```RobotFramework
*** Settings ***
Library    DocTest.PdfTest

*** Test Cases ***
Check if list of strings exists in PDF File
    @{strings}=    Create List    First String    Second String
    PDF Should Contain Strings    ${strings}    Candidate.pdf
    
Compare two PDF Files and only check text content
    Compare Pdf Documents    Reference.pdf    Candidate.pdf    compare=text

Compare two  PDF Files and only check text content and metadata
    Compare Pdf Documents    Reference.pdf    Candidate.pdf    compare=text,metadata
    
Compare two  PDF Files and check all possible content
    Compare Pdf Documents    Reference.pdf    Candidate.pdf
```

### Ignore Watermarks for Visual Comparisons
Store the watermark in a separate B/W image or PDF.
<br>
Watermark area needs to be filled with black color.
<br>
Watermark content will be subtracted from Visual Comparison result.

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest

*** Test Cases ***
Compare two Images and ignore jpg watermark
    Compare Images    Reference.jpg    Candidate.jpg    watermark_file=Watermark.jpg

Compare two Images and ignore pdf watermark
    Compare Images    Reference.pdf    Candidate.pdf    watermark_file=Watermark.pdf

Compare two Images and ignore watermark folder
    Compare Images    Reference.pdf    Candidate.pdf    watermark_file=${CURDIR}${/}watermarks
```

Watermarks can also be passed on Library import. This setting will apply to all Test Cases in Test Suite

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest   watermark_file=${CURDIR}${/}watermarks

*** Test Cases ***
Compare two Images and ignore watermarks
    Compare Images    Reference.jpg    Candidate.jpg
```

### Get Text From Documents or Images

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest

*** Test Cases ***
Get Text Content And Compare
    ${text}    Get Text From Document    Reference.pdf
    List Should Contain Value    ${text}    Test String
```

### Get Barcodes From Documents or Images

```RobotFramework
*** Settings ***
Library    DocTest.VisualTest

*** Test Cases ***
Get Text Content And Compare
    ${text}    Get Barcodes From Document    reference.jpg
    List Should Contain Value    ${text}    123456789
```

### Using pabot to run tests in parallel

Document Testing can be run in parallel using [pabot](https://pabot.org/).  
However, you need to pass the additional arguments `--artifacts` and `--artifactsinsubfolders` to the `pabot` command, to move the screenshots to the correct subfolder.  
Otherwise the screenshots will not be visible in the `log.html`

```
pabot --testlevelsplit --processes 8 --artifacts png,jpg,pdf,xml --artifactsinsubfolders /path/to/your/tests/
```

### Visual Testing of Web Applications

I experimented a bit and tried to use this library for Visual Testing of Web Applications.  
Please have a look at this pilot example [here](https://github.com/manykarim/robotframework-doctestlibrary/blob/main/atest/Browser.robot)

# Development

Feel free to create issues or pull requests.  
I'm always happy for any feedback.

## Core team

In order of appearance.

  * Many Kasiriha
  * April Wang

## Contributors

This project is community driven and becomes a reality only through the work of all the people who contribute.
"""
try:  # pragma: no cover
    from DocTest.llm import ensure_dotenv_loaded  # type: ignore
except Exception:  # pragma: no cover - LLM extra not installed
    ensure_dotenv_loaded = None  # type: ignore

if ensure_dotenv_loaded is not None:  # pragma: no cover - optional
    try:
        ensure_dotenv_loaded()
    except Exception:
        pass

from importlib import metadata

try:
    __version__ = metadata.version("robotframework-doctestlibrary")
except metadata.PackageNotFoundError:
    pass
