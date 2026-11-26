*** Settings ***
Documentation    Cross-browser visual comparison: a chromium-created baseline
...              compared against a firefox capture of the same page. Rendering
...              noise (anti-aliasing) must be tolerable without hiding content
...              changes. Captures use the fixed viewport: full-page heights
...              differ by a pixel between engines on some platforms (Windows
...              font metrics), which is layout, not anti-aliasing.
Library    Browser
Library    DocTest.WebVisualTest    baseline_directory=${OUTPUT_DIR}${/}xbaselines    retry_timeout=0

*** Variables ***
${DEMO_PAGE}    ${CURDIR}${/}pages${/}demo.html

*** Keywords ***
Open Page In
    [Arguments]    ${engine}
    New Browser    ${engine}    headless=True
    New Context    viewport={'width': 900, 'height': 600}
    New Page    file://${DEMO_PAGE}

*** Test Cases ***
Chromium Creates The Baseline
    Open Page In    chromium
    Compare Page To Baseline    xbrowser-demo    full_page=False
    Close Browser

Firefox Fails Without Antialiasing Tolerance
    Open Page In    firefox
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Page To Baseline    xbrowser-demo    full_page=False

Firefox Passes With Antialiasing Tolerance
    Open Page In    firefox
    Compare Page To Baseline    xbrowser-demo    full_page=False    ignore_antialiasing=True    max_diff_pixels=100

Real Change Still Fails Cross Browser
    Open Page In    firefox
    Evaluate JavaScript    ${None}    () => document.getElementById('content').firstElementChild.textContent = 'SOMETHING COMPLETELY DIFFERENT'
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Page To Baseline    xbrowser-demo    full_page=False    ignore_antialiasing=True    max_diff_pixels=100
