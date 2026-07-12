*** Settings ***
Library    SeleniumLibrary
Library    DocTest.WebVisualTest    baseline_directory=${OUTPUT_DIR}${/}baselines    retry_timeout=1s
Library    OperatingSystem
Suite Setup       Open Browser    file://${CURDIR}${/}pages${/}demo.html    headlesschrome
Suite Teardown    Close All Browsers

*** Test Cases ***
First Run Creates Viewport Baseline
    Compare Page To Baseline    selenium-demo-page    full_page=False
    File Should Exist    ${OUTPUT_DIR}${/}baselines${/}selenium-demo-page.png

Unchanged Viewport Matches Baseline
    Compare Page To Baseline    selenium-demo-page    full_page=False

Element Baseline Roundtrip
    Compare Element To Baseline    id=header    selenium-demo-header
    Compare Element To Baseline    id=header    selenium-demo-header

Full Page Capture Is Rejected With Clear Error
    Run Keyword And Expect Error    *full-page*
    ...    Compare Page To Baseline    selenium-anything

Changed Page Fails
    Execute Javascript    document.getElementById('content').firstElementChild.textContent = 'SOMETHING COMPLETELY DIFFERENT'
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Page To Baseline    selenium-demo-page    full_page=False
    Execute Javascript    document.getElementById('content').firstElementChild.textContent = 'Stable content area with fixed text.'

Changed Region Passes With Ignore Elements
    Execute Javascript    document.getElementById('content').firstElementChild.textContent = 'SOMETHING COMPLETELY DIFFERENT'
    Compare Page To Baseline    selenium-demo-page    full_page=False    ignore_elements=id=content
    Execute Javascript    document.getElementById('content').firstElementChild.textContent = 'Stable content area with fixed text.'

Dom Snapshot Stored With Selenium
    Compare Page To Baseline    selenium-dom-demo    full_page=False    dom_analysis=True
    OperatingSystem.File Should Exist    ${OUTPUT_DIR}${/}baselines${/}selenium-dom-demo.dom.json
    Compare Page To Baseline    selenium-dom-demo    full_page=False    dom_analysis=True
