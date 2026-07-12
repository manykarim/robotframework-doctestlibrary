*** Settings ***
Library    Browser
Library    DocTest.WebVisualTest    baseline_directory=${OUTPUT_DIR}${/}baselines    retry_timeout=1s    result_json=true
Library    OperatingSystem
Suite Setup    Open Demo Page

*** Variables ***
${DEMO_PAGE}    ${CURDIR}${/}pages${/}demo.html

*** Keywords ***
Open Demo Page
    New Browser    chromium    headless=True
    New Context    viewport={'width': 900, 'height': 600}
    New Page    file://${DEMO_PAGE}

Mutate Content
    Evaluate JavaScript    ${None}    () => document.getElementById('content').firstElementChild.textContent = 'SOMETHING COMPLETELY DIFFERENT'

Restore Content
    Evaluate JavaScript    ${None}    () => document.getElementById('content').firstElementChild.textContent = 'Stable content area with fixed text.'

*** Test Cases ***
First Run Creates Page Baseline
    Compare Page To Baseline    browser-demo-page
    File Should Exist    ${OUTPUT_DIR}${/}baselines${/}browser-demo-page.png

Unchanged Page Matches Baseline
    Compare Page To Baseline    browser-demo-page

Element Baseline Roundtrip
    Compare Element To Baseline    id=header    browser-demo-header
    Compare Element To Baseline    id=header    browser-demo-header

Changed Page Fails And Recovered Page Passes
    Mutate Content
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Page To Baseline    browser-demo-page
    Restore Content
    Compare Page To Baseline    browser-demo-page

Changed Region Passes With Mask
    Mutate Content
    ${mask}=    Evaluate    {"page": "all", "type": "coordinates", "unit": "px", "x": 0, "y": 180, "width": 900, "height": 420}
    Compare Page To Baseline    browser-demo-page    mask=${mask}
    Restore Content

Changed Region Passes With Ignore Elements
    Mutate Content
    Compare Page To Baseline    browser-demo-page    ignore_elements=id=content
    Restore Content

Element Comparison With Ignored Child
    Compare Element To Baseline    id=hero    browser-demo-hero
    Evaluate JavaScript    ${None}    () => document.querySelector('#hero h2').textContent = 'DIFFERENT HEADLINE'
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Element To Baseline    id=hero    browser-demo-hero
    Compare Element To Baseline    id=hero    browser-demo-hero    ignore_elements=css=#hero h2
    Evaluate JavaScript    ${None}    () => document.querySelector('#hero h2').textContent = 'Visual testing, reviewed properly'

Sidecar Records Capture Context
    Compare Page To Baseline    browser-demo-page
    ${files}=    List Files In Directory    ${OUTPUT_DIR}${/}doctest_results    *.json    absolute=${True}
    ${sidecars}=    Evaluate    [f for f in $files if not f.endswith('run.json')]
    ${latest}=    Evaluate    sorted(${sidecars}, key=lambda p: __import__('os').path.getmtime(p))[-1]
    ${content}=    Evaluate    json.load(open(r"""${latest}"""))    modules=json
    Should Be Equal    ${content}[context][library]    Browser
    Should Be Equal    ${content}[context][browser]    chromium
    Should Be Equal    ${content}[context][viewport]    900x600
    Should Not Be Empty    ${content}[context][url]

Tiny Rendering Noise Accepted With Pixel Budget
    Evaluate JavaScript    ${None}    () => { const d = document.createElement('div'); d.id = 'speck'; d.style.cssText = 'position:absolute;top:5px;left:5px;width:6px;height:6px;background:#000'; document.body.appendChild(d); }
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Page To Baseline    browser-demo-page
    Compare Page To Baseline    browser-demo-page    max_diff_pixels=500
    Compare Page To Baseline    browser-demo-page    max_diff_ratio=0.001
    Evaluate JavaScript    ${None}    () => document.getElementById('speck').remove()

Rendering Only Change Accepted With Dom Analysis
    Compare Page To Baseline    browser-dom-demo    dom_analysis=True
    Evaluate JavaScript    ${None}    () => document.querySelector('header').style.background = '#dc2626'
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Page To Baseline    browser-dom-demo    dom_analysis=True
    Compare Page To Baseline    browser-dom-demo    dom_analysis=True    accept_rendering_only=True
    Evaluate JavaScript    ${None}    () => document.querySelector('header').style.background = '#2563eb'

Semantic Change Never Auto Accepted
    Compare Page To Baseline    browser-dom-sem    dom_analysis=True
    Evaluate JavaScript    ${None}    () => document.getElementById('content').firstElementChild.textContent = 'ENTIRELY NEW CONTENT'
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Page To Baseline    browser-dom-sem    dom_analysis=True    accept_rendering_only=True
    Evaluate JavaScript    ${None}    () => document.getElementById('content').firstElementChild.textContent = 'Stable content area with fixed text.'
