*** Settings ***
Documentation    Runnable tour of DocTest web visual testing.
...
...    Setup:   pip install robotframework-doctestlibrary robotframework-browser
...             rfbrowser init
...    Run:     robot web_visual_demo.robot
...
...    First run creates all baselines under ./visual_baselines (passes with
...    warnings) — run twice to see real comparisons. Delete a baseline or run
...    with --variable REFERENCE_RUN:True to refresh it. Add result_json=true
...    and ingest output.xml into `doctest-dashboard serve` for visual review.
Library    Browser
Library    DocTest.WebVisualTest
...    baseline_directory=${CURDIR}${/}visual_baselines
...    dom_analysis=True

*** Variables ***
${PAGES}    ${CURDIR}${/}pages

*** Keywords ***
Open Example Page
    [Arguments]    ${page}    ${width}=1000
    New Browser    chromium    headless=True
    New Context    viewport={'width': ${width}, 'height': 700}
    New Page    file://${PAGES}${/}${page}

*** Test Cases ***
Dashboard Ignoring Dynamic Widgets
    [Documentation]    The clock and ad banner change constantly — mask them by
    ...                locator instead of maintaining pixel coordinates.
    Open Example Page    dashboard.html
    Compare Page To Baseline    dashboard    ignore_elements=id=clock;id=ad

Header Element Only
    [Documentation]    Element baselines isolate a component from page noise.
    Open Example Page    dashboard.html
    Compare Element To Baseline    css=header    dashboard-header    ignore_elements=id=clock

Checkout Form
    [Documentation]    Form values are part of the semantic DOM snapshot —
    ...                a filled field is a real change, not rendering noise.
    Open Example Page    form.html    width=700
    Compare Page To Baseline    checkout-form

Long Article Cross Render Tolerant
    [Documentation]    Typography-heavy full page; tolerate anti-aliasing noise
    ...                (e.g. baselines created on another machine or browser).
    Open Example Page    article.html    width=800
    Compare Page To Baseline    article    ignore_antialiasing=True    max_diff_pixels=100

Charts Need Pixel Vigilance
    [Documentation]    Canvas/SVG content is invisible to DOM analysis — never
    ...                combine accept_rendering_only with canvas-rendered UIs.
    Open Example Page    chart.html    width=800
    Compare Page To Baseline    charts
