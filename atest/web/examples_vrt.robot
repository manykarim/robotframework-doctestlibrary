*** Settings ***
Documentation    End-to-end scenarios over the example gallery: every
...              reliability feature exercised against realistic pages,
...              including the documented edge cases.
Library    Browser
Library    DocTest.WebVisualTest    baseline_directory=${OUTPUT_DIR}${/}exbaselines    retry_timeout=1s
Library    OperatingSystem

*** Variables ***
${PAGES}    ${EXECDIR}${/}examples${/}web-visual${/}pages

*** Keywords ***
Open Example
    [Arguments]    ${page}    ${width}=1000    ${height}=700
    New Browser    chromium    headless=True
    New Context    viewport={'width': ${width}, 'height': ${height}}
    New Page    file://${PAGES}${/}${page}

*** Test Cases ***
Dashboard With Dynamic Content Masked By Locator
    [Documentation]    Clock and ad banner change every run — ignore them by
    ...                locator; a real metric change must still fail.
    Open Example    dashboard.html
    Compare Page To Baseline    ex-dashboard
    Evaluate JavaScript    ${None}    () => { window.demo.tickClock(); window.demo.rotateAd(); }
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Page To Baseline    ex-dashboard
    Compare Page To Baseline    ex-dashboard    ignore_elements=id=clock;id=ad
    Evaluate JavaScript    ${None}    () => window.demo.changeRevenue('$999,999')
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Page To Baseline    ex-dashboard    ignore_elements=id=clock;id=ad

Restyle Accepted As Rendering Only
    [Documentation]    A pure CSS restyle fails pixels but keeps the DOM intact —
    ...                accept_rendering_only passes it while recording for review.
    Open Example    dashboard.html
    Compare Page To Baseline    ex-dashboard-style    dom_analysis=True
    Evaluate JavaScript    ${None}    () => window.demo.restyleHeader('#7c2d12')
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Page To Baseline    ex-dashboard-style    dom_analysis=True
    Compare Page To Baseline    ex-dashboard-style    dom_analysis=True    accept_rendering_only=True

Form Value Change Is Semantic
    [Documentation]    Typing into an input is a DOM value change — never
    ...                auto-accepted even though it looks minor.
    Open Example    form.html    width=700    height=600
    Compare Page To Baseline    ex-form    dom_analysis=True
    Evaluate JavaScript    ${None}    () => window.demo.fillName('Ada Lovelace')
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Page To Baseline    ex-form    dom_analysis=True    accept_rendering_only=True

Validation State Covered By Element Baseline
    [Documentation]    The error box gets its own element baseline; showing it
    ...                changes both pixels and DOM.
    Open Example    form.html    width=700    height=600
    Evaluate JavaScript    ${None}    () => window.demo.showError()
    Compare Element To Baseline    id=error    ex-form-error
    Compare Element To Baseline    id=error    ex-form-error

Canvas Change Is A DOM Blind Spot But Pixels Catch It
    [Documentation]    Canvas content is invisible to DOM analysis (verdict
    ...                stays identical) — pixel comparison remains the guard.
    Open Example    chart.html    width=800    height=500
    Compare Page To Baseline    ex-chart    dom_analysis=True
    Evaluate JavaScript    ${None}    () => window.demo.redrawCanvas(180)
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Page To Baseline    ex-chart    dom_analysis=True

Long Article Full Page With Antialiasing Tolerance
    [Documentation]    Full-page capture of a long typographic page; AA
    ...                tolerance + pixel budget roundtrip.
    Open Example    article.html    width=800    height=600
    Compare Page To Baseline    ex-article
    Compare Page To Baseline    ex-article    ignore_antialiasing=True    max_diff_pixels=50

Responsive Grid Gets Per Viewport Baselines
    [Documentation]    The same page reflows per viewport — split_baselines_by
    ...                keeps one baseline per configuration.
    Import Library    DocTest.WebVisualTest    baseline_directory=${OUTPUT_DIR}${/}exbaselines    split_baselines_by=viewport    AS    SplitVisual
    Open Example    products.html    width=1000    height=700
    SplitVisual.Compare Page To Baseline    ex-products
    Close Browser
    Open Example    products.html    width=500    height=700
    SplitVisual.Compare Page To Baseline    ex-products
    File Should Exist    ${OUTPUT_DIR}${/}exbaselines${/}1000x700${/}ex-products.png
    File Should Exist    ${OUTPUT_DIR}${/}exbaselines${/}500x700${/}ex-products.png
