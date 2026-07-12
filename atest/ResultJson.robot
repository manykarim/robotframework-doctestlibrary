*** Settings ***
Library    DocTest.VisualTest    result_json=true    take_screenshots=false
Library    DocTest.PdfTest    result_json=true
Library    OperatingSystem
Library    Collections
Library    String

*** Variables ***
${TESTDATA_DIR}    testdata

*** Test Cases ***
Failing Image Comparison Writes Sidecar Into Output Dir
    ${before}=    Count Sidecars
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images    ${TESTDATA_DIR}${/}Beach_left.jpg    ${TESTDATA_DIR}${/}Beach_right.jpg
    ${after}=    Count Sidecars
    Should Be Equal As Integers    ${after}    ${before + 1}
    ${sidecar}=    Get Latest Sidecar
    ${content}=    Evaluate    json.load(open(r'''${sidecar}'''))    modules=json
    Should Be Equal    ${content}[status]    FAIL
    Should Be Equal As Integers    ${content}[schema_version]    1
    ${pages}=    Get From Dictionary    ${content}    pages
    Should Not Be Empty    ${pages}

Passing Image Comparison Writes Sidecar
    ${before}=    Count Sidecars
    Compare Images    ${TESTDATA_DIR}${/}Beach_left.jpg    ${TESTDATA_DIR}${/}Beach_left.jpg
    ${after}=    Count Sidecars
    Should Be Equal As Integers    ${after}    ${before + 1}
    ${sidecar}=    Get Latest Sidecar
    ${content}=    Evaluate    json.load(open(r'''${sidecar}'''))    modules=json
    Should Be Equal    ${content}[status]    PASS

Pdf Comparison Writes Document Level Sidecar
    ${before}=    Count Sidecars
    Compare Pdf Documents    ${TESTDATA_DIR}${/}sample_1_page.pdf    ${TESTDATA_DIR}${/}sample_1_page.pdf
    ${after}=    Count Sidecars
    Should Be Equal As Integers    ${after}    ${before + 1}
    ${sidecar}=    Get Latest Sidecar
    ${content}=    Evaluate    json.load(open(r'''${sidecar}'''))    modules=json
    Should Be Equal    ${content}[status]    PASS
    Should Be Equal    ${content}[library]    DocTest.PdfTest

Sidecar Can Be Disabled With Keyword
    ${before}=    Count Sidecars
    DocTest.VisualTest.Set Result Json    ${False}
    Compare Images    ${TESTDATA_DIR}${/}Beach_left.jpg    ${TESTDATA_DIR}${/}Beach_left.jpg
    ${after}=    Count Sidecars
    Should Be Equal As Integers    ${after}    ${before}
    [Teardown]    DocTest.VisualTest.Set Result Json    ${True}

*** Keywords ***
Count Sidecars
    # run.json is the run manifest (sidecar v1.1), not a comparison sidecar
    ${exists}=    Run Keyword And Return Status    Directory Should Exist    ${OUTPUT_DIR}${/}doctest_results
    IF    not ${exists}    RETURN    ${0}
    ${files}=    List Files In Directory    ${OUTPUT_DIR}${/}doctest_results    *.json
    ${sidecars}=    Evaluate    [f for f in $files if f != 'run.json']
    ${count}=    Get Length    ${sidecars}
    RETURN    ${count}

Get Latest Sidecar
    ${files}=    List Files In Directory    ${OUTPUT_DIR}${/}doctest_results    *.json    absolute=${True}
    ${sidecars}=    Evaluate    [f for f in $files if not f.endswith('run.json')]
    ${sorted}=    Evaluate    sorted(${sidecars}, key=lambda p: __import__('os').path.getmtime(p))
    RETURN    ${sorted}[-1]
