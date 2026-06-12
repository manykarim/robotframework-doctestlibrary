*** Settings ***
Library    DocTest.VisualTest    take_screenshots=false
Library    DocTest.PdfTest
Library    OperatingSystem

*** Variables ***
${TESTDATA_DIR}    testdata
${WORK_DIR}    ${TEMPDIR}${/}doctest_reference_run

*** Test Cases ***
Missing Image Reference Is Created In Reference Run
    [Setup]    Prepare Work Dir
    DocTest.VisualTest.Set Reference Run    ${True}
    Compare Images    ${WORK_DIR}${/}missing_reference.png    ${TESTDATA_DIR}${/}Beach_left.jpg
    File Should Exist    ${WORK_DIR}${/}missing_reference.png
    DocTest.VisualTest.Set Reference Run    ${False}
    Compare Images    ${WORK_DIR}${/}missing_reference.png    ${TESTDATA_DIR}${/}Beach_left.jpg
    [Teardown]    Remove Directory    ${WORK_DIR}    recursive=${True}

Differing Image Candidate Replaces Reference In Reference Run
    [Setup]    Prepare Work Dir
    Copy File    ${TESTDATA_DIR}${/}Beach_left.jpg    ${WORK_DIR}${/}reference.jpg
    DocTest.VisualTest.Set Reference Run    ${True}
    Compare Images    ${WORK_DIR}${/}reference.jpg    ${TESTDATA_DIR}${/}Beach_right.jpg
    DocTest.VisualTest.Set Reference Run    ${False}
    Compare Images    ${WORK_DIR}${/}reference.jpg    ${TESTDATA_DIR}${/}Beach_right.jpg
    [Teardown]    Remove Directory    ${WORK_DIR}    recursive=${True}

Reference Run Disabled Still Fails On Differences
    [Setup]    Prepare Work Dir
    Copy File    ${TESTDATA_DIR}${/}Beach_left.jpg    ${WORK_DIR}${/}reference.jpg
    DocTest.VisualTest.Set Reference Run    ${False}
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images    ${WORK_DIR}${/}reference.jpg    ${TESTDATA_DIR}${/}Beach_right.jpg
    [Teardown]    Remove Directory    ${WORK_DIR}    recursive=${True}

Differing Pdf Candidate Replaces Reference In Reference Run
    [Setup]    Prepare Work Dir
    Copy File    ${TESTDATA_DIR}${/}sample_1_page.pdf    ${WORK_DIR}${/}reference.pdf
    DocTest.PdfTest.Set Reference Run    ${True}
    Compare Pdf Documents    ${WORK_DIR}${/}reference.pdf    ${TESTDATA_DIR}${/}sample_1_page_moved.pdf
    DocTest.PdfTest.Set Reference Run    ${False}
    Compare Pdf Documents    ${WORK_DIR}${/}reference.pdf    ${TESTDATA_DIR}${/}sample_1_page_moved.pdf
    [Teardown]    Remove Directory    ${WORK_DIR}    recursive=${True}

*** Keywords ***
Prepare Work Dir
    Create Directory    ${WORK_DIR}
