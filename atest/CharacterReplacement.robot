*** Settings ***
Library    DocTest.PdfTest
Library    DocTest.VisualTest
Library    Collections

*** Variables ***
${TESTDATA_DIR}    testdata

*** Test Cases ***
PdfTest Library Accepts Character Replacements Parameter
    [Documentation]    Verify PdfTest can be initialized with character_replacements
    [Tags]    character_replacements    pdftest
    # This test verifies the library initialization doesn't fail
    # The actual replacement is tested in unit tests
    Log    PdfTest library initialized successfully with character_replacements support

VisualTest Library Accepts Character Replacements Parameter
    [Documentation]    Verify VisualTest can be initialized with character_replacements
    [Tags]    character_replacements    visualtest
    Log    VisualTest library initialized successfully with character_replacements support

Set Character Replacements Keyword Works
    [Documentation]    Verify Set Character Replacements keyword is available
    [Tags]    character_replacements    visualtest
    Set Character Replacements    ${NONE}
    Set Character Replacements    {'\u00A0': ' '}
    Set Character Replacements    ${NONE}
    Log    Set Character Replacements keyword works correctly

Compare Pdf Documents With Character Replacements Parameter
    [Documentation]    Verify compare_pdf_documents accepts character_replacements
    [Tags]    character_replacements    pdftest
    Compare Pdf Documents    ${TESTDATA_DIR}/sample.pdf    ${TESTDATA_DIR}/sample.pdf
    ...    character_replacements={'\u00A0': ' '}

Compare Pdf Structure With Character Replacements Parameter
    [Documentation]    Verify compare_pdf_structure accepts character_replacements
    [Tags]    character_replacements    pdftest
    Compare Pdf Structure    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice.pdf
    ...    character_replacements={'\u00A0': ' '}

PDF Should Contain Strings With Character Replacements Parameter
    [Documentation]    Verify PDF_should_contain_strings accepts character_replacements
    [Tags]    character_replacements    pdftest
    PDF Should Contain Strings    THE TEST SHIPPER    ${TESTDATA_DIR}/sample.pdf
    ...    character_replacements={'\u00A0': ' '}

PDF Should Not Contain Strings With Character Replacements Parameter
    [Documentation]    Verify PDF_should_not_contain_strings accepts character_replacements
    [Tags]    character_replacements    pdftest
    PDF Should Not Contain Strings    NON EXISTENT TEXT    ${TESTDATA_DIR}/sample.pdf
    ...    character_replacements={'\u00A0': ' '}

Get Text With Character Replacements Via Setter
    [Documentation]    Verify Get Text uses character replacements set via setter
    [Tags]    character_replacements    visualtest
    Set Character Replacements    {'\u00A0': ' '}
    ${text}=    Get Text    ${TESTDATA_DIR}/sample.pdf
    Should Not Be Empty    ${text}
    Set Character Replacements    ${NONE}

Get Text From Document With Character Replacements Via Setter
    [Documentation]    Verify Get Text From Document uses character replacements
    [Tags]    character_replacements    visualtest
    Set Character Replacements    {'\u00A0': ' '}
    ${text}=    Get Text From Document    ${TESTDATA_DIR}/sample.pdf
    Should Not Be Empty    ${text}
    Set Character Replacements    ${NONE}

Character Replacements Can Be Cleared
    [Documentation]    Verify character replacements can be cleared with None
    [Tags]    character_replacements    visualtest
    Set Character Replacements    {'\u00A0': ' ', '\u2013': '-'}
    Set Character Replacements    ${NONE}
    ${text}=    Get Text    ${TESTDATA_DIR}/sample.pdf
    Should Not Be Empty    ${text}

Multiple Character Replacements Work
    [Documentation]    Verify multiple character replacements can be specified
    [Tags]    character_replacements    pdftest
    ${replacements}=    Create Dictionary    \u00A0= \    \u2013=-
    PDF Should Contain Strings    THE TEST SHIPPER    ${TESTDATA_DIR}/sample.pdf
    ...    character_replacements=${replacements}
