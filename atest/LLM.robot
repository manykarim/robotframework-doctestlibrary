*** Settings ***
Library    OperatingSystem
Library    DocTest.VisualTest
Library    DocTest.PdfTest

Suite Setup    Require LLM Credentials


*** Keywords ***
Require LLM Credentials
    ${openai}=    Get Environment Variable    OPENAI_API_KEY    ${EMPTY}
    ${azure}=    Get Environment Variable    AZURE_OPENAI_API_KEY    ${EMPTY}
    IF    '${openai}' == '' and '${azure}' == ''
        Skip    LLM credentials not configured. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY to run AI tests.
    END


*** Test Cases ***
Compare Documents with minor difference and accept using LLM Keyword
    Compare Images With LLM    ${CURDIR}/../testdata/invoice.pdf    ${CURDIR}/../testdata/invoice_diff_font.pdf
    ...    llm_override=${True}    llm_prompt=Always approve identical images, also if font is slightly different

Compare Documents with minor difference and fail using LLM Keyword
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images With LLM    ${CURDIR}/../testdata/invoice.pdf    ${CURDIR}/../testdata/invoice_diff_font.pdf
    ...    llm_override=${True}

Compare Documents with more differences and fail using LLM Keyword
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images With LLM    ${CURDIR}/../testdata/invoice.pdf    ${CURDIR}/../testdata/invoice_moved_and_different.pdf
    ...    llm_override=${True}

Compare identical Pdf Documents Using LLM Keyword
    Compare Pdf Documents With LLM    ${CURDIR}/../testdata/invoice.pdf
    ...    ${CURDIR}/../testdata/invoice.pdf    llm_override=${False}
    ...    llm_prompt=Approval expected for identical documents
