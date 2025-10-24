*** Settings ***
Library    OperatingSystem
Library    DocTest.VisualTest
Library    DocTest.PdfTest

Suite Setup    Require LLM Credentials

*** Variables ***
${openai}    %{OPENAI_API_KEY=""}
${azure}    %{AZURE_OPENAI_API_KEY=""}

*** Keywords ***
Require LLM Credentials
    IF    '${openai}' == '' and '${azure}' == ''
        Skip    LLM credentials not configured. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY to run AI tests.
    END


*** Test Cases ***
Compare Documents with minor difference and accept using LLM Keyword with custom prompt allowing minor differences
    Compare Images With LLM    ${CURDIR}/../testdata/invoice.pdf    ${CURDIR}/../testdata/invoice_diff_font.pdf
    ...    llm_override=${True}    llm_prompt=Approve identical images if the content is the same, also if font type is slightly different

Compare Documents with minor difference and fail using LLM Keyword with custom prompt forbidding minor differences
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images With LLM    ${CURDIR}/../testdata/invoice.pdf    ${CURDIR}/../testdata/invoice_diff_font.pdf
    ...    llm_override=${True}    llm_prompt=Minor font differences shall fail the comparison

Compare Documents with more differences and fail using LLM Keyword with custom prompt
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images With LLM    ${CURDIR}/../testdata/invoice.pdf    ${CURDIR}/../testdata/invoice_moved_and_different.pdf
    ...    llm_override=${True}    llm_prompt=Approve identical images if the content is the same, also if font type is slightly different


Compare Documents with more differences and fail using LLM Keyword
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images With LLM    ${CURDIR}/../testdata/invoice.pdf    ${CURDIR}/../testdata/invoice_moved_and_different.pdf
    ...    llm_override=${True}

Compare identical Pdf Documents Using LLM Keyword
    Compare Pdf Documents With LLM    ${CURDIR}/../testdata/invoice.pdf
    ...    ${CURDIR}/../testdata/invoice.pdf    llm_override=${False}
    ...    llm_prompt=Approval expected for identical documents
