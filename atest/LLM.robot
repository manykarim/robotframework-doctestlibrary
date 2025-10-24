*** Settings ***
Library    OperatingSystem
Library    DocTest.VisualTest
Library    DocTest.PdfTest
Library    DocTest.ai

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

Compare Documents with more differences and fail using LLM Keyword
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images With LLM    ${CURDIR}/../testdata/invoice.pdf    ${CURDIR}/../testdata/invoice_moved_and_different.pdf
    ...    llm_override=${True}

Compare identical Pdf Documents Using LLM Keyword
    Compare Pdf Documents With LLM    ${CURDIR}/../testdata/invoice.pdf
    ...    ${CURDIR}/../testdata/invoice.pdf    llm_override=${False}
    ...    llm_prompt=Approval expected for identical documents

Get Text With LLM Keyword
    ${text}=    Get Text With LLM    ${CURDIR}/../testdata/invoice.pdf    max_pages=1
    Should Not Be Empty    ${text}

Get Text From Area With LLM Keyword
    &{area}=    Create Dictionary    x=0    y=0    width=400    height=200    page=1
    ${snippet}=    Get Text From Area With LLM    ${CURDIR}/../testdata/invoice.pdf    ${area}
    Should Not Be Empty    ${snippet}

Chat With Document Keyword
    ${answer}=    Chat With Document    prompt=Summarise the document total.    documents=${CURDIR}/../testdata/invoice.pdf
    Should Not Be Empty    ${answer}

Get Item Count From Image Keyword
    ${count}=    Get Item Count From Image    ${CURDIR}/../testdata/invoice.pdf    item_description=distinct table sections
    Should Be True    ${count} >= 0

Image Should Contain Negative Keyword
    Run Keyword And Expect Error    *Expected object 'Non existing watermark' not found*
    ...    Image Should Contain    ${CURDIR}/../testdata/invoice.pdf    Non existing watermark

Get Item Count From Image Negative Keyword
    ${count}=    Get Item Count From Image    ${CURDIR}/../testdata/invoice.pdf    item_description=unicorn statues
    Should Be Equal As Integers    ${count}    0
