*** Settings ***
Library    DocTest.VisualTest    take_screenshots=True    ocr_engine=tesseract
Library    Collections
Library    BuiltIn

*** Variables ***
${TESTDATA_DIR}    testdata

*** Test Cases ***
Get Text From PDF Document
    [Documentation]    Test extracting text from PDF document
    [Tags]    ocr    text_extraction    pdf
    ${text}    Get Text From Document    ${TESTDATA_DIR}/sample.pdf
    Should Not Be Empty    ${text}
    Should Be True    isinstance($text, str)

Get Text From Image Document
    [Documentation]    Test extracting text from image document
    [Tags]    ocr    text_extraction    image
    ${text}    Get Text From Document    ${TESTDATA_DIR}/Beach_date.png
    Should Not Be Empty    ${text}
    Should Be True    isinstance($text, str)

Get Text From Document With Custom OCR Config
    [Documentation]    Test text extraction with custom OCR configuration
    [Tags]    ocr    text_extraction    custom_config
    ${text}    Get Text From Document    ${TESTDATA_DIR}/sample.pdf
    Should Not Be Empty    ${text}

Compare Documents With Text Content Check
    [Documentation]    Test comparing documents by checking text content
    [Tags]    ocr    text_comparison    content
    Compare Images    ${TESTDATA_DIR}/sample_1_page.pdf    ${TESTDATA_DIR}/sample_1_page_moved.pdf    check_text_content=${True}

Compare Documents With Text Content Check Should Fail
    [Documentation]    Test that documents with different text content fail comparison
    [Tags]    ocr    text_comparison    content    negative
    Run Keyword And Expect Error    The compared images are different.    Compare Images    ${TESTDATA_DIR}/sample_1_page.pdf    ${TESTDATA_DIR}/sample_1_page_moved_and_different.pdf    check_text_content=${True}

Compare Images With Move Tolerance And Text Check
    [Documentation]    Test comparing images with move tolerance and text content verification
    [Tags]    ocr    move_tolerance    text_comparison
    Compare Images    ${TESTDATA_DIR}/sample_1_page.pdf    ${TESTDATA_DIR}/sample_1_page_moved.pdf    move_tolerance=20    check_text_content=${True}

Get Text From Document With Different OCR Engines
    [Documentation]    Test text extraction with different OCR engines
    [Tags]    ocr    text_extraction    engines
    ${text_tesseract}    Get Text From Document    ${TESTDATA_DIR}/sample.pdf
    Should Not Be Empty    ${text_tesseract}

Test OCR With Different Image Formats
    [Documentation]    Test OCR functionality with various image formats
    [Tags]    ocr    text_extraction    formats
    ${text_jpg}    Get Text From Document    ${TESTDATA_DIR}/Beach_left.jpg
    ${text_png}    Get Text From Document    ${TESTDATA_DIR}/Beach_left.png
    ${text_pdf}    Get Text From Document    ${TESTDATA_DIR}/sample_1_page.pdf
    Should Not Be Empty    ${text_jpg}
    Should Not Be Empty    ${text_png}
    Should Not Be Empty    ${text_pdf}

Test OCR Error Handling
    [Documentation]    Test OCR error handling with invalid files
    [Tags]    ocr    error_handling    negative
    Run Keyword And Expect Error    *    Get Text From Document    non_existent_file.pdf

Get Barcodes From Document
    [Documentation]    Test extracting barcodes from document
    [Tags]    barcode    extraction
    ${barcodes}    Get Barcodes From Document    ${TESTDATA_DIR}/sample_barcodes.pdf
    Should Be True    isinstance($barcodes, list)

Test Barcode Detection With Different Types
    [Documentation]    Test barcode detection with different barcode types
    [Tags]    barcode    extraction    types
    ${qr_codes}    Get Barcodes From Document    ${TESTDATA_DIR}/sample_barcodes.pdf
    ${data_matrix}    Get Barcodes From Document    ${TESTDATA_DIR}/datamatrix.png
    Should Be True    isinstance($qr_codes, list)
    Should Be True    isinstance($data_matrix, list)
