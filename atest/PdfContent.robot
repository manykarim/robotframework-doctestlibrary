*** Settings ***
Library    DocTest.PdfTest

*** Variables ***
${TESTDATA_DIR}    testdata

*** Test Cases ***
This text list exists in the PDF File
    @{strings}=    Create List    THE TEST SHIPPER    TEST CONSIGNEE
    PDF Should Contain Strings    ${strings}    ${TESTDATA_DIR}/sample.pdf

Some non-expected text items from list exist in the PDF File
    @{strings}=    Create List    THE TEST SHIPPER    THIS STRING DOES NOT EXIST IN THIS DOJO 
    Run Keyword And Expect Error    Some non-expected texts were found in document    PDF Should Not Contain Strings    ${strings}    ${TESTDATA_DIR}/sample.pdf

No non-expected text items from list exist in the PDF File
    @{strings}=    Create List    THIS STRING DOES NOT EXIST IN THIS DOJO    THIS ONE AS WELL 
    PDF Should Not Contain Strings    ${strings}    ${TESTDATA_DIR}/sample.pdf

No non-expected text items from list exist in the PDF File from URL
    @{strings}=    Create List    THIS STRING DOES NOT EXIST IN THIS DOJO    THIS ONE AS WELL 
    PDF Should Not Contain Strings    ${strings}    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample.pdf

This text list does NOT exist in the PDF File
    @{strings}=    Create List    THE TEST SHIPPER    THIS STRING DOES NOT EXIST IN THIS DOJO
    Run Keyword And Expect Error    Some expected texts were not found in document    PDF Should Contain Strings    ${strings}    ${TESTDATA_DIR}/sample.pdf

This single string exists in the PDF File
    PDF Should Contain Strings    THE TEST SHIPPER    ${TESTDATA_DIR}/sample.pdf

This single string does NOT exist in the PDF File
    Run Keyword And Expect Error    Some expected texts were not found in document    PDF Should Contain Strings    THIS STRING DOES NOT EXIST IN THIS DOJO    ${TESTDATA_DIR}/sample.pdf

This single string exists in the PDF File from URL
    PDF Should Contain Strings    THE TEST SHIPPER    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample.pdf

This single string does NOT exist in the PDF File from URL
    Run Keyword And Expect Error    Some expected texts were not found in document    PDF Should Contain Strings    THIS STRING DOES NOT EXIST IN THIS DOJO    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample.pdf

Compare two equal PDF Files without signature
    Compare Pdf Documents    ${TESTDATA_DIR}/sample.pdf    ${TESTDATA_DIR}/sample.pdf

Compare two equal PDF Files from URL
    Compare Pdf Documents    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample.pdf    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample.pdf

Compare two equal PDF Files with signature
    Compare Pdf Documents    ${TESTDATA_DIR}/sample06.pdf   ${TESTDATA_DIR}/sample06.pdf

Compare two different PDF Files with moved textblock and only check text content
    Compare Pdf Documents    ${TESTDATA_DIR}/sample_1_page.pdf    ${TESTDATA_DIR}/sample_1_page_moved.pdf    compare=text

Compare two different PDF Files
    Run Keyword And Expect Error    The compared PDF Document Data is different.    Compare Pdf Documents    ${TESTDATA_DIR}/sample_1_page.pdf    ${TESTDATA_DIR}/sample_1_page_changed.pdf

Compare two different PDF Files from URL
    Run Keyword And Expect Error    The compared PDF Document Data is different.    Compare Pdf Documents    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample_1_page.pdf    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample_1_page_changed.pdf

Compare PDF structure while ignoring fonts
    Compare Pdf Structure    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_font.pdf

Compare PDF structure detects content change
    Run Keyword And Expect Error    The compared PDF structure is different.    Compare Pdf Structure    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_date_id.pdf

Compare Pdf Documents with structure mode
    Compare Pdf Documents    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_font.pdf    compare=structure

Compare Pdf Documents with structure mode strict tolerance fails
    Run Keyword And Expect Error    The compared PDF Document Data is different.    Compare Pdf Documents    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_font.pdf    compare=structure    structure_position_tolerance=0.0    structure_size_tolerance=0.0    structure_relative_tolerance=0.0

Compare PDF Documents with structure mode and ignore differences in date and ID
    ${DATE_MASK}=    Create Dictionary    page=all    type=pattern    pattern=.*(?:January|February|March|April|May|June|July|August|September|October|November|December) \\d{1,2}, \\d{4}.*
    ${INV_MASK}=    Create Dictionary    page=all    type=pattern    pattern=INV-\\d{4}-\\d{6}
    ${JOB_MASK}=    Create Dictionary    page=all    type=pattern    pattern=.*JOB-\\d{4}-\\d{4}
    ${MASKS}=    Create List    ${DATE_MASK}    ${INV_MASK}    ${JOB_MASK}
    Compare Pdf Documents    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_date_id.pdf    compare=structure    mask=${MASKS}

Compare PDF structure and ignore differences in date and ID
    ${DATE_MASK}=    Create Dictionary    page=all    type=pattern    pattern=.*(?:January|February|March|April|May|June|July|August|September|October|November|December) \\d{1,2}, \\d{4}.*
    ${INV_MASK}=    Create Dictionary    page=all    type=pattern    pattern=INV-\\d{4}-\\d{6}
    ${JOB_MASK}=    Create Dictionary    page=all    type=pattern    pattern=.*JOB-\\d{4}-\\d{4}
    ${MASKS}=    Create List    ${DATE_MASK}    ${INV_MASK}    ${JOB_MASK}
    Compare Pdf Structure    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_date_id.pdf    mask=${MASKS}

Compare PDF Structure With Ignore Page Boundaries
    [Documentation]    Test that ignore_page_boundaries option compares text content across pages
    Compare Pdf Structure    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice.pdf
    ...    ignore_page_boundaries=${True}

Compare PDF Structure With Ignore Page Boundaries Passes With Font Difference
    [Documentation]    When ignoring page boundaries, font differences should not cause failures
    Compare Pdf Structure    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_font.pdf
    ...    ignore_page_boundaries=${True}

Compare PDF Structure With Check Geometry Disabled
    [Documentation]    When check_geometry is False, position differences are ignored
    Compare Pdf Structure    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_font.pdf
    ...    check_geometry=${False}

Compare PDF Structure With Check Block Count Disabled
    [Documentation]    When check_block_count is False, block count differences are ignored
    Compare Pdf Structure    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_font.pdf
    ...    check_block_count=${False}

Compare PDF Structure With Both Checks Disabled
    [Documentation]    Both geometry and block count checks can be disabled
    Compare Pdf Structure    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_font.pdf
    ...    check_geometry=${False}    check_block_count=${False}

Compare PDF Structure Still Detects Text Changes With Checks Disabled
    [Documentation]    Even with geometry/block checks disabled, text differences are detected
    Run Keyword And Expect Error    The compared PDF structure is different.
    ...    Compare Pdf Structure    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_date_id.pdf
    ...    check_geometry=${False}    check_block_count=${False}

Compare PDF Structure Still Detects Text Changes With Ignore Page Boundaries
    [Documentation]    Even with ignore_page_boundaries, text differences are detected
    Run Keyword And Expect Error    The compared PDF structure is different.
    ...    Compare Pdf Structure    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_date_id.pdf
    ...    ignore_page_boundaries=${True}

Compare Pdf Documents Structure Mode With Ignore Page Boundaries
    [Documentation]    The ignore_page_boundaries parameter works with compare_pdf_documents
    Compare Pdf Documents    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_font.pdf
    ...    compare=structure    ignore_page_boundaries=${True}

Compare Pdf Documents Structure Mode With Check Geometry Disabled
    [Documentation]    The check_geometry parameter works with compare_pdf_documents
    Compare Pdf Documents    ${TESTDATA_DIR}/invoice.pdf    ${TESTDATA_DIR}/invoice_diff_font.pdf
    ...    compare=structure    check_geometry=${False}