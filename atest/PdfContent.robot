*** Settings ***
Library    DocTest.PdfTest

*** Test Cases ***
This text list exists in the PDF File
    @{strings}=    Create List    THE TEST SHIPPER    TEST CONSIGNEE
    PDF Should Contain Strings    ${strings}    testdata/sample.pdf

Some non-expected text items from list exist in the PDF File
    @{strings}=    Create List    THE TEST SHIPPER    THIS STRING DOES NOT EXIST IN THIS DOJO 
    Run Keyword And Expect Error    Some non-expected texts were found in document    PDF Should Not Contain Strings    ${strings}    testdata/sample.pdf

No non-expected text items from list exist in the PDF File
    @{strings}=    Create List    THIS STRING DOES NOT EXIST IN THIS DOJO    THIS ONE AS WELL 
    PDF Should Not Contain Strings    ${strings}    testdata/sample.pdf

No non-expected text items from list exist in the PDF File from URL
    @{strings}=    Create List    THIS STRING DOES NOT EXIST IN THIS DOJO    THIS ONE AS WELL 
    PDF Should Not Contain Strings    ${strings}    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample.pdf

This text list does NOT exist in the PDF File
    @{strings}=    Create List    THE TEST SHIPPER    THIS STRING DOES NOT EXIST IN THIS DOJO
    Run Keyword And Expect Error    Some expected texts were not found in document    PDF Should Contain Strings    ${strings}    testdata/sample.pdf

This single string exists in the PDF File
    PDF Should Contain Strings    THE TEST SHIPPER    testdata/sample.pdf

This single string does NOT exist in the PDF File
    Run Keyword And Expect Error    Some expected texts were not found in document    PDF Should Contain Strings    THIS STRING DOES NOT EXIST IN THIS DOJO    testdata/sample.pdf

This single string exists in the PDF File from URL
    PDF Should Contain Strings    THE TEST SHIPPER    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample.pdf

This single string does NOT exist in the PDF File from URL
    Run Keyword And Expect Error    Some expected texts were not found in document    PDF Should Contain Strings    THIS STRING DOES NOT EXIST IN THIS DOJO    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample.pdf

Compare two equal PDF Files without signature
    Compare Pdf Documents    testdata/sample.pdf    testdata/sample.pdf

Compare two equal PDF Files from URL
    Compare Pdf Documents    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample.pdf    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample.pdf

Compare two equal PDF Files with signature
    Compare Pdf Documents    testdata/sample06.pdf   testdata/sample06.pdf

Compare two different PDF Files with moved textblock and only check text content
    Compare Pdf Documents    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf    compare=text

Compare two different PDF Files
    Run Keyword And Expect Error    The compared PDF Document Data is different.    Compare Pdf Documents    testdata/sample_1_page.pdf    testdata/sample_1_page_changed.pdf

Compare two different PDF Files from URL
    Run Keyword And Expect Error    The compared PDF Document Data is different.    Compare Pdf Documents    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample_1_page.pdf    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/sample_1_page_changed.pdf

Compare PDF structure while ignoring fonts
    Compare Pdf Structure    testdata/invoice.pdf    testdata/invoice_diff_font.pdf

Compare PDF structure detects content change
    Run Keyword And Expect Error    The compared PDF structure is different.    Compare Pdf Structure    testdata/invoice.pdf    testdata/invoice_diff_date_id.pdf

Compare Pdf Documents with structure mode
    Compare Pdf Documents    testdata/invoice.pdf    testdata/invoice_diff_font.pdf    compare=structure

Compare Pdf Documents with structure mode strict tolerance fails
    Run Keyword And Expect Error    The compared PDF Document Data is different.    Compare Pdf Documents    testdata/invoice.pdf    testdata/invoice_diff_font.pdf    compare=structure    structure_position_tolerance=0.0    structure_size_tolerance=0.0    structure_relative_tolerance=0.0
