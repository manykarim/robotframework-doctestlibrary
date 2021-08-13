*** Settings ***
Library    DocTest.PdfTest

*** Test Cases ***
This text exists in the PDF File
    @{strings}=    Create List    THE TEST SHIPPER    TEST CONSIGNEE
    PDF Should Contain Strings    ${strings}    testdata/sample.pdf

This text does NOT exist in the PDF File
    @{strings}=    Create List    THE TEST SHIPPER    THIS STRING DOES NOT EXIST IN THIS DOJO
    PDF Should Contain Strings    ${strings}    testdata/sample.pdf

Compare two equal PDF Files without signature
    Compare Pdf Documents    testdata/sample.pdf    testdata/sample.pdf

Compare two equal PDF Files with signature
    Compare Pdf Documents    testdata/sample06.pdf   testdata/sample06.pdf

Compare two different PDF Files with moved textblock
    Compare Pdf Documents    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf

Compare two different PDF Files
    Compare Pdf Documents    testdata/sample_1_page.pdf    testdata/sample_1_page_changed.pdf