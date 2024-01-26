*** Settings ***
Library    DocTest.VisualTest    take_screenshots=True        DPI=72
Library    DocTest.PrintJobTests


*** Variables ***
${TESTDATA_DIR}    testdata
${MASK_DIR}    testdata

*** Test Cases ***
Visually Compare Two PCL Invoices with missing Logo
    Run Keyword And Expect Error    The compared images are different.    Compare Images    ${TESTDATA_DIR}/invoice.pcl    ${TESTDATA_DIR}/invoice_no_logo.pcl

Visually Compare Two PCL Invoices with missing Logo and mask
    Compare Images    ${TESTDATA_DIR}/invoice.pcl    ${TESTDATA_DIR}/invoice_no_logo.pcl    placeholder_file=${MASK_DIR}/mask_logo.json

Visually Compare Two PS Invoices with missing Logo
    Run Keyword And Expect Error    The compared images are different.    Compare Images    ${TESTDATA_DIR}/invoice.ps    ${TESTDATA_DIR}/invoice_no_logo.ps

Visually Compare Two PS Invoices with missing Logo and mask
    Compare Images    ${TESTDATA_DIR}/invoice.ps    ${TESTDATA_DIR}/invoice_no_logo.ps    placeholder_file=${MASK_DIR}/mask_logo.json

Compare content of two PCL Print Jobs with same metadata
    Compare Print Jobs    pcl    ${TESTDATA_DIR}/invoice.pcl    ${TESTDATA_DIR}/invoice_no_logo.pcl

Compare content of two PCL Print Jobs from URL with same metadata
    Compare Print Jobs    pcl    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/invoice.pcl    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/invoice_no_logo.pcl

Compare content of two PCL Print Jobs with diff metadata
    Run Keyword And Expect Error    The compared print jobs are different.    Compare Print Jobs    pcl    ${TESTDATA_DIR}/invoice.pcl    ${TESTDATA_DIR}/invoice_letterformat.pcl

Compare content of two PCL Print Jobs from URL with diff metadata
    Run Keyword And Expect Error    The compared print jobs are different.    Compare Print Jobs    pcl    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/invoice.pcl    https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/invoice_letterformat.pcl


Visually Compare Two PCL Invoices with same visuals but different metadata
    Compare Images    ${TESTDATA_DIR}/invoice.pcl    ${TESTDATA_DIR}/invoice_difftray.pcl

Compare content of two PCL Print Jobs with same visuals but different metadata
    Run Keyword And Expect Error    The compared print jobs are different.    Compare Print Jobs    pcl    ${TESTDATA_DIR}/invoice.pcl    ${TESTDATA_DIR}/invoice_difftray.pcl

Check matching property of Print Job
    ${print_job}    Get Pcl Print Job    ${TESTDATA_DIR}/invoice.pcl
    ${page_orientation}    Create Dictionary    page=1    property=page_orientation    value=0
    ${paper_format}    Create Dictionary    page=1    property=paper_format    value=26
    ${copies}    Create Dictionary    page=1    property=copies    value=1
    
    Check Print Job Property    ${print_job}    pcl_commands    ${page_orientation}
    Check Print Job Property    ${print_job}    pcl_commands    ${paper_format}
    Check Print Job Property    ${print_job}    pcl_commands    ${copies}

Check non-matching property of Print Job
    ${print_job}    Get Pcl Print Job    ${TESTDATA_DIR}/invoice.pcl
    ${page_orientation}    Create Dictionary    page=1    property=page_orientation    value=1    
    Run Keyword And Expect Error    The print job property check failed.    Check Print Job Property    ${print_job}    pcl_commands    ${page_orientation}

Check non-existing property of Print Job
    ${print_job}    Get Pcl Print Job    ${TESTDATA_DIR}/invoice.pcl
    ${not_existing}    Create Dictionary    page=1    property=not_existing    value=1    
    Run Keyword And Expect Error    The print job property check failed.    Check Print Job Property    ${print_job}    pcl_commands    ${not_existing}

Check non-existing property group of Print Job
    ${print_job}    Get Pcl Print Job    ${TESTDATA_DIR}/invoice.pcl
    ${page_orientation}    Create Dictionary    page=1    property=page_orientation    value=0    
    Run Keyword And Expect Error    *The property does not exist*    Check Print Job Property    ${print_job}    not_existing    ${page_orientation}
