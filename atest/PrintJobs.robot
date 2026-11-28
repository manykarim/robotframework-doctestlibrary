*** Settings ***
Library    DocTest.VisualTest    take_screenshots=True        DPI=72
Library    DocTest.PrintJobTests


*** Variables ***
${TESTDATA_DIR}    testdata
${MASK_DIR}    testdata
${PJ_DIR}    ${CURDIR}/../utest/testdata/printjobs

*** Test Cases ***
Visually Compare Two PCL Invoices with missing Logo
    Run Keyword And Expect Error    The compared images are different.    Compare Images    ${TESTDATA_DIR}/invoice.pcl    ${TESTDATA_DIR}/invoice_no_logo.pcl

Visually Compare Two PCL Invoices with missing Logo and mask
    Compare Images    ${TESTDATA_DIR}/invoice.pcl    ${TESTDATA_DIR}/invoice_no_logo.pcl    placeholder_file=${MASK_DIR}/mask_logo.json    threshold=0.00004

Visually Compare Two PS Invoices with missing Logo
    Run Keyword And Expect Error    The compared images are different.    Compare Images    ${TESTDATA_DIR}/invoice.ps    ${TESTDATA_DIR}/invoice_no_logo.ps

Visually Compare Two PS Invoices with missing Logo and mask
    Compare Images    ${TESTDATA_DIR}/invoice.ps    ${TESTDATA_DIR}/invoice_no_logo.ps    placeholder_file=${MASK_DIR}/mask_logo.json    threshold=0.00004

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

Compare PCL XL Print Jobs with different paper size
    Run Keyword And Expect Error    The compared print jobs are different.    Compare Print Jobs    pcl    ${TESTDATA_DIR}/invoice_diff_format.pcl    ${PJ_DIR}/pclxl_letter.pcl

Compare PCL 5 with PCL XL rendition of the same document
    Run Keyword And Expect Error    The compared print jobs are different.    Compare Print Jobs    pcl    ${TESTDATA_DIR}/invoice.pcl    ${TESTDATA_DIR}/invoice_diff_format.pcl

Compare PCL Print Jobs with different page count
    Run Keyword And Expect Error    The compared print jobs are different.    Compare Print Jobs    pcl    ${PJ_DIR}/pcl5_one_page.pcl    ${PJ_DIR}/pcl5_two_pages.pcl

Compare PS Print Jobs with different page count
    Run Keyword And Expect Error    The compared print jobs are different.    Compare Print Jobs    ps    ${PJ_DIR}/ps_01_minimal_dsc.ps    ${PJ_DIR}/ps_09_two_pages.ps

Compare PS Print Jobs with A4 and Letter bounding box
    Run Keyword And Expect Error    The compared print jobs are different.    Compare Print Jobs    ps    ${PJ_DIR}/ps_a4.ps    ${PJ_DIR}/ps_letter.ps

Compare PS Print Jobs that differ only in run-specific values
    Compare Print Jobs    ps    ${PJ_DIR}/ps_run_a.ps    ${PJ_DIR}/ps_run_b.ps

Compare PS Print Jobs while ignoring additional properties
    Run Keyword And Expect Error    The compared print jobs are different.    Compare Print Jobs    ps    ${PJ_DIR}/ps_creator_a.ps    ${PJ_DIR}/ps_creator_b.ps
    Compare Print Jobs    ps    ${PJ_DIR}/ps_creator_a.ps    ${PJ_DIR}/ps_creator_b.ps    ignore_properties=Creator

Check page count of a PostScript Print Job without DSC page comments
    ${print_job}    Get Postscript Print Job    ${PJ_DIR}/ps_07_no_dsc_two_pages.ps
    Check Print Job Property    ${print_job}    page_count    2

Visually compare an empty PCL file fails with a rendering error
    Run Keyword And Expect Error    *produced no pages*    Compare Images    ${PJ_DIR}/empty.pcl    ${PJ_DIR}/empty.pcl

Get Print Job Ticket of a PCL 5 invoice
    ${ticket}    Get Print Job Ticket    ${TESTDATA_DIR}/invoice.pcl
    Should Be Equal    ${ticket}[pages][0][paper_size]    A4
    Should Be Equal    ${ticket}[pages][0][input_tray]    auto
    Should Be Equal As Integers    ${ticket}[page_count]    1

Compare Print Job Tickets reports a changed tray
    Run Keyword And Expect Error    *page 1: input_tray auto → pcl5:4*    Compare Print Job Tickets    ${TESTDATA_DIR}/invoice.pcl    ${TESTDATA_DIR}/invoice_difftray.pcl

Compare Print Job Tickets of PCL 5 and PCL XL renditions
    Compare Print Job Tickets    ${TESTDATA_DIR}/invoice.pcl    ${TESTDATA_DIR}/invoice_diff_format.pcl    ignore_properties=pjl,resolution,color_mode

Print Job Ticket Should Contain expected settings
    Print Job Ticket Should Contain    ${PJ_DIR}/ticket_duplex_once.pcl    duplex=long_edge    paper_size=A4
    Print Job Ticket Should Contain    ${PJ_DIR}/ticket_duplex_page1_only.pcl    page=1    duplex=long_edge

Print Job Ticket Should Contain fails on a different page
    Run Keyword And Expect Error    *page 2: duplex is simplex, expected long_edge*    Print Job Ticket Should Contain    ${PJ_DIR}/ticket_duplex_page1_only.pcl    duplex=long_edge

Compare Print Job Tickets with a tray mapping across languages
    ${trays}    Create Dictionary    pcl5:4=drawer2    ps:Tray2=drawer2
    ${mapping}    Create Dictionary    input_tray=${trays}
    Compare Print Job Tickets    ${PJ_DIR}/ticket_tray4.pcl    ${PJ_DIR}/ticket_ps_inputslot_tray2.ps    ignore_properties=pjl,orientation,duplex,paper_size    mapping=${mapping}
