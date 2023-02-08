*** Settings ***
Library    DocTest.VisualTest    show_diff=true    take_screenshots=true    screenshot_format=png    movement_detection=${MOVEMENT_DETECTION}
Library    Collections

*** Variables ***
${MOVEMENT_DETECTION}    orb

*** Test Cases ***
Compare two different PDF Files with moved text within tolerance
    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf    move_tolerance=20    get_pdf_content=${true}

Compare two different PDF Files with moved text outside tolerance
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf    move_tolerance=5    get_pdf_content=${true}

Compare two different Image Files with moved text
    Compare Images    testdata/small_A_reference.png    testdata/small_A_moved.png    move_tolerance=60    ignore_watermarks=False

Compare two different PDF Files with moved text within tolerance using OCR
    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf    move_tolerance=20    force_ocr=True

Compare two different PDF Files with moved text outside tolerance using OCR
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf    move_tolerance=5    force_ocr=True

Compare two small Image Files with moved text without OCR
    Compare Images    testdata/small_A_reference.png    testdata/small_A_moved.png    move_tolerance=60    ignore_watermarks=False

Compare two Images with colored background and moved object outside tolerance
    [Tags]    robot:skip-on-failure
    Compare Images    utest/testdata/Pattern_with_objects.png    utest/testdata/Pattern_with_objects_moved.png    move_tolerance=60    ignore_watermarks=False

Compare two different PDF Files with moved text within tolerance without OCR
    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf    move_tolerance=10

Compare two different PDF Files with moved text outside tolerance without OCR
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf    move_tolerance=7