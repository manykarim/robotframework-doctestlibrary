*** Settings ***
Library    DocTest.VisualTest    show_diff=true    take_screenshots=true    screenshot_format=png    #pdf_rendering_engine=ghostscript
Library    Collections

*** Test Cases ***
Compare two Beach images
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/Beach_left.jpg    testdata/Beach_right.jpg

Compare two Farm images
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/Farm_left.jpg    testdata/Farm_right.jpg

Compare two Farm images with date pattern
    Compare Images    testdata/Beach_date.png    testdata/Beach_left.png    placeholder_file=testdata/pattern_mask.json

Compare two Farm images with area mask as list
    ${top_mask}    Create Dictionary    page=1    type=area    location=top    percent=10
    ${bottom_mask}    Create Dictionary    page=all    type=area    location=bottom    percent=10
    ${masks}    Create List    ${top_mask}    ${bottom_mask}
    Compare Images    testdata/Beach_date.png    testdata/Beach_left.png    mask=${masks}

Compare two Farm images with area mask as string
    Compare Images    testdata/Beach_date.png    testdata/Beach_left.png    mask=top:10;bottom:10

Compare two Farm images with date pattern and east detection
    Compare Images    testdata/Beach_date.png    testdata/Beach_left.png    placeholder_file=testdata/pattern_mask.json    ocr_engine=east

Compare two different PDF Files
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_changed.pdf

Compare two different PDF Files with pattern mask
    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_changed.pdf    placeholder_file=testdata/pdf_pattern_mask.json

Compare two different PDF Files with area mask
    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_changed.pdf    placeholder_file=testdata/pdf_area_mask.json

Compare two different PDF Files with pattern mask as parameter
    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_changed.pdf    mask=[{"page": "all","name": "Ref ID","type": "pattern","pattern": ".*RTMOE.*"},{"page": "all","name": "Job ID","type": "pattern","pattern": "JobID.*"}]

Compare two different PDF Files with moved text
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf

Compare two different PDF Files with moved text but same content OCR
    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf    check_text_content=${true}    force_ocr=${true}

Compare two different PDF Files with moved text but same content pdfminer
    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf    check_text_content=${true}

Compare two different PDF Files with moved text but and different content pdfminer
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved_and_different.pdf    check_text_content=${true}

Compare two different PDF Files with moved text within tolerance
    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf    move_tolerance=20

Compare two different PDF Files with moved text outside tolerance
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf    move_tolerance=5

Compare two different Image Files with moved text
    Compare Images    testdata/small_A_reference.png    testdata/small_A_moved.png    move_tolerance=60    ignore_watermarks=False

Compare two different PDF Files with moved text within tolerance using OCR
    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf    move_tolerance=20    force_ocr=True

Compare two different PDF Files with moved text outside tolerance using OCR
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/sample_1_page.pdf    testdata/sample_1_page_moved.pdf    move_tolerance=5    force_ocr=True

Compare two small Image Files with moved text without OCR
    Compare Images    testdata/small_A_reference.png    testdata/small_A_moved.png    move_tolerance=60    ignore_watermarks=False

Compare Images With Different Shapes
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/Beach_left.jpg    testdata/Beach_cropped.jpg

Compare Text Content from PDF
    ${text}     Get Text From Document    testdata/sample.pdf    *=    THE TEST SHIPPER

Compare Text Content from Image
    ${text}     Get Text From Document    testdata/Beach_date.png    *=    123456789
    ${text}     Get Text From Document    testdata/Beach_date.png    *=    01-Jan-2021
    ${text}     Get Text From Document    testdata/Beach_date.png    !=    Not Equal
    ${text}     Get Text From Document    testdata/Beach_date.png    not contains    Not Equal


Compare Text Content from Image with east
    Set Ocr Engine    east
    ${text}     Get Text From Document    testdata/Beach_date.png
    Should Contain    ${text}    01-Jan-2021
    Should Contain    ${text}    SOUVENIR
    Should Contain    ${text}    123456789
    ${text}     Get Text From Document    testdata/Beach_date.png    *=    01-Jan-2021

Compare Images And Resize With Different Shapes
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/Beach_left.jpg    testdata/Beach_cropped.jpg    resize_candidate=True

Compare Images Different Images With Full Watermark
    Compare Images    testdata/Beach_date.png    testdata/Beach_left.png    watermark_file=testdata/Beach_date_mask_full.png

Compare Images Different Images With Partial Watermark
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/Beach_date.png    testdata/Beach_left.png    watermark_file=testdata/Beach_date_mask_partial.png

Compare Images Different Images With Smaller Watermark
    Compare Images    testdata/Beach_date.png    testdata/Beach_left.png    watermark_file=testdata/Beach_date_mask_full_smaller.png

Compare two Beach images with differences and higher threshold
    Compare Images    testdata/Beach_left.jpg    testdata/Beach_right.jpg    threshold=0.05

Compare two Beach images with differences and higher threshold and block based ssim
    Run Keyword And Expect Error    The compared images are different.    Compare Images    testdata/Beach_left.jpg    testdata/Beach_right.jpg    threshold=0.05    block_based_ssim=True

Compare two Beach images with differences and higher threshold and block based ssim with higher block size
    Compare Images    testdata/Beach_left.jpg    testdata/Beach_right.jpg    threshold=0.05    block_based_ssim=True    block_size=256
