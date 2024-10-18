from DocTest.VisualTest import VisualTest
import pytest
from pathlib import Path

def test_get_text_from_image(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'text_big.png')
    text = visual_tester.get_text_from_document(ref_image)
    assert 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' in text

def test_get_text_from_pdf(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'sample.pdf')
    text = visual_tester.get_text_from_document(ref_image)
    assert 'THE TEST SHIPPER' in text

def test_get_text_from_pdf_does_not_match(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'sample.pdf')
    text = visual_tester.get_text_from_document(ref_image)
    assert 'THIS STRING DOES NOT EXIST' not in text

def test_compare_two_different_images_from_url():
    visual_tester = VisualTest()
    ref_image='https://github.com/manykarim/robotframework-doctestlibrary/raw/main/utest/testdata/birthday_left.png'
    test_image='https://github.com/manykarim/robotframework-doctestlibrary/raw/main/utest/testdata/birthday_right.png'
    with pytest.raises(AssertionError, match='The compared images are different.'):
        visual_tester.compare_images(ref_image, test_image)

def test_compare_two_equal_images_from_url():
    visual_tester = VisualTest()
    ref_image='https://github.com/manykarim/robotframework-doctestlibrary/raw/main/utest/testdata/birthday_left.png'
    test_image='https://github.com/manykarim/robotframework-doctestlibrary/raw/main/utest/testdata/birthday_left_copy.png'
    visual_tester.compare_images(ref_image, test_image)

def test_compare_birthday_image_with_noise(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'birthday_1080.png')
    cand_image=str(testdata_dir / 'birthday_1080_noise_001.png')
    with pytest.raises(AssertionError, match='The compared images are different.'):
        visual_tester.compare_images(ref_image, cand_image)

def test_compare_birthday_image_with_different_noise(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'birthday_1080_noise_001.png')
    cand_image=str(testdata_dir / 'birthday_1080_noise_002.png')
    with pytest.raises(AssertionError, match='The compared images are different.'):
        visual_tester.compare_images(ref_image, cand_image)


def test_compare_birthday_image_with_noise_and_lower_threshold(testdata_dir):
    visual_tester = VisualTest(threshold=0.2)
    ref_image=str(testdata_dir / 'birthday_1080.png')
    cand_image=str(testdata_dir / 'birthday_1080_noise_001.png')
    visual_tester.compare_images(ref_image, cand_image)

def test_compare_birthday_image_with_noise_and_blurring(testdata_dir):
    visual_tester = VisualTest(threshold=0.0058)
    ref_image=str(testdata_dir / 'birthday_1080.png')
    cand_image=str(testdata_dir / 'birthday_1080_noise_001.png')
    visual_tester.compare_images(ref_image, cand_image, blur=True)

def test_text_on_colored_background_with_east(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'Beach_date.png')
    text = visual_tester.get_text_from_document(ref_image, ocr_engine='east')
    assert any('01-Jan-2021' in s for s in text)
    assert any('123456789' in s for s in text)
    assert any('SOUVENIR' in s for s in text)

def test_moved_difference_for_pdf_on_white_background_within_tolerance_default(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'sample_1_page.pdf')
    cand_image=str(testdata_dir / 'sample_1_page_moved.pdf')
    visual_tester.compare_images(ref_image, cand_image, move_tolerance=8)

def test_moved_difference_for_pdf_on_white_background_within_tolerance_classic(testdata_dir):
    visual_tester = VisualTest(movement_detection="classic")
    ref_image=str(testdata_dir / 'sample_1_page.pdf')
    cand_image=str(testdata_dir / 'sample_1_page_moved.pdf')
    visual_tester.compare_images(ref_image, cand_image, move_tolerance=8)

def test_moved_difference_for_pdf_on_white_background_within_tolerance_template(testdata_dir):
    visual_tester = VisualTest(movement_detection="template")
    ref_image=str(testdata_dir / 'sample_1_page.pdf')
    cand_image=str(testdata_dir / 'sample_1_page_moved.pdf')
    visual_tester.compare_images(ref_image, cand_image, move_tolerance=8)

def test_moved_difference_for_pdf_on_white_background_within_tolerance_orb(testdata_dir):
    visual_tester = VisualTest(movement_detection="orb")
    ref_image=str(testdata_dir / 'sample_1_page.pdf')
    cand_image=str(testdata_dir / 'sample_1_page_moved.pdf')
    visual_tester.compare_images(ref_image, cand_image, move_tolerance=8)

def test_moved_difference_for_pdf_on_white_background_outside_tolerance_default(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'sample_1_page.pdf')
    cand_image=str(testdata_dir / 'sample_1_page_moved.pdf')
    with pytest.raises(AssertionError, match='The compared images are different.'):
        visual_tester.compare_images(ref_image, cand_image, move_tolerance=7)

def test_moved_difference_for_pdf_on_white_background_outside_tolerance_classic(testdata_dir):
    visual_tester = VisualTest(movement_detection="classic")
    ref_image=str(testdata_dir / 'sample_1_page.pdf')
    cand_image=str(testdata_dir / 'sample_1_page_moved.pdf')
    with pytest.raises(AssertionError, match='The compared images are different.'):
        visual_tester.compare_images(ref_image, cand_image, move_tolerance=7)

def test_moved_difference_for_pdf_on_white_background_outside_tolerance_template(testdata_dir):
    visual_tester = VisualTest(movement_detection="template")
    ref_image=str(testdata_dir / 'sample_1_page.pdf')
    cand_image=str(testdata_dir / 'sample_1_page_moved.pdf')
    with pytest.raises(AssertionError, match='The compared images are different.'):
        visual_tester.compare_images(ref_image, cand_image, move_tolerance=7)

def test_moved_difference_for_pdf_on_white_background_outside_tolerance_orb(testdata_dir):
    visual_tester = VisualTest(movement_detection="orb")
    ref_image=str(testdata_dir / 'sample_1_page.pdf')
    cand_image=str(testdata_dir / 'sample_1_page_moved.pdf')
    with pytest.raises(AssertionError, match='The compared images are different.'):
        visual_tester.compare_images(ref_image, cand_image, move_tolerance=7)

def test_get_barcode_values(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'sample_barcodes.pdf')
    barcode_data = visual_tester.get_barcodes_from_document(ref_image)
    # barcode_data is a list of dictionaries
    # Only collect all the values of the key "value" from the dictionaries
    barcode_data = [d['value'] for d in barcode_data]
    assert barcode_data == ['This is a QR Code by TEC-IT', 'This is a QR Code by TEC-IT for mobile applications', '1234567890', 'ABC-1234', 'ABC-1234-/+', 'ABC-abc-1234', '0012345000065', '90311017', '0725272730706', '9780201379624', 'This is a Data Matrix by TEC-IT', 'This is a Data Matrix by TEC-IT']

def test_get_barcode_coordinates(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'sample_barcodes.pdf')
    barcode_coordinates = visual_tester.get_barcodes_from_document(ref_image)
    # barcode_coordinates is a list of dictionaries
    # Only collect all the values of the keys "x", "y", "width" and "height" from the dictionaries
    barcode_coordinates = [{k: d[k] for k in ('x', 'y', 'width', 'height')} for d in barcode_coordinates]
    assert barcode_coordinates == [{'x':757, 'y':1620, 'width':207, 'height':207}, 
                                   {'x':1198, 'y':1598, 'width':244, 'height':244}, 
                                   {'x':160, 'y':1651, 'width':413, 'height':122}, 
                                   {'x':467, 'y':1309, 'width':663, 'height':159}, 
                                   {'x':509, 'y':1021, 'width':564, 'height':159}, 
                                   {'x':485, 'y':725, 'width':629, 'height':159}, 
                                   {'x':312, 'y':399, 'width':204, 'height':159}, 
                                   {'x':1039, 'y':399, 'width':278, 'height':159}, 
                                   {'x':984, 'y':93, 'width':396, 'height':158}, 
                                   {'x':236, 'y':90, 'width':396, 'height':158}, 
                                   {'x':480, 'y':2025, 'width':184, 'height':183}, 
                                   {'x':979, 'y':1971, 'width':272, 'height':271}]

def test_get_barcode_all(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'sample_barcodes.pdf')
    barcode_all = visual_tester.get_barcodes_from_document(ref_image)
    barcode_values = [d['value'] for d in barcode_all]
    barcode_coordinates = [{k: d[k] for k in ('x', 'y', 'width', 'height')} for d in barcode_all]    
    assert barcode_values == ['This is a QR Code by TEC-IT', 'This is a QR Code by TEC-IT for mobile applications', '1234567890', 'ABC-1234', 'ABC-1234-/+', 'ABC-abc-1234', '0012345000065', '90311017', '0725272730706', '9780201379624', 'This is a Data Matrix by TEC-IT', 'This is a Data Matrix by TEC-IT']
    assert barcode_coordinates == [{'x':757, 'y':1620, 'width':207, 'height':207}, 
                                   {'x':1198, 'y':1598, 'width':244, 'height':244}, 
                                   {'x':160, 'y':1651, 'width':413, 'height':122}, 
                                   {'x':467, 'y':1309, 'width':663, 'height':159}, 
                                   {'x':509, 'y':1021, 'width':564, 'height':159}, 
                                   {'x':485, 'y':725, 'width':629, 'height':159}, 
                                   {'x':312, 'y':399, 'width':204, 'height':159}, 
                                   {'x':1039, 'y':399, 'width':278, 'height':159}, 
                                   {'x':984, 'y':93, 'width':396, 'height':158}, 
                                   {'x':236, 'y':90, 'width':396, 'height':158}, 
                                   {'x':480, 'y':2025, 'width':184, 'height':183}, 
                                   {'x':979, 'y':1971, 'width':272, 'height':271}]

def test_find_existing_partial_image_with_template(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'birthday_1080.png')
    template_image=str(testdata_dir / 'birthday_partial_banana.png')
    position = visual_tester.image_should_contain_template(ref_image, template_image, detection='template')
    assert position['pt1'] == (154, 1001)
    assert position['pt2'] == (351, 1152)

def test_find_no_existing_partial_image_with_template(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'birthday_1080.png')
    template_image=str(testdata_dir / 'text.png')
    with pytest.raises(AssertionError, match='The Template was not found in the Image'):
        visual_tester.image_should_contain_template(ref_image, template_image, detection='template')

def test_find_existing_partial_image_with_orb(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'birthday_1080.png')
    template_image=str(testdata_dir / 'birthday_partial_banana.png')
    position = visual_tester.image_should_contain_template(ref_image, template_image, detection='orb')
    assert 153 <= position['pt1'][0] <= 156
    assert 1000 <= position['pt1'][1] <= 1003
    assert 350 <= position['pt2'][0] <= 353
    assert 1150 <= position['pt2'][1] <= 1154

def test_find_existing_partial_image_with_sift(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'birthday_1080.png')
    template_image=str(testdata_dir / 'birthday_partial_banana.png')
    position = visual_tester.image_should_contain_template(ref_image, template_image, detection='sift')

    # Make the position assertion fuzzy to avoid false negatives
    # If the match differs in 1-3 pixels, it is still considered a match
    assert 153 <= position['pt1'][0] <= 156
    assert 1000 <= position['pt1'][1] <= 1003
    assert 350 <= position['pt2'][0] <= 353
    assert 1150 <= position['pt2'][1] <= 1154

    
def test_find_no_existing_partial_image_with_sift(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'birthday_1080.png')
    template_image=str(testdata_dir / 'text.png')
    with pytest.raises(AssertionError, match='The Template was not found in the Image'):
        visual_tester.image_should_contain_template(ref_image, template_image, detection='sift')