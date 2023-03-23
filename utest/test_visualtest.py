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
    visual_tester = VisualTest(threshold=0.005)
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
    assert barcode_data == ['This is a QR Code by TEC-IT', 'This is a QR Code by TEC-IT for mobile applications', '1234567890', 'ABC-1234', 'ABC-1234-/+', 'ABC-abc-1234', '0012345000065', '90311017', '0725272730706', '9780201379624', 'This is a Data Matrix by TEC-IT', 'This is a Data Matrix by TEC-IT']

def test_get_barcode_coordinates(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'sample_barcodes.pdf')
    barcode_coordinates = visual_tester.get_barcodes_from_document(ref_image, return_type='coordinates')
    assert barcode_coordinates == [(757, 1620, 207, 207), (1198, 1598, 244, 244), (160, 1651, 122, 413), (467, 1309, 159, 663), (509, 1021, 159, 564), (485, 725, 159, 629), (312, 399, 159, 204), (1039, 399, 159, 278), (984, 93, 158, 396), (236, 90, 158, 396), (480, 2025, 183, 184), (979, 1971, 271, 272)]

def test_get_barcode_all(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'sample_barcodes.pdf')
    barcode_all = visual_tester.get_barcodes_from_document(ref_image, return_type='all')
    assert barcode_all[0] == ['This is a QR Code by TEC-IT', 'This is a QR Code by TEC-IT for mobile applications', '1234567890', 'ABC-1234', 'ABC-1234-/+', 'ABC-abc-1234', '0012345000065', '90311017', '0725272730706', '9780201379624', 'This is a Data Matrix by TEC-IT', 'This is a Data Matrix by TEC-IT']
    assert barcode_all[1] == [(757, 1620, 207, 207), (1198, 1598, 244, 244), (160, 1651, 122, 413), (467, 1309, 159, 663), (509, 1021, 159, 564), (485, 725, 159, 629), (312, 399, 159, 204), (1039, 399, 159, 278), (984, 93, 158, 396), (236, 90, 158, 396), (480, 2025, 183, 184), (979, 1971, 271, 272)]
