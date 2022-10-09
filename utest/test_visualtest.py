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
    ref_image='https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/Beach_left.jpg'
    test_image='https://github.com/manykarim/robotframework-doctestlibrary/raw/main/testdata/Beach_right.jpg'
    with pytest.raises(AssertionError):
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