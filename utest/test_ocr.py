from DocTest.DocumentRepresentation import DocumentRepresentation
import pytest
from pathlib import Path

def test_big_text_from_image(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'text_big.png')
    assert 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' in img.get_text()

def test_medium_text_from_image(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'text_medium.png')
    assert 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' in img.get_text()

def test_small_text_from_image(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'text_small.png')
    assert '1234567890' in img.get_text()

#@pytest.mark.skip(reason="Disabled until OCR using tesseract has been improved. Only partial text match.")
def test_simple_text_from_pdf(testdata_dir):
    # Perform a fuzzy match
    # Check if FB1DES0A3D5EFE2A60B0B1AE616C653 is in the text
    # But: 0 can be O, 5 can be S and I can be l

    img = DocumentRepresentation(testdata_dir / 'sample_1_page.pdf')

    import re
    text = img.get_text()
    assert re.search(r'FB1DES0A3D5EFE2A60B0B1AE616C653', text)

    assert 'FB1DES0A3D5EFE2A60B0B1AE616C653' in img.get_text()

    


def test_text_on_colored_background(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'Beach_date.png')
    assert "01-Jan-2021" in img.get_text()
    assert "123456789" in img.get_text()

def test_image_text_content_with_east(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'birthday_1080_date_id.png', ocr_engine='east')
    img.apply_ocr()
    assert "01-Jan-2021" in img.get_text()
    assert "ABCDEFGHI" in img.get_text()

def test_image_text_content_with_pytesseract(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'birthday_1080_date_id.png')
    assert "01-Jan-2021" in img.get_text()
    assert "ABCDEFGHI" in img.get_text()

def test_image_text_content_with_pytesseract_custom_options_01(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'text_small.png')
    assert "ABCDEFGHI" in img.get_text()
    assert "abcdefghi" in img.get_text()
    assert "1234567890" in img.get_text()

def test_image_text_content_with_pytesseract_custom_options_02(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'text_small.png')
    img.get_text(tesseract_config='--psm 6')
    assert "ABCDEFGHI" in img.get_text()
    assert "abcdefghi" in img.get_text()
    assert "1234567890" in img.get_text()

def test_image_text_content_with_pytesseract_custom_options_03(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'text_big.png')
    assert "ABCDEFGHI" in img.get_text()
    assert "abcdefghi" in img.get_text()
    assert "1234567890" in img.get_text()

def test_white_text_on_dark_background(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'whitetext_blackbackground.png')
    assert '0123456789' in img.get_text()

@pytest.mark.skip(reason="To be implemented")
def test_text_in_low_resolution():
    pass

@pytest.mark.skip(reason="To be implemented")
def test_text_in_overlapping_elements():
    pass

def test_text_on_colored_background_with_east(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'Beach_date.png', ocr_engine="east" )
    assert "01-Jan-2021" in img.get_text()
    assert "123456789" in img.get_text()
    assert "SOUVENIR" in img.get_text()

def test_ocr_in_hires_without_rerender(testdata_dir):
    import cv2
    low_res_image = cv2.imread(str(testdata_dir / 'birthday_1080_date_id.png'))
    # resize image to 10x bigger
    low_res_image = cv2.resize(low_res_image, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(testdata_dir / 'birthday_1080_date_id_10x.png'), low_res_image)
    img = DocumentRepresentation(testdata_dir / 'birthday_1080_date_id_10x.png')
    assert "01-Jan-2021" in img.get_text()
