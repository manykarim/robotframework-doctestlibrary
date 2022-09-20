from DocTest.CompareImage import CompareImage
import pytest
from pathlib import Path

def test_big_text_from_image(testdata_dir):
    img = CompareImage(testdata_dir / 'text_big.png')
    img.get_ocr_text_data()
    assert 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' in img.text_content[0]['text']

def test_medium_text_from_image(testdata_dir):
    img = CompareImage(testdata_dir / 'text_medium.png')
    img.get_ocr_text_data()
    assert 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' in img.text_content[0]['text']

def test_small_text_from_image(testdata_dir):
    img = CompareImage(testdata_dir / 'text_small.png')
    img.get_ocr_text_data()
    assert '1234567890' in img.text_content[0]['text']

#@pytest.mark.skip(reason="Disabled until OCR using tesseract has been improved. Only partial text match.")
def test_simple_text_from_pdf(testdata_dir):
    img = CompareImage(testdata_dir / 'sample_1_page.pdf')
    img.get_ocr_text_data()
    assert 'FB1DES0A3D5EFE2A60B0B1AE616C653' in img.text_content[0]['text']

def test_text_on_colored_background(testdata_dir):
    img = CompareImage(testdata_dir / 'Beach_date.png')
    img.get_ocr_text_data()
    assert "01-Jan-2021" in img.text_content[0]['text']
    assert "123456789" in img.text_content[0]['text']

@pytest.mark.skip(reason="Currently, tesseract is not so good at recognizing bright text")
def test_white_text_on_dark_background(testdata_dir):
    img = CompareImage(testdata_dir / 'whitetext_blackbackground.png')
    img.get_ocr_text_data()
    assert '0123456789' in img.text_content[0]['text']

@pytest.mark.skip(reason="To be implemented")
def test_text_in_low_resolution():
    pass

@pytest.mark.skip(reason="To be implemented")
def test_text_in_overlapping_elements():
    pass

def test_text_on_colored_background_with_east(testdata_dir):
    img = CompareImage(testdata_dir / 'Beach_date.png')
    img.get_text_content_with_east()
    assert any('01-Jan-2021' in s for s in img.text_content[0]['text'])
    assert any('123456789' in s for s in img.text_content[0]['text'])
    assert any('SOUVENIR' in s for s in img.text_content[0]['text'])
