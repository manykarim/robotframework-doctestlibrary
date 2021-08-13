from DocTest.CompareImage import CompareImage
#from ..DocTest.CompareImage import CompareImage
import pytest
from pathlib import Path

def test_big_text_from_image(testdata_dir):
    img = CompareImage(testdata_dir / 'text_big.png')
    img.get_ocr_text_data()
    assert 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' in img.text_content[0]['text']
    pass

def test_medium_text_from_image(testdata_dir):
    img = CompareImage(testdata_dir / 'text_medium.png')
    img.get_ocr_text_data()
    assert 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' in img.text_content[0]['text']
    pass

def test_small_text_from_image(testdata_dir):
    img = CompareImage(testdata_dir / 'text_small.png')
    img.get_ocr_text_data()
    assert 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' in img.text_content[0]['text']
    pass


def test_simple_text_from_pdf():
    pass

def test_text_on_colored_background():
    pass

def test_white_text_on_dark_background():
    pass

def test_text_in_low_resolution():
    pass

def test_text_in_overlapping_elements():
    pass