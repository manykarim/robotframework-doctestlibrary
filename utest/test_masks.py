from DocTest.CompareImage import CompareImage
import pytest
from pathlib import Path

def test_image_coordinate_mask(testdata_dir):
    pass

def test_image_area_mask(testdata_dir):
    img = CompareImage(testdata_dir / 'Beach_date.png', placeholder_file=testdata_dir / 'area_mask.json')
    assert len(img.placeholders)==1
    img_with_mask = img.get_image_with_placeholders()
    assert img != img_with_mask

def test_image_text_mask(testdata_dir):
    img = CompareImage(testdata_dir / 'Beach_date.png', placeholder_file=testdata_dir / 'pattern_mask.json')
    assert len(img.placeholders)==2
    img_with_mask = img.get_image_with_placeholders()
    assert img != img_with_mask

def test_image_barcode_mask(testdata_dir):
    pass

def test_pdf_coordinate_mask(testdata_dir):
    pass

def test_pdf_area_mask(testdata_dir):
    img = CompareImage(testdata_dir / 'sample_1_page.pdf', placeholder_file=testdata_dir / 'pdf_area_mask.json')
    assert len(img.placeholders)==1
    img_with_mask = img.get_image_with_placeholders()
    assert img != img_with_mask

def test_pdf_text_mask(testdata_dir):
    img = CompareImage(testdata_dir / 'sample_1_page.pdf', placeholder_file=testdata_dir / 'pdf_pattern_mask.json')
    assert len(img.placeholders)==3
    img_with_mask = img.get_image_with_placeholders()
    assert img != img_with_mask

def test_pdf_barcode_mask(testdata_dir):
    pass

