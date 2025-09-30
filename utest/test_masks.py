from DocTest.DocumentRepresentation import DocumentRepresentation
from DocTest.DocumentRepresentation import Page
import pytest
from pathlib import Path
import numpy as np

pytestmark = [
    pytest.mark.usefixtures("fake_ocrs"),
    pytest.mark.usefixtures("require_image_samples"),
]

def test_image_area_mask(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'Beach_date.png', ignore_area_file=testdata_dir / 'area_mask.json')
    assert len(img.abstract_ignore_areas)==1
    assert np.not_equal(img.pages[0].get_image_with_ignore_areas(), img.pages[0].image).any()


def test_image_text_mask(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'Beach_date.png', ignore_area_file=testdata_dir / 'pattern_mask.json')
    assert len(img.abstract_ignore_areas)==2
    assert np.not_equal(img.pages[0].get_image_with_ignore_areas(), img.pages[0].image).any()

def test_image_text_mask_with_east(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'Beach_date.png', ignore_area_file=testdata_dir / 'pattern_mask.json', ocr_engine='east')
    assert len(img.abstract_ignore_areas)>=1
    assert np.not_equal(img.pages[0].get_image_with_ignore_areas(), img.pages[0].image).any()

def test_pdf_area_mask(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'sample_1_page.pdf', ignore_area_file=testdata_dir / 'pdf_area_mask.json')
    assert len(img.abstract_ignore_areas)==1
    assert np.not_equal(img.pages[0].get_image_with_ignore_areas(), img.pages[0].image).any()

def test_pdf_text_mask(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'sample_1_page.pdf', ignore_area_file=testdata_dir / 'pdf_pattern_mask.json')
    assert len(img.abstract_ignore_areas)==2
    assert np.not_equal(img.pages[0].get_image_with_ignore_areas(), img.pages[0].image).any()

def test_pdf_word_pattern_mask_dimensions(testdata_dir):
    mask = {
        'page': 'all',
        'type': 'word_pattern',
        'pattern': '12345678901234'
    }
    doc = DocumentRepresentation(testdata_dir / 'sample_1_page.pdf', ignore_area=mask)
    area = doc.pages[0].pixel_ignore_areas[0]
    assert area['width'] == 233
    assert area['height'] == 31

def test_pdf_pattern_mask_dimensions(testdata_dir):
    mask = {
        'page': 'all',
        'type': 'pattern',
        'pattern': '.*RTMOE.*'
    }
    doc = DocumentRepresentation(testdata_dir / 'sample_1_page.pdf', ignore_area=mask)
    area = doc.pages[0].pixel_ignore_areas[0]
    assert area['width'] == 516
    assert area['height'] == 31
def test_pattern_mask_handles_umlauts_and_symbols():
    image = np.zeros((50, 200, 3), dtype=np.uint8)
    page = Page(image, page_number=1, dpi=200)
    page.ocr_performed = True
    page.ocr_text_data = {
        'text': ['Änderung', 'Café-123#'],
        'left': [10, 60],
        'top': [5, 10],
        'width': [40, 80],
        'height': [15, 18],
        'conf': ['95', '95'],
    }

    umlaut_mask = {
        'type': 'pattern',
        'pattern': '(?i).*ÄNDERUNG.*',
        'xoffset': 2,
        'yoffset': 3,
    }
    symbol_mask = {
        'type': 'pattern',
        'pattern': '(?i).*CAFÉ-123#.*',
        'xoffset': 1,
        'yoffset': 1,
    }

    page._process_pattern_ignore_area_from_ocr(umlaut_mask)
    page._process_pattern_ignore_area_from_ocr(symbol_mask)

    assert len(page.pixel_ignore_areas) == 2
    first, second = page.pixel_ignore_areas
    assert first['x'] <= 12 and first['width'] >= 40
    assert second['x'] <= 61 and second['width'] >= 80
