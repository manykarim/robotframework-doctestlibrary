from DocTest.DocumentRepresentation import DocumentRepresentation
import pytest
from pathlib import Path
import numpy as np

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
    assert len(img.abstract_ignore_areas)==3
    assert np.not_equal(img.pages[0].get_image_with_ignore_areas(), img.pages[0].image).any()