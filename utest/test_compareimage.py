from DocTest.CompareImage import CompareImage
import pytest
from pathlib import Path
import numpy

def test_single_png(testdata_dir):
    img = CompareImage(testdata_dir / 'text_big.png')
    assert len(img.opencv_images)==1
    assert type(img.opencv_images)==list
    type(img.opencv_images[0])==numpy.ndarray

def test_single_pdf(testdata_dir):
    img = CompareImage(testdata_dir / 'sample_1_page.pdf')
    assert len(img.opencv_images)==1
    assert type(img.opencv_images)==list
    type(img.opencv_images[0])==numpy.ndarray
    pass

def test_multipage_pdf(testdata_dir):
    img = CompareImage(testdata_dir / 'sample.pdf')
    assert len(img.opencv_images)==2
    assert type(img.opencv_images)==list
    type(img.opencv_images[0])==numpy.ndarray
    type(img.opencv_images[1])==numpy.ndarray
    pass

def test_huge_pdf(testdata_dir):
    pass

def test_image_text_content(testdata_dir):
    img = CompareImage(testdata_dir / 'Beach_date.png')
    img.get_ocr_text_data()
    assert "01-Jan-2021" in img.text_content[0]['text']
    assert "123456789" in img.text_content[0]['text']

def test_pdf_text_content(testdata_dir):
    img = CompareImage(testdata_dir / 'sample_1_page.pdf')
    assert len(img.mupdfdoc.get_page_text(0, "WORDS"))>0

def test_non_existing_file(testdata_dir):
    with pytest.raises(AssertionError):
        img = CompareImage(testdata_dir / 'does_not_exist.png')
        
def test_corrupt_image(testdata_dir):
    with pytest.raises(AssertionError):
        img = CompareImage(testdata_dir / 'corrupt_image.png')

def test_corrupt_pdf(testdata_dir):
    with pytest.raises(AssertionError):
        img = CompareImage(testdata_dir / 'corrupt_pdf.pdf')
