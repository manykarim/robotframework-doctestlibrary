from DocTest.CompareImage import CompareImage
import pytest
from pathlib import Path
import numpy

def test_single_png_with_barcode(testdata_dir):
    img = CompareImage(testdata_dir / 'datamatrix.png', contains_barcodes=True)
    assert len(img.placeholders) == 2

def test_single_pdf_with_barcode(testdata_dir):
    img = CompareImage(testdata_dir / 'sample_1_page_with_barcodes.pdf', contains_barcodes=True)
    assert len(img.placeholders) == 2

def test_single_pdf_without_barcode(testdata_dir):
    img = CompareImage(testdata_dir / 'sample_1_page.pdf', contains_barcodes=True)
    assert len(img.placeholders) == 0