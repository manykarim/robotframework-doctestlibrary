from DocTest.DocumentRepresentation import DocumentRepresentation
import pytest
from pathlib import Path
import numpy

def test_single_png_with_barcode(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'datamatrix.png', contains_barcodes=True)
    assert len(img.get_barcodes()) == 2
    assert len(img.pages[0].barcodes) == 2
    assert len(img.pages[0].pixel_ignore_areas) == 2
    

def test_single_pdf_with_barcode(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'sample_1_page_with_barcodes.pdf', contains_barcodes=True)
    assert len(img.get_barcodes()) == 2
    assert len(img.pages[0].barcodes) == 2
    assert len(img.pages[0].pixel_ignore_areas) == 2

def test_single_pdf_without_barcode(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'sample_1_page.pdf', contains_barcodes=True)
    assert len(img.get_barcodes()) == 0
    assert len(img.pages[0].barcodes) == 0
    assert len(img.pages[0].pixel_ignore_areas) == 0

def test_barcode_sample_page(testdata_dir):
    img = DocumentRepresentation(testdata_dir / 'sample_barcodes.pdf', contains_barcodes=True)
    assert len(img.get_barcodes()) == 12
    assert len(img.pages[0].barcodes) == 12
    assert len(img.pages[0].pixel_ignore_areas) == 12
