from DocTest.DocumentRepresentation import DocumentRepresentation
import pytest
from pathlib import Path
import numpy

IGNORE_AREA_REQUIRED_KEYS = {"x", "y", "height", "width"}

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

def test_ignore_area_schema_for_datamatrix(testdata_dir):
    """Verify that datamatrix ignore areas have the correct keys (x, y, height, width)."""
    img = DocumentRepresentation(testdata_dir / 'datamatrix.png', contains_barcodes=True)
    for area in img.pages[0].pixel_ignore_areas:
        assert set(area.keys()) == IGNORE_AREA_REQUIRED_KEYS, (
            f"Ignore area has wrong keys: {set(area.keys())}. "
            f"Expected: {IGNORE_AREA_REQUIRED_KEYS}"
        )

def test_ignore_area_schema_for_zbar_barcodes(testdata_dir):
    """Verify that zbar barcode ignore areas have the correct keys."""
    img = DocumentRepresentation(testdata_dir / 'sample_1_page_with_barcodes.pdf', contains_barcodes=True)
    for area in img.pages[0].pixel_ignore_areas:
        assert set(area.keys()) == IGNORE_AREA_REQUIRED_KEYS, (
            f"Ignore area has wrong keys: {set(area.keys())}. "
            f"Expected: {IGNORE_AREA_REQUIRED_KEYS}"
        )

def test_ignore_area_coordinates_are_integers(testdata_dir):
    """Verify that all ignore area coordinates are integers."""
    img = DocumentRepresentation(testdata_dir / 'datamatrix.png', contains_barcodes=True)
    for area in img.pages[0].pixel_ignore_areas:
        for key in IGNORE_AREA_REQUIRED_KEYS:
            assert isinstance(area[key], int), (
                f"Ignore area key '{key}' should be int, got {type(area[key]).__name__}"
            )

def test_get_image_with_ignore_areas_does_not_crash_datamatrix(testdata_dir):
    """Regression test: get_image_with_ignore_areas must not crash when datamatrices are detected.

    Before the fix, identify_datamatrices() stored the key "y:" instead of "y",
    causing a TypeError in get_image_with_ignore_areas() when it tried arithmetic
    on None.
    """
    img = DocumentRepresentation(testdata_dir / 'datamatrix.png', contains_barcodes=True)
    page = img.pages[0]
    assert len(page.pixel_ignore_areas) > 0, "Expected at least one ignore area"
    result = page.get_image_with_ignore_areas()
    assert result is not None
    assert result.shape == page.image.shape

def test_get_image_with_ignore_areas_does_not_crash_zbar(testdata_dir):
    """Verify get_image_with_ignore_areas works for zbar-detected barcodes too."""
    img = DocumentRepresentation(testdata_dir / 'sample_1_page_with_barcodes.pdf', contains_barcodes=True)
    page = img.pages[0]
    assert len(page.pixel_ignore_areas) > 0, "Expected at least one ignore area"
    result = page.get_image_with_ignore_areas()
    assert result is not None
    assert result.shape == page.image.shape
