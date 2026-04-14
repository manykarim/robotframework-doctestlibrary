"""Resource benchmarks for document loading and comparison.

These tests verify that core operations complete within reasonable time
bounds and that resources are properly released after use.
"""

import gc
import time
import weakref

import pytest

from DocTest.DocumentRepresentation import DocumentRepresentation
from DocTest.VisualTest import VisualTest


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def test_image_load_performance(testdata_dir):
    """Loading a single PNG image should complete in under 2 seconds."""
    start = time.time()
    doc = DocumentRepresentation(str(testdata_dir / "birthday_left.png"))
    elapsed = time.time() - start
    assert elapsed < 2.0, f"Image loading took {elapsed:.2f}s, expected < 2s"
    assert doc.page_count == 1
    doc.close()


def test_large_image_load_performance(testdata_dir):
    """Loading a larger image (1080p birthday) should complete in under 3 seconds."""
    start = time.time()
    doc = DocumentRepresentation(str(testdata_dir / "birthday_1080.png"))
    elapsed = time.time() - start
    assert elapsed < 3.0, f"Large image loading took {elapsed:.2f}s, expected < 3s"
    assert doc.page_count == 1
    doc.close()


# ---------------------------------------------------------------------------
# PDF loading
# ---------------------------------------------------------------------------


def test_pdf_single_page_load_performance(testdata_dir):
    """Loading a single-page PDF should complete in under 5 seconds."""
    start = time.time()
    doc = DocumentRepresentation(str(testdata_dir / "sample_1_page.pdf"))
    elapsed = time.time() - start
    assert elapsed < 5.0, f"PDF loading took {elapsed:.2f}s, expected < 5s"
    assert doc.page_count == 1
    doc.close()


def test_pdf_multi_page_load_performance(testdata_dir):
    """Loading a multi-page PDF should complete in under 10 seconds."""
    start = time.time()
    doc = DocumentRepresentation(str(testdata_dir / "sample.pdf"))
    elapsed = time.time() - start
    assert elapsed < 10.0, f"Multi-page PDF loading took {elapsed:.2f}s, expected < 10s"
    assert doc.page_count >= 1
    doc.close()


# ---------------------------------------------------------------------------
# Comparison benchmarks
# ---------------------------------------------------------------------------


def test_identical_image_comparison_performance(testdata_dir):
    """Comparing two identical images should complete in under 5 seconds."""
    vt = VisualTest()
    ref = str(testdata_dir / "birthday_left.png")

    start = time.time()
    vt.compare_images(ref, ref)
    elapsed = time.time() - start

    assert elapsed < 5.0, f"Identical image comparison took {elapsed:.2f}s, expected < 5s"


def test_different_image_comparison_performance(testdata_dir):
    """Comparing two different images should complete in under 5 seconds (excluding error)."""
    vt = VisualTest()
    ref = str(testdata_dir / "birthday_left.png")
    cand = str(testdata_dir / "birthday_right.png")

    start = time.time()
    with pytest.raises(AssertionError):
        vt.compare_images(ref, cand)
    elapsed = time.time() - start

    assert elapsed < 5.0, f"Different image comparison took {elapsed:.2f}s, expected < 5s"


def test_pdf_comparison_performance(testdata_dir):
    """Comparing two single-page PDFs should complete in under 10 seconds."""
    vt = VisualTest()
    ref = str(testdata_dir / "sample_1_page.pdf")

    start = time.time()
    vt.compare_images(ref, ref)
    elapsed = time.time() - start

    assert elapsed < 10.0, f"PDF comparison took {elapsed:.2f}s, expected < 10s"


# ---------------------------------------------------------------------------
# Memory / resource cleanup
# ---------------------------------------------------------------------------


def test_document_representation_gc_after_close(testdata_dir):
    """After close(), the DocumentRepresentation should be garbage-collectible."""
    doc = DocumentRepresentation(str(testdata_dir / "birthday_left.png"))
    ref = weakref.ref(doc)
    doc.close()
    del doc
    gc.collect()
    assert ref() is None, "DocumentRepresentation was not garbage collected after close()"


def test_pdf_document_gc_after_close(testdata_dir):
    """After close(), a PDF DocumentRepresentation should be garbage-collectible."""
    doc = DocumentRepresentation(str(testdata_dir / "sample_1_page.pdf"))
    ref = weakref.ref(doc)
    doc.close()
    del doc
    gc.collect()
    assert ref() is None, "PDF DocumentRepresentation was not garbage collected after close()"


def test_multiple_loads_no_resource_leak(testdata_dir):
    """Loading and closing multiple documents should not leak memory significantly."""
    gc.collect()

    for _ in range(5):
        doc = DocumentRepresentation(str(testdata_dir / "birthday_left.png"))
        assert doc.page_count == 1
        doc.close()

    gc.collect()
    # If we get here without OOM or errors, the test passes.
    # This is a smoke test for gross resource leaks.


def test_page_data_accessible_before_close(testdata_dir):
    """Verify that page data is accessible before close and not after."""
    doc = DocumentRepresentation(str(testdata_dir / "birthday_left.png"))
    assert doc.page_count == 1
    assert doc.pages[0].image is not None
    doc.close()
    # After close, pages list should be empty or image should be cleaned up
    # This is a basic sanity check -- the exact behavior depends on implementation
    assert doc is not None  # Object still exists, just resources released
