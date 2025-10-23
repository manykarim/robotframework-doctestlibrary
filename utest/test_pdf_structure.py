import copy
import os

import pytest

from DocTest.DocumentRepresentation import DocumentRepresentation
from DocTest.PdfStructureComparator import StructureTolerance, compare_document_structures
from DocTest.PdfStructureModels import StructureExtractionConfig
from DocTest.PdfTest import PdfTest


TESTDATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "testdata"))
REFERENCE_PDF = os.path.join(TESTDATA_DIR, "invoice.pdf")
CANDIDATE_PDF = os.path.join(TESTDATA_DIR, "invoice_diff_font.pdf")
DIFFERENT_CONTENT_PDF = os.path.join(TESTDATA_DIR, "invoice_diff_date_id.pdf")


@pytest.fixture(scope="module")
def reference_structure():
    doc = DocumentRepresentation(REFERENCE_PDF)
    return doc.get_pdf_structure(config=StructureExtractionConfig())


@pytest.fixture(scope="module")
def candidate_structure():
    doc = DocumentRepresentation(CANDIDATE_PDF)
    return doc.get_pdf_structure(config=StructureExtractionConfig())


def test_structure_comparison_tolerates_font_change(reference_structure, candidate_structure):
    result = compare_document_structures(
        reference_structure,
        candidate_structure,
        tolerance=StructureTolerance(),
    )
    assert result.passed, f"Unexpected structural differences: {result.summary} {result.page_differences}"


def test_structure_comparison_detects_geometry_shift(reference_structure):
    modified = copy.deepcopy(reference_structure)
    first_line = modified.pages[0].blocks[0].lines[0]
    # Shift the line horizontally to exceed tolerance.
    first_line.bbox = (
        first_line.bbox[0] + 20.0,
        first_line.bbox[1],
        first_line.bbox[2] + 20.0,
        first_line.bbox[3],
    )

    result = compare_document_structures(
        reference_structure,
        modified,
        tolerance=StructureTolerance(position=0.5, size=0.5, relative=0.0),
    )
    assert not result.passed
    all_diffs = [diff for diffs in result.page_differences.values() for diff in diffs]
    assert any(diff.diff_type == "geometry_mismatch" for diff in all_diffs)


def test_compare_pdf_structure_keyword_passes_with_default_tolerance():
    tester = PdfTest()
    tester.compare_pdf_structure(REFERENCE_PDF, CANDIDATE_PDF)


def test_compare_pdf_structure_keyword_raises_for_text_difference():
    tester = PdfTest()
    with pytest.raises(AssertionError):
        tester.compare_pdf_structure(
            REFERENCE_PDF,
            DIFFERENT_CONTENT_PDF,
        )
