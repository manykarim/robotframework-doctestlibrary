import copy
import os

import pytest

from DocTest.DocumentRepresentation import DocumentRepresentation
from DocTest.PdfStructureComparator import (
    StructureTolerance,
    compare_document_structures,
    compare_document_text_only,
)
from DocTest.PdfStructureModels import (
    DocumentStructure,
    PageStructure,
    StructureExtractionConfig,
    TextBlock,
    TextLine,
    flatten_document_text,
)
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


# -----------------------------------------------------------------------------
# Helper to create test document structures
# -----------------------------------------------------------------------------


def _create_doc(pages_data):
    """Create a DocumentStructure from nested text lists.

    Args:
        pages_data: List of pages, each page is a list of blocks,
                   each block is a list of text strings (one per line).
                   Example: [[["Hello", "World"], ["Foo"]], [["Page 2"]]]
                   creates a doc with 2 pages.

    Returns:
        A DocumentStructure for testing.
    """
    config = StructureExtractionConfig()
    pages = []
    for page_num, blocks_data in enumerate(pages_data, start=1):
        blocks = []
        global_line_index = 0
        for block_idx, lines_data in enumerate(blocks_data):
            lines = []
            for line_text in lines_data:
                lines.append(
                    TextLine(
                        index=global_line_index,
                        text=line_text,
                        bbox=(0.0, float(global_line_index * 20), 100.0, float(global_line_index * 20 + 15)),
                    )
                )
                global_line_index += 1
            blocks.append(
                TextBlock(
                    index=block_idx,
                    bbox=(0.0, 0.0, 100.0, 100.0),
                    lines=lines,
                )
            )
        pages.append(
            PageStructure(
                page_number=page_num,
                width=612.0,
                height=792.0,
                blocks=blocks,
            )
        )
    return DocumentStructure(pages=pages, config=config)


# -----------------------------------------------------------------------------
# Tests for flatten_document_text
# -----------------------------------------------------------------------------


def test_flatten_document_text_single_page():
    """Test flattening a single page document."""
    doc = _create_doc([[["Hello", "World", "Foo"]]])
    texts = flatten_document_text(doc)
    assert texts == ["Hello", "World", "Foo"]


def test_flatten_document_text_multiple_pages():
    """Test flattening preserves order across pages."""
    doc = _create_doc([
        [["Page 1 Line 1", "Page 1 Line 2"]],
        [["Page 2 Line 1"]],
        [["Page 3 Line 1", "Page 3 Line 2"]],
    ])
    texts = flatten_document_text(doc)
    assert texts == [
        "Page 1 Line 1", "Page 1 Line 2",
        "Page 2 Line 1",
        "Page 3 Line 1", "Page 3 Line 2",
    ]


def test_flatten_document_text_multiple_blocks():
    """Test flattening preserves order across blocks."""
    doc = _create_doc([
        [["Block 1 Line 1"], ["Block 2 Line 1", "Block 2 Line 2"]],
    ])
    texts = flatten_document_text(doc)
    assert texts == ["Block 1 Line 1", "Block 2 Line 1", "Block 2 Line 2"]


def test_flatten_document_text_empty_document():
    """Test flattening an empty document."""
    doc = _create_doc([])
    texts = flatten_document_text(doc)
    assert texts == []


# -----------------------------------------------------------------------------
# Tests for compare_document_text_only
# -----------------------------------------------------------------------------


def test_compare_text_only_identical_content_different_pages():
    """Text-only comparison passes when content matches across different page structures."""
    ref = _create_doc([
        [["Hello", "World"]],  # Page 1
        [["Foo", "Bar"]],      # Page 2
    ])
    cand = _create_doc([
        [["Hello", "World", "Foo", "Bar"]],  # All on page 1
    ])

    result = compare_document_text_only(ref, cand)
    assert result.passed
    assert result.difference_count() == 0


def test_compare_text_only_detects_missing_text():
    """Text-only comparison detects missing content."""
    ref = _create_doc([[["Hello", "World", "Foo"]]])
    cand = _create_doc([[["Hello", "World"]]])  # Missing "Foo"

    result = compare_document_text_only(ref, cand)
    assert not result.passed
    assert any(d.diff_type == "missing_text" for d in result.document_differences)
    assert any("Foo" in d.message for d in result.document_differences)


def test_compare_text_only_detects_extra_text():
    """Text-only comparison detects extra content."""
    ref = _create_doc([[["Hello", "World"]]])
    cand = _create_doc([[["Hello", "World", "Extra"]]])

    result = compare_document_text_only(ref, cand)
    assert not result.passed
    assert any(d.diff_type == "extra_text" for d in result.document_differences)
    assert any("Extra" in d.message for d in result.document_differences)


def test_compare_text_only_detects_order_change():
    """Text-only comparison detects content in wrong order."""
    ref = _create_doc([[["A", "B", "C"]]])
    cand = _create_doc([[["A", "C", "B"]]])  # B and C swapped

    result = compare_document_text_only(ref, cand)
    assert not result.passed
    assert result.difference_count() > 0


def test_compare_text_only_detects_text_mismatch():
    """Text-only comparison detects text replacement."""
    ref = _create_doc([[["Hello", "Old Text", "World"]]])
    cand = _create_doc([[["Hello", "New Text", "World"]]])

    result = compare_document_text_only(ref, cand)
    assert not result.passed
    assert any(d.diff_type == "text_mismatch" for d in result.document_differences)


def test_compare_text_only_case_insensitive():
    """Text-only comparison respects case_sensitive parameter."""
    ref = _create_doc([[["Hello", "WORLD"]]])
    cand = _create_doc([[["hello", "world"]]])

    # Case sensitive - should fail
    result = compare_document_text_only(ref, cand, case_sensitive=True)
    assert not result.passed

    # Case insensitive - should pass
    result = compare_document_text_only(ref, cand, case_sensitive=False)
    assert result.passed


def test_compare_text_only_empty_documents():
    """Text-only comparison passes for two empty documents."""
    ref = _create_doc([])
    cand = _create_doc([])

    result = compare_document_text_only(ref, cand)
    assert result.passed


# -----------------------------------------------------------------------------
# Tests for check_geometry flag
# -----------------------------------------------------------------------------


def test_structure_comparison_with_geometry_disabled(reference_structure):
    """Geometry differences ignored when check_geometry=False."""
    modified = copy.deepcopy(reference_structure)
    first_line = modified.pages[0].blocks[0].lines[0]
    # Shift the line significantly
    first_line.bbox = (
        first_line.bbox[0] + 100.0,
        first_line.bbox[1] + 100.0,
        first_line.bbox[2] + 100.0,
        first_line.bbox[3] + 100.0,
    )

    # With geometry check - should fail
    result = compare_document_structures(
        reference_structure,
        modified,
        tolerance=StructureTolerance(position=1.0),
        check_geometry=True,
    )
    assert not result.passed
    all_diffs = [diff for diffs in result.page_differences.values() for diff in diffs]
    assert any(diff.diff_type == "geometry_mismatch" for diff in all_diffs)

    # Without geometry check - should pass
    result = compare_document_structures(
        reference_structure,
        modified,
        tolerance=StructureTolerance(),
        check_geometry=False,
    )
    assert result.passed


# -----------------------------------------------------------------------------
# Tests for check_block_count flag
# -----------------------------------------------------------------------------


def test_structure_comparison_with_block_count_disabled():
    """Block count differences ignored when check_block_count=False."""
    # Reference: 2 blocks
    ref = _create_doc([
        [["Block 1"], ["Block 2"]],
    ])
    # Candidate: 1 block with same text
    cand = _create_doc([
        [["Block 1", "Block 2"]],
    ])

    # With block count check - should report mismatch
    result = compare_document_structures(ref, cand, StructureTolerance(), check_block_count=True)
    all_diffs = [diff for diffs in result.page_differences.values() for diff in diffs]
    assert any(diff.diff_type == "block_count_mismatch" for diff in all_diffs)

    # Without block count check - should not report block mismatch
    result = compare_document_structures(ref, cand, StructureTolerance(), check_block_count=False)
    all_diffs = [diff for diffs in result.page_differences.values() for diff in diffs]
    assert not any(diff.diff_type == "block_count_mismatch" for diff in all_diffs)


def test_structure_comparison_both_checks_disabled():
    """Both geometry and block count checks can be disabled together."""
    ref = _create_doc([
        [["Line 1"], ["Line 2"]],  # 2 blocks
    ])
    cand = _create_doc([
        [["Line 1", "Line 2"]],  # 1 block, same text
    ])

    # Modify bboxes significantly
    for block in cand.pages[0].blocks:
        for line in block.lines:
            line.bbox = (999.0, 999.0, 1099.0, 1019.0)

    # With both checks disabled - should pass (only text matters)
    result = compare_document_structures(
        ref, cand, StructureTolerance(),
        check_geometry=False,
        check_block_count=False,
    )
    assert result.passed


# -----------------------------------------------------------------------------
# Tests for ignore_page_boundaries parameter in keyword
# -----------------------------------------------------------------------------


def test_compare_pdf_structure_keyword_with_check_geometry_disabled():
    """Keyword accepts check_geometry parameter."""
    tester = PdfTest()
    # This should pass even with font differences when geometry check is disabled
    tester.compare_pdf_structure(
        REFERENCE_PDF,
        CANDIDATE_PDF,
        check_geometry=False,
    )


def test_compare_pdf_structure_keyword_with_check_block_count_disabled():
    """Keyword accepts check_block_count parameter."""
    tester = PdfTest()
    # This should pass with block count check disabled
    tester.compare_pdf_structure(
        REFERENCE_PDF,
        CANDIDATE_PDF,
        check_block_count=False,
    )


def test_compare_pdf_structure_keyword_with_ignore_page_boundaries():
    """Keyword accepts ignore_page_boundaries parameter."""
    tester = PdfTest()
    # Same document should pass with ignore_page_boundaries
    tester.compare_pdf_structure(
        REFERENCE_PDF,
        REFERENCE_PDF,
        ignore_page_boundaries=True,
    )


def test_compare_pdf_structure_keyword_ignore_page_boundaries_disables_checks():
    """When ignore_page_boundaries=True, geometry and block checks are disabled."""
    tester = PdfTest()
    # With ignore_page_boundaries, even with explicit check_geometry=True,
    # the checks should be disabled
    tester.compare_pdf_structure(
        REFERENCE_PDF,
        CANDIDATE_PDF,
        ignore_page_boundaries=True,
        check_geometry=True,  # Should be overridden to False
        check_block_count=True,  # Should be overridden to False
    )


# -----------------------------------------------------------------------------
# Tests for difference_count method
# -----------------------------------------------------------------------------


def test_structure_comparison_result_difference_count():
    """Test difference_count method returns correct total."""
    ref = _create_doc([[["A", "B", "C"]]])
    cand = _create_doc([[["A", "X", "C"]]])  # B changed to X

    result = compare_document_structures(ref, cand, StructureTolerance())
    assert result.difference_count() > 0


def test_text_only_comparison_result_difference_count():
    """Test difference_count includes document_differences."""
    ref = _create_doc([[["A", "B"]]])
    cand = _create_doc([[["A"]]])  # Missing B

    result = compare_document_text_only(ref, cand)
    assert result.difference_count() == 1
    assert len(result.document_differences) == 1
