"""Unit tests for compare_document_words() -- ADR-001 Word-Level Token Comparison."""

import pytest

from DocTest.PdfStructureComparator import (
    DocumentWordDifference,
    StructureComparisonResult,
    compare_document_words,
)
from DocTest.PdfStructureModels import (
    DocumentStructure,
    PageStructure,
    StructureExtractionConfig,
    TextBlock,
    TextLine,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_doc(*page_texts):
    """Create a DocumentStructure from lists of line texts per page.

    Usage: _make_doc(["line1", "line2"], ["line3"]) creates 2 pages.
    Each positional argument is a list of line-text strings for one page.
    All lines are placed in a single block per page.
    """
    config = StructureExtractionConfig()
    pages = []
    for page_num, lines in enumerate(page_texts):
        text_lines = []
        for i, text in enumerate(lines):
            text_lines.append(
                TextLine(index=i, text=text, bbox=(0.0, 0.0, 100.0, 10.0))
            )
        block = TextBlock(index=0, bbox=(0.0, 0.0, 100.0, 100.0), lines=text_lines)
        page = PageStructure(
            page_number=page_num, width=612.0, height=792.0, blocks=[block]
        )
        pages.append(page)
    return DocumentStructure(pages=pages, config=config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_identical_content_same_lines():
    """Same text, same lines -> passed=True, no word_differences."""
    ref = _make_doc(["the quick brown fox"])
    cand = _make_doc(["the quick brown fox"])
    result = compare_document_words(ref, cand)
    assert result.passed
    assert result.word_differences == []


def test_identical_content_different_lines():
    """Same words split across different lines -> passed=True.

    This is the KEY test for reflow tolerance: the words are identical,
    only the line breaks differ.
    """
    ref = _make_doc(["the quick brown", "fox jumps"])
    cand = _make_doc(["the quick", "brown fox jumps"])
    result = compare_document_words(ref, cand)
    assert result.passed
    assert result.word_differences == []


def test_identical_content_different_pages():
    """Same words on different pages -> passed=True."""
    ref = _make_doc(["hello world"], ["foo bar"])
    cand = _make_doc(["hello world foo bar"])
    result = compare_document_words(ref, cand)
    assert result.passed
    assert result.word_differences == []


def test_single_word_replacement():
    """'fox' vs 'cat' -> one word_mismatch difference."""
    ref = _make_doc(["the quick fox"])
    cand = _make_doc(["the quick cat"])
    result = compare_document_words(ref, cand)
    assert not result.passed
    assert len(result.word_differences) >= 1
    mismatch_diffs = [
        d for d in result.word_differences if d.diff_type == "word_mismatch"
    ]
    assert len(mismatch_diffs) >= 1
    diff = mismatch_diffs[0]
    assert "fox" in diff.ref_words
    assert "cat" in diff.cand_words


def test_single_word_insertion():
    """Candidate has extra word -> one extra_words difference."""
    ref = _make_doc(["the fox"])
    cand = _make_doc(["the quick fox"])
    result = compare_document_words(ref, cand)
    assert not result.passed
    extra_diffs = [
        d for d in result.word_differences if d.diff_type == "extra_words"
    ]
    assert len(extra_diffs) >= 1
    diff = extra_diffs[0]
    assert "quick" in diff.cand_words


def test_single_word_deletion():
    """Candidate missing a word -> one missing_words difference."""
    ref = _make_doc(["the quick fox"])
    cand = _make_doc(["the fox"])
    result = compare_document_words(ref, cand)
    assert not result.passed
    missing_diffs = [
        d for d in result.word_differences if d.diff_type == "missing_words"
    ]
    assert len(missing_diffs) >= 1
    diff = missing_diffs[0]
    assert "quick" in diff.ref_words


def test_multi_word_replacement():
    """Contiguous block of different words -> one grouped mismatch."""
    ref = _make_doc(["the quick brown fox"])
    cand = _make_doc(["the slow red fox"])
    result = compare_document_words(ref, cand)
    assert not result.passed
    mismatch_diffs = [
        d for d in result.word_differences if d.diff_type == "word_mismatch"
    ]
    assert len(mismatch_diffs) >= 1
    # The replaced block should be grouped into a single diff
    diff = mismatch_diffs[0]
    assert diff.ref_words is not None
    assert diff.cand_words is not None
    assert "quick" in diff.ref_words
    assert "brown" in diff.ref_words
    assert "slow" in diff.cand_words
    assert "red" in diff.cand_words


def test_case_sensitive_default():
    """'Hello' vs 'hello' -> mismatch when case_sensitive=True (default)."""
    ref = _make_doc(["Hello World"])
    cand = _make_doc(["hello World"])
    result = compare_document_words(ref, cand)
    assert not result.passed
    assert len(result.word_differences) >= 1


def test_case_insensitive():
    """'Hello' vs 'hello' -> passed=True when case_sensitive=False."""
    ref = _make_doc(["Hello WORLD"])
    cand = _make_doc(["hello world"])
    result = compare_document_words(ref, cand, case_sensitive=False)
    assert result.passed
    assert result.word_differences == []


def test_both_empty_documents():
    """Both empty -> passed=True."""
    ref = _make_doc()
    cand = _make_doc()
    result = compare_document_words(ref, cand)
    assert result.passed
    assert result.word_differences == []


def test_one_empty_one_not():
    """One empty, one with text -> differences reported."""
    ref = _make_doc(["hello world"])
    cand = _make_doc()
    result = compare_document_words(ref, cand)
    assert not result.passed
    assert len(result.word_differences) >= 1


def test_difference_count_includes_word_diffs():
    """result.difference_count() counts word_differences."""
    ref = _make_doc(["the quick fox"])
    cand = _make_doc(["the slow fox"])
    result = compare_document_words(ref, cand)
    assert result.difference_count() >= 1
    assert result.difference_count() >= len(result.word_differences)


def test_word_differences_have_correct_indices():
    """Verify ref_start_index/ref_end_index/cand_start_index/cand_end_index."""
    ref = _make_doc(["a b c d e"])
    cand = _make_doc(["a b x d e"])  # 'c' replaced by 'x'
    result = compare_document_words(ref, cand)
    assert not result.passed
    assert len(result.word_differences) >= 1

    diff = result.word_differences[0]
    # The replaced word 'c' is at index 2 in the reference
    assert diff.ref_start_index is not None
    assert diff.ref_end_index is not None
    assert diff.cand_start_index is not None
    assert diff.cand_end_index is not None
    # 'c' is the 3rd word (index 2), so ref range should be [2, 3)
    assert diff.ref_start_index == 2
    assert diff.ref_end_index == 3
    # 'x' is the 3rd word (index 2), so cand range should be [2, 3)
    assert diff.cand_start_index == 2
    assert diff.cand_end_index == 3


def test_reflow_across_lines_and_pages():
    """Complex reflow scenario: identical words, different line/page breaks.

    Reference:
        page 0: ["The quick brown fox", "jumps over the"]
        page 1: ["lazy dog"]

    Candidate:
        page 0: ["The quick", "brown fox jumps"]
        page 1: ["over the lazy dog"]

    Should pass because the word sequence is identical.
    """
    ref = _make_doc(
        ["The quick brown fox", "jumps over the"],
        ["lazy dog"],
    )
    cand = _make_doc(
        ["The quick", "brown fox jumps"],
        ["over the lazy dog"],
    )
    result = compare_document_words(ref, cand)
    assert result.passed
    assert result.word_differences == []


def test_result_is_structure_comparison_result():
    """compare_document_words returns a StructureComparisonResult."""
    ref = _make_doc(["hello"])
    cand = _make_doc(["hello"])
    result = compare_document_words(ref, cand)
    assert isinstance(result, StructureComparisonResult)


def test_word_difference_has_message():
    """Each DocumentWordDifference has a non-empty message."""
    ref = _make_doc(["hello world"])
    cand = _make_doc(["hello earth"])
    result = compare_document_words(ref, cand)
    assert not result.passed
    for diff in result.word_differences:
        assert isinstance(diff.message, str)
        assert len(diff.message) > 0


def test_empty_ref_nonempty_cand():
    """Empty reference, non-empty candidate -> extra words reported."""
    ref = _make_doc()
    cand = _make_doc(["hello world"])
    result = compare_document_words(ref, cand)
    assert not result.passed
    extra_diffs = [
        d for d in result.word_differences if d.diff_type == "extra_words"
    ]
    assert len(extra_diffs) >= 1
