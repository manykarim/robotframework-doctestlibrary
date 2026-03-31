"""Unit tests for unordered (bag-of-words) comparison mode in compare_document_words().

Tests cover:
  - Basic unordered comparison: identical, reordered, missing, extra words
  - Interaction with normalization flags (ligatures, word boundaries, case)
  - Reporting: diff_type values, ref_words/cand_words content, message labels
  - Side-by-side ordered vs unordered behaviour differences
"""

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
    TextSpan,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_structure(pages_data):
    """Build a DocumentStructure from simplified data.

    pages_data: list of list of strings. Each outer list is a page,
    each string is a line of text.
    """
    pages = []
    for page_num, lines in enumerate(pages_data, 1):
        text_lines = []
        for idx, text in enumerate(lines):
            text_lines.append(TextLine(
                index=idx,
                text=text,
                bbox=(0.0, float(idx * 12), 100.0, float(idx * 12 + 12)),
                fonts=set(),
                spans=[TextSpan(text=text, font="Arial", size=12.0)],
            ))
        block = TextBlock(
            index=0,
            bbox=(0.0, 0.0, 100.0, float(len(lines) * 12)),
            lines=text_lines,
        )
        pages.append(PageStructure(
            page_number=page_num,
            width=612.0,
            height=792.0,
            blocks=[block],
        ))
    return DocumentStructure(pages=pages, config=StructureExtractionConfig())


# ===========================================================================
# TestUnorderedComparisonBasic
# ===========================================================================


class TestUnorderedComparisonBasic:
    """Core unordered comparison behaviour."""

    def test_identical_documents_pass(self):
        """Identical content in the same order passes with zero differences."""
        ref = _make_structure([["the quick brown fox"]])
        cand = _make_structure([["the quick brown fox"]])
        result = compare_document_words(ref, cand, compare_order="unordered")
        assert result.passed
        assert result.word_differences == []

    def test_reordered_lines_pass(self):
        """Lines in different order, same words -- passes unordered, fails ordered."""
        ref = _make_structure([["alpha beta", "gamma delta"]])
        cand = _make_structure([["gamma delta", "alpha beta"]])

        unordered = compare_document_words(ref, cand, compare_order="unordered")
        assert unordered.passed, "Unordered mode should pass when words are reordered"
        assert unordered.word_differences == []

        ordered = compare_document_words(ref, cand, compare_order="ordered")
        assert not ordered.passed, "Ordered mode should fail when lines are swapped"
        assert len(ordered.word_differences) >= 1

    def test_reordered_across_pages_pass(self):
        """Words shifted to different pages still pass in unordered mode."""
        ref = _make_structure([["hello world"], ["foo bar"]])
        cand = _make_structure([["foo bar"], ["hello world"]])

        result = compare_document_words(ref, cand, compare_order="unordered")
        assert result.passed
        assert result.word_differences == []

    def test_missing_word_detected(self):
        """Reference has a word candidate doesn't -> diff_type='missing_words'."""
        ref = _make_structure([["alpha beta gamma"]])
        cand = _make_structure([["alpha gamma"]])

        result = compare_document_words(ref, cand, compare_order="unordered")
        assert not result.passed
        missing = [d for d in result.word_differences if d.diff_type == "missing_words"]
        assert len(missing) == 1
        assert "beta" in missing[0].ref_words

    def test_extra_word_detected(self):
        """Candidate has a word reference doesn't -> diff_type='extra_words'."""
        ref = _make_structure([["alpha gamma"]])
        cand = _make_structure([["alpha beta gamma"]])

        result = compare_document_words(ref, cand, compare_order="unordered")
        assert not result.passed
        extra = [d for d in result.word_differences if d.diff_type == "extra_words"]
        assert len(extra) == 1
        assert "beta" in extra[0].cand_words

    def test_both_missing_and_extra(self):
        """Ref has 'foo' not in cand; cand has 'bar' not in ref -> 2 diffs."""
        ref = _make_structure([["alpha foo gamma"]])
        cand = _make_structure([["alpha bar gamma"]])

        result = compare_document_words(ref, cand, compare_order="unordered")
        assert not result.passed

        missing = [d for d in result.word_differences if d.diff_type == "missing_words"]
        extra = [d for d in result.word_differences if d.diff_type == "extra_words"]
        assert len(missing) == 1
        assert len(extra) == 1
        assert "foo" in missing[0].ref_words
        assert "bar" in extra[0].cand_words

    def test_duplicate_word_count_matters(self):
        """Ref has 'hello hello', cand has 'hello' -> missing_words=['hello']."""
        ref = _make_structure([["hello hello"]])
        cand = _make_structure([["hello"]])

        result = compare_document_words(ref, cand, compare_order="unordered")
        assert not result.passed
        missing = [d for d in result.word_differences if d.diff_type == "missing_words"]
        assert len(missing) == 1
        assert missing[0].ref_words == ["hello"]

    def test_empty_documents_pass(self):
        """Both documents empty -> passed=True, no differences."""
        ref = _make_structure([])
        cand = _make_structure([])

        result = compare_document_words(ref, cand, compare_order="unordered")
        assert result.passed
        assert result.word_differences == []

    def test_default_is_ordered(self):
        """Without compare_order param, uses ordered comparison (same as before).

        Swapped lines should fail under ordered mode (the default).
        """
        ref = _make_structure([["alpha beta", "gamma delta"]])
        cand = _make_structure([["gamma delta", "alpha beta"]])

        result = compare_document_words(ref, cand)
        assert not result.passed, "Default (ordered) mode should fail on reordered text"


# ===========================================================================
# TestUnorderedWithNormalization
# ===========================================================================


class TestUnorderedWithNormalization:
    """Unordered mode combined with normalization flags."""

    def test_unordered_with_ligature_normalization(self):
        """Ligature in ref, ASCII in cand -> passes with normalize_ligatures + unordered."""
        # \ufb01 = fi ligature
        ref = _make_structure([["the certi\ufb01cates are valid"]])
        cand = _make_structure([["the certificates are valid"]])

        result = compare_document_words(
            ref, cand,
            compare_order="unordered",
            normalize_ligatures=True,
        )
        assert result.passed
        assert result.word_differences == []

    def test_unordered_with_word_boundary_normalization(self):
        """Words split across lines merged before unordered comparison."""
        # ref splits "path/file" across lines; cand has it on one line
        ref = _make_structure([["path/", "file here"]])
        cand = _make_structure([["path/file here"]])

        result = compare_document_words(
            ref, cand,
            compare_order="unordered",
            normalize_word_boundaries=True,
        )
        assert result.passed
        assert result.word_differences == []

    def test_unordered_with_case_insensitive(self):
        """Case mismatch passes with case_sensitive=False + unordered."""
        ref = _make_structure([["Hello World"]])
        cand = _make_structure([["hello world"]])

        result = compare_document_words(
            ref, cand,
            compare_order="unordered",
            case_sensitive=False,
        )
        assert result.passed
        assert result.word_differences == []

    def test_unordered_combined_normalizations(self):
        """All normalization options together with unordered mode.

        Ref has ligature + split word + different case; cand has ASCII on one line.
        Reordered across pages to exercise unordered logic.
        """
        # Page 1: ligature word split across lines with different case
        # Page 2: normal text
        ref = _make_structure([
            ["the \ufb01le-", "name is ready"],
            ["HELLO world"],
        ])
        # Candidate: same content, different order, normalized forms
        cand = _make_structure([
            ["hello world"],
            ["the file-name is ready"],
        ])

        result = compare_document_words(
            ref, cand,
            compare_order="unordered",
            case_sensitive=False,
            normalize_ligatures=True,
            normalize_word_boundaries=True,
        )
        assert result.passed
        assert result.word_differences == []


# ===========================================================================
# TestUnorderedReporting
# ===========================================================================


class TestUnorderedReporting:
    """Verify the content and shape of difference reports in unordered mode."""

    def test_missing_words_report_content(self):
        """ref_words contains the actual excess words from the reference."""
        ref = _make_structure([["apple banana cherry"]])
        cand = _make_structure([["apple cherry"]])

        result = compare_document_words(ref, cand, compare_order="unordered")
        missing = [d for d in result.word_differences if d.diff_type == "missing_words"]
        assert len(missing) == 1
        assert missing[0].ref_words is not None
        assert "banana" in missing[0].ref_words

    def test_extra_words_report_content(self):
        """cand_words contains the actual excess words from the candidate."""
        ref = _make_structure([["apple cherry"]])
        cand = _make_structure([["apple banana cherry"]])

        result = compare_document_words(ref, cand, compare_order="unordered")
        extra = [d for d in result.word_differences if d.diff_type == "extra_words"]
        assert len(extra) == 1
        assert extra[0].cand_words is not None
        assert "banana" in extra[0].cand_words

    def test_message_contains_unordered_label(self):
        """The message string should contain 'unordered' to clarify the mode."""
        ref = _make_structure([["hello world"]])
        cand = _make_structure([["hello"]])

        result = compare_document_words(ref, cand, compare_order="unordered")
        assert len(result.word_differences) >= 1
        for diff in result.word_differences:
            assert "unordered" in diff.message.lower(), (
                f"Expected 'unordered' in message, got: {diff.message!r}"
            )

    def test_result_type_is_structure_comparison_result(self):
        """Return type is always StructureComparisonResult."""
        ref = _make_structure([["hello"]])
        cand = _make_structure([["hello"]])

        result = compare_document_words(ref, cand, compare_order="unordered")
        assert isinstance(result, StructureComparisonResult)


# ===========================================================================
# TestUnorderedVsOrdered
# ===========================================================================


class TestUnorderedVsOrdered:
    """Side-by-side comparison of ordered and unordered modes."""

    def test_reorder_fails_ordered_passes_unordered(self):
        """Direct comparison: ordered fails, unordered passes for reordered content."""
        ref = _make_structure([["one two three four"]])
        cand = _make_structure([["four three two one"]])

        ordered_result = compare_document_words(ref, cand, compare_order="ordered")
        unordered_result = compare_document_words(ref, cand, compare_order="unordered")

        assert not ordered_result.passed, "Ordered should fail on reversed word order"
        assert len(ordered_result.word_differences) >= 1

        assert unordered_result.passed, "Unordered should pass when words are the same"
        assert unordered_result.word_differences == []

    def test_genuine_diff_fails_both_modes(self):
        """Genuinely different content fails in both ordered and unordered modes."""
        ref = _make_structure([["the quick brown fox"]])
        cand = _make_structure([["the slow red cat"]])

        ordered_result = compare_document_words(ref, cand, compare_order="ordered")
        unordered_result = compare_document_words(ref, cand, compare_order="unordered")

        assert not ordered_result.passed
        assert not unordered_result.passed
        assert len(ordered_result.word_differences) >= 1
        assert len(unordered_result.word_differences) >= 1

    def test_word_count_preservation(self):
        """Same words but different frequency fails in both modes."""
        ref = _make_structure([["hello hello world"]])
        cand = _make_structure([["hello world world"]])

        ordered_result = compare_document_words(ref, cand, compare_order="ordered")
        unordered_result = compare_document_words(ref, cand, compare_order="unordered")

        assert not ordered_result.passed, "Ordered should detect the word change"
        assert not unordered_result.passed, "Unordered should detect frequency mismatch"

        # In unordered mode, we expect both missing and extra diffs
        missing = [d for d in unordered_result.word_differences
                   if d.diff_type == "missing_words"]
        extra = [d for d in unordered_result.word_differences
                 if d.diff_type == "extra_words"]
        assert len(missing) == 1
        assert len(extra) == 1
        assert "hello" in missing[0].ref_words
        assert "world" in extra[0].cand_words
