"""Unit tests for word-level normalization features.

Tests cover:
  - merge_split_words() in TextNormalization.py
  - flatten_document_words() with normalize_ligatures_in_words and
    normalize_word_boundaries keyword parameters
  - compare_document_words() with normalize_ligatures and
    normalize_word_boundaries keyword parameters
"""

import pytest

from DocTest.PdfStructureModels import (
    DocumentStructure,
    PageStructure,
    StructureExtractionConfig,
    TextBlock,
    TextLine,
    TextSpan,
    WordToken,
    flatten_document_words,
)
from DocTest.PdfStructureComparator import (
    StructureComparisonResult,
    compare_document_words,
)
from DocTest.TextNormalization import (
    _WORD_BOUNDARY_CONNECTORS,
    merge_split_words,
    normalize_ligatures,
)


# ---------------------------------------------------------------------------
# Helpers
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


def _make_tokens_from_lines(*line_groups):
    """Build parallel word/token lists from line groups.

    Each argument is a tuple of (line_index, page, words_string).
    Words are split on whitespace from the string.  This allows precise
    control over source_line_index for each word.

    Returns (words, tokens) matching the merge_split_words signature.
    """
    words = []
    tokens = []
    global_word_idx = 0
    for line_index, page, text in line_groups:
        for w in text.split():
            words.append(w)
            tokens.append(WordToken(
                text=w,
                source_page=page,
                source_line_index=line_index,
                word_index=global_word_idx,
            ))
            global_word_idx += 1
    return words, tokens


# ===========================================================================
# TestMergeSplitWords
# ===========================================================================


class TestMergeSplitWords:
    """Tests for merge_split_words() in TextNormalization."""

    def test_empty_input(self):
        """Empty lists return empty lists."""
        words, tokens = merge_split_words([], [])
        assert words == []
        assert tokens == []

    def test_single_word(self):
        """Single word returns unchanged."""
        w = ["hello"]
        t = [WordToken(text="hello", source_page=1, source_line_index=0, word_index=0)]
        out_w, out_t = merge_split_words(w, t)
        assert out_w == ["hello"]
        assert len(out_t) == 1
        assert out_t[0].text == "hello"

    def test_no_merge_same_line(self):
        """Two words from same line are NOT merged even if connector present."""
        words, tokens = _make_tokens_from_lines(
            (0, 1, "path/ file"),
        )
        out_w, out_t = merge_split_words(words, tokens)
        assert out_w == ["path/", "file"]
        assert len(out_t) == 2

    def test_merge_slash_connector(self):
        """Words ending with / from different lines get merged."""
        words, tokens = _make_tokens_from_lines(
            (0, 1, "path/to/"),
            (1, 1, "file"),
        )
        out_w, out_t = merge_split_words(words, tokens)
        assert out_w == ["path/to/file"]
        assert len(out_t) == 1

    def test_merge_hyphen_connector(self):
        """Words ending with - from different lines get merged."""
        words, tokens = _make_tokens_from_lines(
            (0, 1, "anti-"),
            (1, 1, "virus"),
        )
        out_w, out_t = merge_split_words(words, tokens)
        assert out_w == ["anti-virus"]
        assert len(out_t) == 1

    def test_merge_backslash_connector(self):
        r"""Words ending with \ from different lines get merged."""
        words, tokens = _make_tokens_from_lines(
            (0, 1, "C:\\Users\\"),
            (1, 1, "name"),
        )
        out_w, out_t = merge_split_words(words, tokens)
        assert out_w == ["C:\\Users\\name"]
        assert len(out_t) == 1

    def test_no_merge_without_connector(self):
        """Words from different lines without connector stay separate."""
        words, tokens = _make_tokens_from_lines(
            (0, 1, "hello"),
            (1, 1, "world"),
        )
        out_w, out_t = merge_split_words(words, tokens)
        assert out_w == ["hello", "world"]
        assert len(out_t) == 2

    def test_multiple_consecutive_merges(self):
        """Chain of merges: a/ + b/ + c each from different lines becomes a/b/c."""
        words, tokens = _make_tokens_from_lines(
            (0, 1, "a/"),
            (1, 1, "b/"),
            (2, 1, "c"),
        )
        out_w, out_t = merge_split_words(words, tokens)
        assert out_w == ["a/b/c"]
        assert len(out_t) == 1

    def test_custom_connectors(self):
        """Custom connectors={"_"} merges only on underscore."""
        words, tokens = _make_tokens_from_lines(
            (0, 1, "foo_"),
            (1, 1, "bar"),
        )
        out_w, out_t = merge_split_words(words, tokens, connectors={"_"})
        assert out_w == ["foo_bar"]
        assert len(out_t) == 1

    def test_custom_connectors_ignores_default(self):
        """Custom connectors={"_"} does NOT merge on default slash."""
        words, tokens = _make_tokens_from_lines(
            (0, 1, "path/"),
            (1, 1, "file"),
        )
        out_w, out_t = merge_split_words(words, tokens, connectors={"_"})
        assert out_w == ["path/", "file"]
        assert len(out_t) == 2

    def test_token_provenance_preserved(self):
        """Merged token keeps first token's source_page and source_line_index."""
        t1 = WordToken(text="JS2/", source_page=2, source_line_index=5, word_index=10)
        t2 = WordToken(text="H8", source_page=2, source_line_index=6, word_index=11)
        out_w, out_t = merge_split_words(["JS2/", "H8"], [t1, t2])
        assert out_w == ["JS2/H8"]
        assert len(out_t) == 1
        assert out_t[0].source_page == 2
        assert out_t[0].source_line_index == 5
        assert out_t[0].word_index == 10
        assert out_t[0].text == "JS2/H8"

    def test_mixed_merge_and_no_merge(self):
        """Some pairs merge while others do not."""
        # Line 0: "start path/"   (line 0)
        # Line 1: "file end"      (line 1)
        # "path/" ends with /, different line -> merge with "file"
        # "start" does not end with connector -> no merge with "path/"
        # "file" (after merge becomes "path/file") does not end with connector -> no merge with "end"
        words, tokens = _make_tokens_from_lines(
            (0, 1, "start path/"),
            (1, 1, "file end"),
        )
        out_w, out_t = merge_split_words(words, tokens)
        assert out_w == ["start", "path/file", "end"]
        assert len(out_t) == 3

    def test_merge_realistic_part_number(self):
        """Realistic example: JS2_D48/F16/ + H8 from different lines."""
        words, tokens = _make_tokens_from_lines(
            (5, 1, "JS2_D48/F16/"),
            (6, 1, "H8"),
        )
        out_w, out_t = merge_split_words(words, tokens)
        assert out_w == ["JS2_D48/F16/H8"]
        assert len(out_t) == 1


# ===========================================================================
# TestFlattenDocumentWordsNormalization
# ===========================================================================


class TestFlattenDocumentWordsNormalization:
    """Tests for flatten_document_words() with new normalization params."""

    def test_default_no_normalization(self):
        """Default params return same result as before (no new normalization)."""
        doc = _make_structure([["hello world", "foo bar"]])
        words, tokens = flatten_document_words(doc)
        assert words == ["hello", "world", "foo", "bar"]
        assert len(tokens) == 4

    def test_ligature_normalization(self):
        """Words with ligatures are normalized when normalize_ligatures_in_words=True."""
        # \ufb01 = fi ligature, \ufb02 = fl ligature
        doc = _make_structure([["\ufb01le on the \ufb02oor"]])
        words, tokens = flatten_document_words(
            doc, normalize_ligatures_in_words=True,
        )
        # "file" and "floor" should appear with ASCII equivalents
        assert "file" in words or "\ufb01le" in words
        # When normalization is enabled, ligatures should be replaced
        assert "\ufb01le" not in words
        assert "\ufb02oor" not in words
        assert "file" in words
        assert "floor" in words

    def test_ligature_normalization_disabled_by_default(self):
        """Ligatures are preserved when normalize_ligatures_in_words is not set."""
        doc = _make_structure([["\ufb01le"]])
        words, tokens = flatten_document_words(doc)
        # Without normalization, the ligature character should be preserved
        assert words == ["\ufb01le"]

    def test_word_boundary_normalization(self):
        """Words split across lines get merged when normalize_word_boundaries=True."""
        doc = _make_structure([["path/to/", "file here"]])
        words, tokens = flatten_document_words(
            doc, normalize_word_boundaries=True,
        )
        assert "path/to/file" in words
        assert "here" in words

    def test_word_boundary_normalization_disabled_by_default(self):
        """Word boundaries are not merged by default."""
        doc = _make_structure([["path/to/", "file here"]])
        words, tokens = flatten_document_words(doc)
        assert "path/to/" in words
        assert "file" in words

    def test_both_normalizations(self):
        """Ligatures normalized AND boundaries merged when both enabled."""
        # Line 1 has a word with ligature ending with connector
        # Line 2 continues the word
        doc = _make_structure([
            ["the \ufb01le-", "name is test"],
        ])
        words, tokens = flatten_document_words(
            doc,
            normalize_ligatures_in_words=True,
            normalize_word_boundaries=True,
        )
        # Ligature should be normalized: \ufb01le- -> file-
        # Then merge across line boundary: file- + name -> file-name
        assert "file-name" in words
        assert "is" in words
        assert "test" in words

    def test_ligature_then_merge_order(self):
        """Ligature normalization happens before merge.

        If a word has a ligature AND ends with a connector, the ligature
        must be resolved first so that the merged result is fully normalized.
        """
        # \ufb01 (fi) + "x/" on line 0, "bar" on line 1
        doc = _make_structure([
            ["\ufb01x/", "bar"],
        ])
        words, tokens = flatten_document_words(
            doc,
            normalize_ligatures_in_words=True,
            normalize_word_boundaries=True,
        )
        # First ligature: \ufb01x/ -> fix/
        # Then merge: fix/ + bar -> fix/bar
        assert "fix/bar" in words

    def test_tokens_length_matches_words(self):
        """After normalization, words and tokens lists still have same length."""
        doc = _make_structure([
            ["path/", "file test"],
        ])
        words, tokens = flatten_document_words(
            doc, normalize_word_boundaries=True,
        )
        assert len(words) == len(tokens)
        for word, token in zip(words, tokens):
            assert word == token.text


# ===========================================================================
# TestCompareDocumentWordsNormalization
# ===========================================================================


class TestCompareDocumentWordsNormalization:
    """Tests for compare_document_words() with normalization params."""

    def test_ligature_diff_without_normalization(self):
        """Ligature vs ASCII text flags differences without normalization."""
        ref = _make_structure([["the \ufb01le is here"]])
        cand = _make_structure([["the file is here"]])
        result = compare_document_words(ref, cand)
        assert not result.passed
        assert len(result.word_differences) >= 1

    def test_ligature_diff_with_normalization(self):
        """Ligature vs ASCII text passes with normalize_ligatures=True."""
        ref = _make_structure([["the \ufb01le is here"]])
        cand = _make_structure([["the file is here"]])
        result = compare_document_words(
            ref, cand, normalize_ligatures=True,
        )
        assert result.passed
        assert result.word_differences == []

    def test_word_boundary_diff_without_normalization(self):
        """Word split across lines vs single line flags differences."""
        # Reference has "path/to/file" split across two lines
        ref = _make_structure([["path/to/", "file here"]])
        # Candidate has it on one line
        cand = _make_structure([["path/to/file here"]])
        result = compare_document_words(ref, cand)
        assert not result.passed
        assert len(result.word_differences) >= 1

    def test_word_boundary_diff_with_normalization(self):
        """Word split across lines passes with normalize_word_boundaries=True."""
        ref = _make_structure([["path/to/", "file here"]])
        cand = _make_structure([["path/to/file here"]])
        result = compare_document_words(
            ref, cand, normalize_word_boundaries=True,
        )
        assert result.passed
        assert result.word_differences == []

    def test_combined_normalization(self):
        """Both ligatures and word boundaries together make comparison pass."""
        # Reference: ligature + split across lines
        ref = _make_structure([
            ["the \ufb01le-", "name is ready"],
        ])
        # Candidate: ASCII + single line
        cand = _make_structure([
            ["the file-name is ready"],
        ])
        result = compare_document_words(
            ref, cand,
            normalize_ligatures=True,
            normalize_word_boundaries=True,
        )
        assert result.passed
        assert result.word_differences == []

    def test_combined_normalization_still_detects_real_diffs(self):
        """Even with both normalizations, actual word differences are detected."""
        ref = _make_structure([["the \ufb01le is ready"]])
        cand = _make_structure([["the file is NOT ready"]])
        result = compare_document_words(
            ref, cand,
            normalize_ligatures=True,
            normalize_word_boundaries=True,
        )
        assert not result.passed
        assert len(result.word_differences) >= 1

    def test_normalization_with_case_insensitive(self):
        """Normalization works together with case_sensitive=False."""
        ref = _make_structure([["The \ufb01le is HERE"]])
        cand = _make_structure([["the file is here"]])
        result = compare_document_words(
            ref, cand,
            case_sensitive=False,
            normalize_ligatures=True,
        )
        assert result.passed
        assert result.word_differences == []

    def test_result_type(self):
        """compare_document_words with normalization returns StructureComparisonResult."""
        ref = _make_structure([["hello"]])
        cand = _make_structure([["hello"]])
        result = compare_document_words(
            ref, cand,
            normalize_ligatures=True,
            normalize_word_boundaries=True,
        )
        assert isinstance(result, StructureComparisonResult)

    def test_word_boundary_across_pages(self):
        """Word boundary normalization works across page boundaries too."""
        # Reference: last word on page 1 ends with /, first word on page 2 continues
        ref = _make_structure([
            ["section/"],
            ["header rest"],
        ])
        cand = _make_structure([
            ["section/header rest"],
        ])
        result = compare_document_words(
            ref, cand, normalize_word_boundaries=True,
        )
        assert result.passed
        assert result.word_differences == []

    def test_multiple_ligatures_in_document(self):
        """Multiple ligature words across the document are all normalized."""
        ref = _make_structure([
            ["\ufb01rst \ufb02oor"],
            ["\ufb03ce work"],
        ])
        cand = _make_structure([
            ["first floor"],
            ["ffice work"],
        ])
        result = compare_document_words(
            ref, cand, normalize_ligatures=True,
        )
        assert result.passed
        assert result.word_differences == []
