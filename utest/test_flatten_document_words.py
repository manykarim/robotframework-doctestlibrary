"""Unit tests for flatten_document_words() -- ADR-001 Word-Level Token Comparison."""

import pytest

from DocTest.PdfStructureModels import (
    DocumentStructure,
    PageStructure,
    StructureExtractionConfig,
    TextBlock,
    TextLine,
    WordToken,
    flatten_document_words,
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


def test_empty_document():
    """Empty DocumentStructure returns ([], [])."""
    doc = _make_doc()
    words, tokens = flatten_document_words(doc)
    assert words == []
    assert tokens == []


def test_single_line_single_word():
    """One line 'hello' produces ['hello'] and one WordToken."""
    doc = _make_doc(["hello"])
    words, tokens = flatten_document_words(doc)
    assert words == ["hello"]
    assert len(tokens) == 1
    assert tokens[0].text == "hello"


def test_single_line_multiple_words():
    """'hello world' produces ['hello', 'world'] and two WordTokens."""
    doc = _make_doc(["hello world"])
    words, tokens = flatten_document_words(doc)
    assert words == ["hello", "world"]
    assert len(tokens) == 2
    assert tokens[0].text == "hello"
    assert tokens[1].text == "world"


def test_multiple_lines():
    """Two lines 'foo bar' and 'baz' produce ['foo', 'bar', 'baz'] in order."""
    doc = _make_doc(["foo bar", "baz"])
    words, tokens = flatten_document_words(doc)
    assert words == ["foo", "bar", "baz"]
    assert len(tokens) == 3


def test_multiple_pages():
    """Words from page 0 and page 1 are concatenated in order."""
    doc = _make_doc(["alpha beta"], ["gamma"])
    words, tokens = flatten_document_words(doc)
    assert words == ["alpha", "beta", "gamma"]
    assert len(tokens) == 3


def test_empty_lines_skipped():
    """Lines with empty text produce no tokens."""
    doc = _make_doc(["hello", "", "world"])
    words, tokens = flatten_document_words(doc)
    assert words == ["hello", "world"]
    assert len(tokens) == 2


def test_whitespace_only_lines_skipped():
    """Lines with only whitespace produce no tokens (split yields [])."""
    doc = _make_doc(["hello", "   ", "world"])
    words, tokens = flatten_document_words(doc)
    assert words == ["hello", "world"]
    assert len(tokens) == 2


def test_provenance_metadata_correct():
    """source_page, source_line_index, and word_index are correct across pages."""
    doc = _make_doc(["a b"], ["c"])
    words, tokens = flatten_document_words(doc)

    # First page, first line, word 0
    assert tokens[0].text == "a"
    assert tokens[0].source_page == 0
    assert tokens[0].source_line_index == 0
    assert tokens[0].word_index == 0

    # First page, first line, word 1
    assert tokens[1].text == "b"
    assert tokens[1].source_page == 0
    assert tokens[1].source_line_index == 0
    assert tokens[1].word_index == 1

    # Second page, first line, word 2
    assert tokens[2].text == "c"
    assert tokens[2].source_page == 1
    assert tokens[2].source_line_index == 1
    assert tokens[2].word_index == 2


def test_multiple_spaces_normalized():
    """'hello   world' is split to ['hello', 'world'] (str.split normalizes)."""
    doc = _make_doc(["hello   world"])
    words, tokens = flatten_document_words(doc)
    assert words == ["hello", "world"]
    assert len(tokens) == 2


def test_word_index_is_global():
    """word_index is sequential across all pages, blocks, and lines."""
    doc = _make_doc(["a b", "c"], ["d e f"])
    words, tokens = flatten_document_words(doc)
    assert words == ["a", "b", "c", "d", "e", "f"]

    expected_indices = list(range(6))
    actual_indices = [t.word_index for t in tokens]
    assert actual_indices == expected_indices


def test_word_token_is_frozen():
    """WordToken instances are immutable (frozen dataclass)."""
    token = WordToken(text="hello", source_page=0, source_line_index=0, word_index=0)
    with pytest.raises(AttributeError):
        token.text = "changed"


def test_words_and_tokens_have_same_length():
    """The word strings list and tokens list always have the same length."""
    doc = _make_doc(["the quick brown fox", "jumps over"], ["the lazy dog"])
    words, tokens = flatten_document_words(doc)
    assert len(words) == len(tokens)
    for word, token in zip(words, tokens):
        assert word == token.text
