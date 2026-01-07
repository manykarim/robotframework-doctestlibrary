"""Unit tests for TextNormalization module."""

import pytest

from DocTest.TextNormalization import (
    apply_character_replacements,
    normalize_ligatures,
)


class TestNormalizeLigatures:
    """Test cases for normalize_ligatures function."""

    def test_normalize_ligatures_empty_string(self):
        """Test that empty string returns empty string."""
        assert normalize_ligatures("") == ""

    def test_normalize_ligatures_none(self):
        """Test that None returns empty string."""
        assert normalize_ligatures(None) == ""

    def test_normalize_ligatures_no_ligatures(self):
        """Test that text without ligatures is unchanged."""
        text = "hello world"
        assert normalize_ligatures(text) == text

    def test_normalize_ligatures_fi_ligature(self):
        """Test that fi ligature is replaced."""
        assert normalize_ligatures("\ufb01le") == "file"

    def test_normalize_ligatures_multiple(self):
        """Test multiple ligatures in one string."""
        text = "\ufb01le with \ufb02oor"
        assert normalize_ligatures(text) == "file with floor"


class TestApplyCharacterReplacements:
    """Test cases for apply_character_replacements function."""

    def test_empty_text_returns_empty(self):
        """Test that empty string returns empty string."""
        assert apply_character_replacements("", {"\u00A0": " "}) == ""

    def test_none_text_returns_empty(self):
        """Test that None returns empty string."""
        assert apply_character_replacements(None, {"\u00A0": " "}) == ""

    def test_none_replacements_returns_original(self):
        """Test that None replacements returns original text."""
        text = "hello world"
        assert apply_character_replacements(text, None) == text

    def test_empty_replacements_returns_original(self):
        """Test that empty replacements dict returns original text."""
        text = "hello world"
        assert apply_character_replacements(text, {}) == text

    def test_nbsp_to_space(self):
        """Test non-breaking space replacement."""
        text = "hello\u00A0world"
        result = apply_character_replacements(text, {"\u00A0": " "})
        assert result == "hello world"

    def test_multiple_nbsp_replacements(self):
        """Test multiple non-breaking spaces are replaced."""
        text = "one\u00A0two\u00A0three"
        result = apply_character_replacements(text, {"\u00A0": " "})
        assert result == "one two three"

    def test_en_dash_to_hyphen(self):
        """Test en-dash replacement."""
        text = "2020\u20132021"
        result = apply_character_replacements(text, {"\u2013": "-"})
        assert result == "2020-2021"

    def test_em_dash_to_hyphen(self):
        """Test em-dash replacement."""
        text = "word\u2014another"
        result = apply_character_replacements(text, {"\u2014": "-"})
        assert result == "word-another"

    def test_multiple_replacements(self):
        """Test multiple different replacements in one call."""
        text = "hello\u00A0world\u2013test"
        replacements = {"\u00A0": " ", "\u2013": "-"}
        result = apply_character_replacements(text, replacements)
        assert result == "hello world-test"

    def test_character_removal(self):
        """Test replacing character with empty string removes it."""
        text = "hello\u200Bworld"  # zero-width space
        result = apply_character_replacements(text, {"\u200B": ""})
        assert result == "helloworld"

    def test_multichar_replacement(self):
        """Test replacing multi-character strings."""
        text = "hello...world"
        result = apply_character_replacements(text, {"...": "…"})
        assert result == "hello…world"

    def test_no_match_returns_original(self):
        """Test that text without matching chars is unchanged."""
        text = "hello world"
        result = apply_character_replacements(text, {"\u00A0": " "})
        assert result == text

    def test_unicode_in_replacement_value(self):
        """Test that replacement value can be unicode."""
        text = "hello world"
        result = apply_character_replacements(text, {" ": "\u00A0"})
        assert result == "hello\u00A0world"

    def test_replacement_order_independence(self):
        """Test that multiple replacements don't interfere."""
        text = "a\u00A0b\u2013c"
        # Order shouldn't matter since we're replacing different chars
        result1 = apply_character_replacements(text, {"\u00A0": " ", "\u2013": "-"})
        result2 = apply_character_replacements(text, {"\u2013": "-", "\u00A0": " "})
        assert result1 == result2 == "a b-c"

    def test_common_pdf_whitespace_characters(self):
        """Test common problematic whitespace characters from PDFs."""
        replacements = {
            "\u00A0": " ",  # NO-BREAK SPACE
            "\u2002": " ",  # EN SPACE
            "\u2003": " ",  # EM SPACE
            "\u2009": " ",  # THIN SPACE
        }
        text = "a\u00A0b\u2002c\u2003d\u2009e"
        result = apply_character_replacements(text, replacements)
        assert result == "a b c d e"

    def test_common_pdf_dash_characters(self):
        """Test common problematic dash characters from PDFs."""
        replacements = {
            "\u2010": "-",  # HYPHEN
            "\u2011": "-",  # NON-BREAKING HYPHEN
            "\u2012": "-",  # FIGURE DASH
            "\u2013": "-",  # EN DASH
            "\u2014": "-",  # EM DASH
        }
        text = "a\u2010b\u2011c\u2012d\u2013e\u2014f"
        result = apply_character_replacements(text, replacements)
        assert result == "a-b-c-d-e-f"
