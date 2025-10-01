"""Tests for the OCR text correction helpers."""

from DocTest import text_corrector


def test_correct_token_normalizes_date():
    assert text_corrector.correct_token("1O01-Jan-2021") == "01-Jan-2021"


def test_correct_token_fixes_common_word():
    assert text_corrector.correct_token("SOUYENIR") == "SOUVENIR"


def test_correct_token_preserves_serials():
    value = text_corrector.correct_token("ABC1234")
    assert value == "ABC1234"
