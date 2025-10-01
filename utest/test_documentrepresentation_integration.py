"""Integration-style tests for DocumentRepresentation."""
from __future__ import annotations

from pathlib import Path

from DocTest.DocumentRepresentation import DocumentRepresentation


def _word_area(word, dpi: int) -> dict:
    x0, y0, x1, y1 = word[:4]
    x = int(x0 * dpi / 72)
    y = int(y0 * dpi / 72)
    width = int((x1 - x0) * dpi / 72)
    height = int((y1 - y0) * dpi / 72)
    margin = 4
    return {
        "x": max(0, x - margin),
        "y": max(0, y - margin),
        "width": width + 2 * margin,
        "height": height + 2 * margin,
    }


def test_get_text_from_area_pdf_uses_pdf_words(testdata_dir: Path):
    doc = DocumentRepresentation(testdata_dir / "sample_1_page.pdf")
    page = doc.pages[0]
    first_word = page.pdf_text_words[0]
    area = _word_area(first_word, page.dpi)

    text = doc.get_text_from_area(area)

    assert page._normalize_token(first_word[4]) in text


def test_get_text_from_area_force_ocr_image(testdata_dir: Path):
    doc = DocumentRepresentation(testdata_dir / "Beach_date.png")
    page = doc.pages[0]
    page.apply_ocr()

    for idx, token in enumerate(page.ocr_text_data["text"]):
        if token and any(ch.isdigit() for ch in token):
            left = page.ocr_text_data["left"][idx]
            top = page.ocr_text_data["top"][idx]
            width = page.ocr_text_data["width"][idx]
            height = page.ocr_text_data["height"][idx]
            area = {"x": left, "y": top, "width": width, "height": height}
            break
    else:
        raise AssertionError("Digit-containing token not found in OCR output")

    text = doc.get_text_from_area(area, force_ocr=True)
    compact = text.replace(" ", "")
    assert any(ch.isdigit() for ch in compact)
