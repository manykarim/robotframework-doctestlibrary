import fitz
import pytest

from DocTest.DocumentRepresentation import DocumentRepresentation
from DocTest.VisualTest import VisualTest


def _create_multipage_pdf(tmp_path, pages=3):
    doc = fitz.open()
    for idx in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {idx + 1}")
    pdf_path = tmp_path / "streaming_test.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_document_representation_iterates_streamed_pages(tmp_path):
    pdf_path = _create_multipage_pdf(tmp_path, pages=3)
    doc = DocumentRepresentation(
        str(pdf_path),
        stream_pages=True,
        page_cache_size=1,
    )
    try:
        assert doc.page_count == 3
        assert doc.pages == []
        seen = [page.page_number for page in doc.iter_pages()]
        assert seen == [1, 2, 3]
    finally:
        doc.close()


def test_visualtest_compare_images_streaming(tmp_path):
    pdf_path = _create_multipage_pdf(tmp_path, pages=2)
    visual = VisualTest(stream_documents=True, document_page_cache_size=1)
    visual.compare_images(str(pdf_path), str(pdf_path))
