from DocTest.PdfTest import PdfTest
import pytest
from pathlib import Path

def test_compare_pdf_content_with_different_text_with_masks(testdata_dir):
    pdf_tester = PdfTest()
    reference_pdf = testdata_dir / 'sample_1_page.pdf'
    candidate_pdf = testdata_dir / 'sample_1_page_different_text.pdf'
    pdf_tester.compare_pdf_documents(reference_pdf, candidate_pdf, mask=[".*JobID.*", ".*RTM.*"], compare="text")  

def test_compare_pdf_content_with_differences(testdata_dir):
    pdf_tester = PdfTest()
    reference_pdf = testdata_dir / 'sample_1_page.pdf'
    candidate_pdf = testdata_dir / 'sample_1_page_different_text.pdf'
    with pytest.raises(AssertionError):
        pdf_tester.compare_pdf_documents(reference_pdf, candidate_pdf)

def test_compare_pdf_content_with_different_text(testdata_dir):
    pdf_tester = PdfTest()
    reference_pdf = testdata_dir / 'sample_1_page.pdf'
    candidate_pdf = testdata_dir / 'sample_1_page_different_text.pdf'
    with pytest.raises(AssertionError):
        pdf_tester.compare_pdf_documents(reference_pdf, candidate_pdf, compare="text")  

def test_compare_pdf_content_with_different_metadata(testdata_dir):
    pdf_tester = PdfTest()
    reference_pdf = testdata_dir / 'sample_1_page.pdf'
    candidate_pdf = testdata_dir / 'sample_1_page_different_text.pdf'
    with pytest.raises(AssertionError):
        pdf_tester.compare_pdf_documents(reference_pdf, candidate_pdf,  compare="metadata")  

def test_compare_pdf_content_with_different_fonts(testdata_dir):
    pdf_tester = PdfTest()
    reference_pdf = testdata_dir / 'sample_1_page.pdf'
    candidate_pdf = testdata_dir / 'sample_1_page_different_text.pdf'
    with pytest.raises(AssertionError):
        pdf_tester.compare_pdf_documents(reference_pdf, candidate_pdf,  compare="fonts")

def test_compare_pdf_content_with_different_images(testdata_dir):
    pdf_tester = PdfTest()
    reference_pdf = testdata_dir / 'sample_1_page.pdf'
    candidate_pdf = testdata_dir / 'sample_1_page_different_text.pdf'
    with pytest.raises(AssertionError):
        pdf_tester.compare_pdf_documents(reference_pdf, candidate_pdf,  compare="images")
