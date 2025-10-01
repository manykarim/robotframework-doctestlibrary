from DocTest.PdfTest import PdfTest
import pytest
from pathlib import Path

TESTDATA_ROOT = Path(__file__).resolve().parent / "testdata"

pytestmark = pytest.mark.skipif(
    not (TESTDATA_ROOT / "sample_1_page_different_text.pdf").exists(),
    reason="PDF comparison test data is unavailable in this environment",
)

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

def test_check_text_content_exists_success(testdata_dir):
    pdf_tester = PdfTest()
    pdf_document = testdata_dir / 'sample.pdf'
    pdf_tester.PDF_should_contain_strings('THE TEST SHIPPER', pdf_document)

def test_check_text_content_exists_failure(testdata_dir):
    pdf_tester = PdfTest()
    pdf_document = testdata_dir / 'sample.pdf'
    with pytest.raises(AssertionError):
        pdf_tester.PDF_should_contain_strings('This Text does not exist', pdf_document)

def test_check_text_content_not_exists_success(testdata_dir):
    pdf_tester = PdfTest()
    pdf_document = testdata_dir / 'sample.pdf'
    pdf_tester.PDF_should_not_contain_strings('This Text does not exist', pdf_document)

def test_check_text_content_not_exists_list_failure(testdata_dir):
    pdf_tester = PdfTest()
    pdf_document = testdata_dir / 'sample.pdf'
    with pytest.raises(AssertionError):
        pdf_tester.PDF_should_not_contain_strings(['This Text does not exist', 'THE TEST SHIPPER'], pdf_document)

def test_check_text_content_not_exists_single_failure(testdata_dir):
    pdf_tester = PdfTest()
    pdf_document = testdata_dir / 'sample.pdf'
    with pytest.raises(AssertionError):
        pdf_tester.PDF_should_not_contain_strings('THE TEST SHIPPER', pdf_document)
