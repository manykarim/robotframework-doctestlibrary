import shutil

import pytest

from DocTest.PdfTest import PdfTest
from DocTest.ReferencePromotion import promote_candidate_to_reference
from DocTest.VisualTest import VisualTest


def _bytes(path):
    with open(path, 'rb') as f:
        return f.read()


def test_promote_creates_parent_dirs(tmp_path, testdata_dir):
    candidate = testdata_dir / 'birthday_1080.png'
    target = tmp_path / 'nested' / 'dir' / 'reference.png'
    promote_candidate_to_reference(str(target), str(candidate))
    assert _bytes(target) == _bytes(candidate)


def test_promote_missing_candidate_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        promote_candidate_to_reference(str(tmp_path / 'ref.png'), str(tmp_path / 'missing.png'))


def test_promote_same_path_is_noop(testdata_dir):
    candidate = str(testdata_dir / 'birthday_1080.png')
    before = _bytes(candidate)
    promote_candidate_to_reference(candidate, candidate)
    assert _bytes(candidate) == before


def test_visual_missing_reference_is_created_in_reference_run(tmp_path, testdata_dir):
    visual_tester = VisualTest()
    visual_tester.set_reference_run(True)
    reference = tmp_path / 'reference.png'
    candidate = testdata_dir / 'birthday_1080.png'
    visual_tester.compare_images(str(reference), str(candidate))
    assert _bytes(reference) == _bytes(candidate)


def test_visual_differing_candidate_replaces_reference_in_reference_run(tmp_path, testdata_dir):
    visual_tester = VisualTest()
    visual_tester.set_reference_run(True)
    reference = tmp_path / 'reference.png'
    shutil.copyfile(testdata_dir / 'birthday_1080.png', reference)
    candidate = testdata_dir / 'birthday_1080_date_id.png'
    visual_tester.compare_images(str(reference), str(candidate))
    assert _bytes(reference) == _bytes(candidate)


def test_visual_reference_run_off_still_fails(tmp_path, testdata_dir):
    visual_tester = VisualTest()
    reference = tmp_path / 'reference.png'
    shutil.copyfile(testdata_dir / 'birthday_1080.png', reference)
    candidate = testdata_dir / 'birthday_1080_date_id.png'
    with pytest.raises(AssertionError, match='The compared images are different.'):
        visual_tester.compare_images(str(reference), str(candidate))
    assert _bytes(reference) != _bytes(candidate)


def test_visual_reference_run_off_missing_reference_still_fails(tmp_path, testdata_dir):
    visual_tester = VisualTest()
    with pytest.raises(Exception):
        visual_tester.compare_images(
            str(tmp_path / 'missing.png'), str(testdata_dir / 'birthday_1080.png'))


def test_pdf_missing_reference_is_created_in_reference_run(tmp_path, testdata_dir):
    pdf_tester = PdfTest()
    pdf_tester.set_reference_run(True)
    reference = tmp_path / 'reference.pdf'
    candidate = testdata_dir / 'sample_1_page.pdf'
    pdf_tester.compare_pdf_documents(str(reference), str(candidate))
    assert _bytes(reference) == _bytes(candidate)


def test_pdf_differing_candidate_replaces_reference_in_reference_run(tmp_path, testdata_dir):
    pdf_tester = PdfTest()
    pdf_tester.set_reference_run(True)
    reference = tmp_path / 'reference.pdf'
    shutil.copyfile(testdata_dir / 'sample_1_page.pdf', reference)
    candidate = testdata_dir / 'sample_1_page_different_text.pdf'
    pdf_tester.compare_pdf_documents(str(reference), str(candidate))
    assert _bytes(reference) == _bytes(candidate)


def test_pdf_reference_run_off_still_fails(tmp_path, testdata_dir):
    pdf_tester = PdfTest()
    reference = tmp_path / 'reference.pdf'
    shutil.copyfile(testdata_dir / 'sample_1_page.pdf', reference)
    candidate = testdata_dir / 'sample_1_page_different_text.pdf'
    with pytest.raises(AssertionError):
        pdf_tester.compare_pdf_documents(str(reference), str(candidate))
    assert _bytes(reference) != _bytes(candidate)
