from DocTest.VisualTest import VisualTest
import pytest
from pathlib import Path


def test_different_watermark_fails(testdata_dir):
    visual_tester = VisualTest()
    ref_image=str(testdata_dir / 'sample_1_page.pdf')
    cand_img=str(testdata_dir / 'sample_1_page_with_watermark.pdf')
    with pytest.raises(Exception):
        visual_tester.compare_images(ref_image, cand_img)

def test_remove_watermark(testdata_dir):
    visual_tester = VisualTest(show_diff=True, watermark_file=str(testdata_dir / 'watermark_confidential.pdf'))
    ref_image=str(testdata_dir / 'sample_1_page.pdf')
    cand_img=str(testdata_dir / 'sample_1_page_with_watermark.pdf')
    visual_tester.compare_images(ref_image, cand_img)

def test_watermark_is_different_and_watermark_file_does_no_match(testdata_dir):
    visual_tester = VisualTest(show_diff=True, watermark_file=str(testdata_dir / 'text_big.png'))
    ref_image=str(testdata_dir / 'sample_1_page.pdf')
    cand_img=str(testdata_dir / 'sample_1_page_with_watermark.pdf')
    with pytest.raises(Exception):
        visual_tester.compare_images(ref_image, cand_img)

def test_watermark_is_invalid(testdata_dir):
    visual_tester = VisualTest(watermark_file=str(testdata_dir / 'non_existing.pdf'))
    ref_image=str(testdata_dir / 'sample_1_page.pdf')
    cand_img=str(testdata_dir / 'sample_1_page_with_watermark.pdf')
    with pytest.raises(Exception):
        visual_tester.compare_images(ref_image, cand_img)

def test_load_folder_as_watermark(testdata_dir):
    visual_tester = VisualTest(watermark_file=str(testdata_dir))
    ref_image=str(testdata_dir / 'sample_1_page.pdf')
    cand_img=str(testdata_dir / 'sample_1_page_with_watermark.pdf')
    visual_tester.compare_images(ref_image, cand_img)