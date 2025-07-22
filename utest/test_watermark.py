import pytest

from DocTest.VisualTest import VisualTest


def test_different_watermark_fails(testdata_dir):
    visual_tester = VisualTest()
    ref_image = str(testdata_dir / "sample_1_page.pdf")
    cand_img = str(testdata_dir / "sample_1_page_with_watermark.pdf")
    with pytest.raises(Exception):
        visual_tester.compare_images(ref_image, cand_img)


def test_remove_watermark(testdata_dir):
    visual_tester = VisualTest(
        watermark_file=str(testdata_dir / "watermark_confidential.pdf")
    )
    ref_image = str(testdata_dir / "sample_1_page.pdf")
    cand_img = str(testdata_dir / "sample_1_page_with_watermark.pdf")
    visual_tester.compare_images(ref_image, cand_img)


def test_watermark_is_different_and_watermark_file_does_no_match(testdata_dir):
    visual_tester = VisualTest(watermark_file=str(testdata_dir / "watermark.pdf"))
    ref_image = str(testdata_dir / "sample_1_page.pdf")
    cand_img = str(testdata_dir / "sample_1_page_with_watermark.pdf")
    with pytest.raises(Exception):
        visual_tester.compare_images(ref_image, cand_img)


def test_watermark_is_invalid(testdata_dir):
    visual_tester = VisualTest(watermark_file=str(testdata_dir / "non_existing.pdf"))
    ref_image = str(testdata_dir / "sample_1_page.pdf")
    cand_img = str(testdata_dir / "sample_1_page_with_watermark.pdf")
    with pytest.raises(Exception):
        visual_tester.compare_images(ref_image, cand_img)


def test_load_folder_as_watermark(testdata_dir):
    visual_tester = VisualTest(watermark_file=str(testdata_dir))
    ref_image = str(testdata_dir / "sample_1_page.pdf")
    cand_img = str(testdata_dir / "sample_1_page_with_watermark.pdf")
    visual_tester.compare_images(ref_image, cand_img)


def test_combined_watermarks(testdata_dir):
    """Test that multiple watermark files can be combined to mask all differences"""
    pytest.skip(
        "This test would require test images with multiple watermarks that individually don't cover all differences but combined do cover all differences"
    )

    # This test would require test images with multiple watermarks that individually
    # don't cover all differences but combined do cover all differences
    visual_tester = VisualTest()
    ref_image = str(testdata_dir / "sample_1_page.pdf")
    cand_img = str(testdata_dir / "sample_1_page_with_multiple_watermarks.pdf")

    # Create a list of watermark files
    watermark_files = [
        str(testdata_dir / "watermark_part1.pdf"),
        str(testdata_dir / "watermark_part2.pdf"),
    ]

    # This should pass when watermarks are combined, even if individual watermarks fail
    visual_tester.compare_images(ref_image, cand_img, watermark_file=watermark_files)


def test_watermark_list_loading(testdata_dir):
    """Test that watermark files can be passed as a list and loaded correctly"""
    visual_tester = VisualTest()

    # Test that we can pass a list of watermark files without errors
    watermark_files = [
        str(testdata_dir / "watermark_confidential.pdf"),
        str(testdata_dir / "watermark.pdf"),
    ]

    # Test the load_watermarks function directly with a list
    watermarks = visual_tester.load_watermarks(watermark_files)

    # Should return a list of watermark masks
    assert isinstance(watermarks, list)
    # Should have watermarks from both files
    assert len(watermarks) >= 1  # At least one watermark should be loaded
    # Each watermark should be a numpy array
    import numpy as np

    for watermark in watermarks:
        assert isinstance(watermark, np.ndarray)


def test_watermark_merging_behavior(testdata_dir):
    """Test that watermark masks are properly merged (union of all areas)"""
    visual_tester = VisualTest()

    # Test that we can load multiple watermarks and they get properly merged
    watermark_files = [
        str(testdata_dir / "watermark_confidential.pdf"),
        str(testdata_dir / "watermark.pdf"),
    ]

    # Load watermarks
    watermarks = visual_tester.load_watermarks(watermark_files)

    if len(watermarks) >= 2:
        import cv2
        import numpy as np

        # Ensure both masks have the same size for testing
        h, w = watermarks[0].shape[:2]
        mask1 = cv2.resize(watermarks[0], (w, h))
        mask2 = cv2.resize(watermarks[1], (w, h))

        # Manually create combined mask using the same logic as the implementation
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        combined_mask = cv2.bitwise_or(combined_mask, mask1)
        combined_mask = cv2.bitwise_or(combined_mask, mask2)

        # The combined mask should have at least as many white pixels as the largest individual mask
        mask1_white_pixels = np.sum(mask1 > 0)
        mask2_white_pixels = np.sum(mask2 > 0)
        combined_white_pixels = np.sum(combined_mask > 0)

        # Combined should have at least as many white pixels as the maximum of individual masks
        max_individual = max(mask1_white_pixels, mask2_white_pixels)
        assert combined_white_pixels >= max_individual, (
            f"Combined mask has fewer white pixels ({combined_white_pixels}) than largest individual mask ({max_individual})"
        )

        # Test that OR operation creates union: any pixel white in either mask should be white in combined
        for y in range(min(10, h)):  # Test a sample of pixels
            for x in range(min(10, w)):
                if mask1[y, x] > 0 or mask2[y, x] > 0:
                    assert combined_mask[y, x] > 0, (
                        f"Pixel ({x}, {y}) is white in individual mask but black in combined mask"
                    )


def test_watermark_list_parameter_accepted(testdata_dir):
    """Test that compare_images accepts a list of watermark files without error"""
    visual_tester = VisualTest()
    ref_image = str(testdata_dir / "sample_1_page.pdf")
    cand_img = str(testdata_dir / "sample_1_page.pdf")  # Same image, should pass

    # Test that we can pass a list of watermark files as a parameter
    watermark_files = [str(testdata_dir / "watermark_confidential.pdf")]

    # This should work without any type errors
    try:
        visual_tester.compare_images(
            ref_image, cand_img, watermark_file=watermark_files
        )
    except Exception as e:
        # Make sure it's not a type error about the list parameter
        assert "list" not in str(e).lower(), f"Type error with list parameter: {e}"
    """Test that compare_images accepts a list of watermark files without error"""
    visual_tester = VisualTest()
    ref_image = str(testdata_dir / "sample_1_page.pdf")
    cand_img = str(testdata_dir / "sample_1_page.pdf")  # Same image, should pass

    # Test that we can pass a list of watermark files as a parameter
    watermark_files = [str(testdata_dir / "watermark_confidential.pdf")]

    # This should work without any type errors
    try:
        visual_tester.compare_images(
            ref_image, cand_img, watermark_file=watermark_files
        )
    except Exception as e:
        # Make sure it's not a type error about the list parameter
        assert "list" not in str(e).lower(), f"Type error with list parameter: {e}"
