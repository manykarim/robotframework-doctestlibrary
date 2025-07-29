"""Additional unit tests for VisualTest module covering specific edge cases."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from DocTest.VisualTest import VisualTest


class TestVisualTestEdgeCases:
    """Test cases for specific edge cases and utility functions."""

    def test_find_partial_image_position_template_method(self):
        """Test find_partial_image_position with template method."""
        vt = VisualTest(movement_detection="template")

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        template = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch.object(
            vt, "find_partial_image_distance_with_matchtemplate"
        ) as mock_template:
            mock_template.return_value = (10, 20, 0.8)

            result = vt.find_partial_image_position(img, template, threshold=0.1)

            assert result == (10, 20, 0.8)
            mock_template.assert_called_once_with(img, template, 0.1)

    def test_find_partial_image_position_orb_method(self):
        """Test find_partial_image_position with ORB method."""
        vt = VisualTest(movement_detection="orb")

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        template = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch.object(vt, "find_partial_image_distance_with_orb") as mock_orb:
            mock_orb.return_value = (15, 25, 0.7)

            result = vt.find_partial_image_position(img, template, threshold=0.1)

            assert result == (15, 25, 0.7)
            mock_orb.assert_called_once_with(img, template)

    def test_find_partial_image_position_sift_method(self):
        """Test find_partial_image_position with SIFT method."""
        vt = VisualTest(movement_detection="sift")

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        template = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch.object(vt, "find_partial_image_distance_with_sift") as mock_sift:
            mock_sift.return_value = (5, 15, 0.9)

            result = vt.find_partial_image_position(img, template, threshold=0.1)

            assert result == (5, 15, 0.9)
            mock_sift.assert_called_once_with(img, template, 0.1)

    def test_find_partial_image_position_unknown_method(self):
        """Test find_partial_image_position with fallback to template method."""
        vt = VisualTest(movement_detection="template")  # Use valid value
        # Simulate unknown method by directly setting attribute
        vt.movement_detection = "unknown"

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        template = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch.object(
            vt, "find_partial_image_distance_with_matchtemplate"
        ) as mock_template:
            mock_template.return_value = (0, 0, 0.5)

            result = vt.find_partial_image_position(img, template, threshold=0.1)

            # Should fall back to template method
            assert result == (0, 0, 0.5)
            mock_template.assert_called_once_with(img, template, 0.1)

    def test_get_orb_keypoints_and_descriptors(self):
        """Test ORB keypoint detection."""
        vt = VisualTest()

        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)

        # Add some features to the images
        img1[30:70, 30:70] = 255
        img2[35:75, 35:75] = 255

        with patch("cv2.ORB_create") as mock_orb_create:
            mock_orb = MagicMock()
            mock_orb_create.return_value = mock_orb
            mock_orb.detectAndCompute.side_effect = [
                (
                    [MagicMock()],
                    np.array([[1, 2, 3]]),
                ),  # img1 keypoints and descriptors
                (
                    [MagicMock()],
                    np.array([[4, 5, 6]]),
                ),  # img2 keypoints and descriptors
            ]

            kp1, desc1, kp2, desc2 = vt.get_orb_keypoints_and_descriptors(img1, img2)

            assert len(kp1) == 1
            assert len(kp2) == 1
            assert desc1.shape == (1, 3)
            assert desc2.shape == (1, 3)

    def test_get_sift_keypoints_and_descriptors(self):
        """Test SIFT keypoint detection."""
        vt = VisualTest()

        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("cv2.SIFT_create") as mock_sift_create:
            mock_sift = MagicMock()
            mock_sift_create.return_value = mock_sift
            mock_sift.detectAndCompute.side_effect = [
                (
                    [MagicMock(), MagicMock()],
                    np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                ),
                (
                    [MagicMock(), MagicMock()],
                    np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
                ),
            ]

            kp1, desc1, kp2, desc2 = vt.get_sift_keypoints_and_descriptors(img1, img2)

            # Check that we get valid results when there are sufficient keypoints
            assert len(kp1) == 2
            assert len(kp2) == 2
            assert desc1.shape == (2, 2)
            assert desc2.shape == (2, 2)

    def test_add_screenshot_to_log_take_screenshots_false(self):
        """Test add_screenshot_to_log when take_screenshots is False."""
        vt = VisualTest(take_screenshots=False)

        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Should not do anything when take_screenshots is False
        result = vt.add_screenshot_to_log(img, "_test")

        assert result is None

    def test_load_placeholders_file_path(self):
        """Test _load_placeholders with file path."""
        vt = VisualTest()

        with patch("os.path.exists", return_value=True):
            with patch(
                "builtins.open", mock_open(read_data='[{"page": 1, "type": "area"}]')
            ):
                with patch("json.load") as mock_json_load:
                    mock_json_load.return_value = [{"page": 1, "type": "area"}]

                    result = vt._load_placeholders("placeholders.json")

                    assert result == [{"page": 1, "type": "area"}]

    def test_load_placeholders_dict(self):
        """Test _load_placeholders with dictionary input."""
        vt = VisualTest()

        placeholder = {
            "page": 1,
            "type": "coordinates",
            "x": 0,
            "y": 0,
            "width": 100,
            "height": 100,
        }
        result = vt._load_placeholders(placeholder)

        # Returns the input as-is since it's not a file path
        assert result == placeholder

    def test_load_placeholders_list(self):
        """Test _load_placeholders with list input."""
        vt = VisualTest()

        placeholders = [{"page": 1, "type": "area"}, {"page": 2, "type": "coordinates"}]
        result = vt._load_placeholders(placeholders)

        assert result == placeholders

    def test_get_images_with_highlighted_differences(self):
        """Test get_images_with_highlighted_differences method."""
        vt = VisualTest()

        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        img1_copy = img1.copy()
        img2_copy = img2.copy()
        diff = np.zeros((100, 100), dtype=np.uint8)
        diff[30:70, 30:70] = 255  # White square indicating difference

        ref_img, cand_img, contours = vt.get_images_with_highlighted_differences(
            diff, img1, img2
        )

        # Images should be returned with differences highlighted
        assert ref_img.shape == img1.shape
        assert cand_img.shape == img2.shape
        assert ref_img.dtype == np.uint8
        assert cand_img.dtype == np.uint8

        # The difference area should be highlighted in red on both images
        assert not np.array_equal(ref_img[30:70, 30:70], img1_copy[30:70, 30:70])
        assert not np.array_equal(cand_img[30:70, 30:70], img2_copy[30:70, 30:70])

    def test_compare_text_content_method(self):
        """Test _compare_text_content method."""
        vt = VisualTest()

        # Mock DocumentRepresentation objects
        ref_doc = MagicMock()
        cand_doc = MagicMock()

        # Same text content
        ref_doc.get_text_content.return_value = "Hello World"
        cand_doc.get_text_content.return_value = "Hello World"

        # Should not raise an exception
        vt._compare_text_content(ref_doc, cand_doc)

        # Different text content
        cand_doc.get_text_content.return_value = "Hello Mars"

        with pytest.raises(AssertionError, match="Text content differs"):
            vt._compare_text_content(ref_doc, cand_doc)

    def test_raise_comparison_failure(self):
        """Test _raise_comparison_failure method."""
        vt = VisualTest()

        with pytest.raises(AssertionError, match="The compared images are different"):
            vt._raise_comparison_failure()

        # Test with custom message
        with pytest.raises(AssertionError, match="Custom error message"):
            vt._raise_comparison_failure("Custom error message")


class TestVisualTestWatermarkCombination:
    """Test cases for combined watermark functionality."""

    def test_load_watermarks_combined_check(self):
        """Test loading and processing multiple watermarks."""
        vt = VisualTest()

        watermarks = ["w1.pdf", "w2.jpg"]

        with patch("os.path.exists", return_value=False):
            # When files don't exist, method should return empty list
            result = vt.load_watermarks(watermarks)
            assert result == []


def mock_open(read_data=""):
    """Helper function to create a mock file open."""
    from unittest.mock import mock_open as orig_mock_open

    return orig_mock_open(read_data=read_data)
