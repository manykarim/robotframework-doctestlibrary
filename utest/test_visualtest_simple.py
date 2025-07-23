"""Simple unit tests for VisualTest module covering basic functionality."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from DocTest.VisualTest import VisualTest


class TestVisualTestBasics:
    """Test basic VisualTest functionality."""

    def test_init_default(self):
        """Test VisualTest initialization with defaults."""
        vt = VisualTest()

        assert vt.dpi == 200
        assert vt.threshold == 0.0
        assert vt.take_screenshots is False  # Default is False
        assert vt.screenshot_format == "jpg"
        assert vt.movement_detection == "template"

    def test_init_custom_parameters(self):
        """Test VisualTest initialization with custom parameters."""
        vt = VisualTest(
            dpi=300, threshold=0.1, take_screenshots=False, movement_detection="orb"
        )

        assert vt.dpi == 300
        assert vt.threshold == 0.1
        assert vt.take_screenshots is False
        assert vt.movement_detection == "orb"

    def test_concatenate_images_safely_basic(self):
        """Test basic image concatenation."""
        vt = VisualTest()

        img1 = np.zeros((100, 50, 3), dtype=np.uint8)
        img2 = np.zeros((100, 80, 3), dtype=np.uint8)

        result = vt.concatenate_images_safely(img1, img2, axis=1)

        assert result.shape == (100, 130, 3)

    def test_get_diff_rectangles_single(self):
        """Test getting difference rectangles from binary image."""
        vt = VisualTest()

        diff = np.zeros((100, 100), dtype=np.uint8)
        diff[20:60, 30:70] = 255

        rectangles = vt.get_diff_rectangles(diff)

        assert len(rectangles) >= 1
        rect = rectangles[0]
        assert "x" in rect
        assert "y" in rect
        assert "width" in rect
        assert "height" in rect

    def test_load_placeholders_dict_input(self):
        """Test _load_placeholders with dictionary."""
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

        assert result == placeholder

    def test_load_placeholders_list_input(self):
        """Test _load_placeholders with list."""
        vt = VisualTest()

        placeholders = [{"page": 1, "type": "area"}, {"page": 2, "type": "coordinates"}]
        result = vt._load_placeholders(placeholders)

        assert result == placeholders

    def test_raise_comparison_failure_default(self):
        """Test _raise_comparison_failure with default message."""
        vt = VisualTest()

        with pytest.raises(AssertionError, match="The compared images are different"):
            vt._raise_comparison_failure()

    def test_raise_comparison_failure_custom(self):
        """Test _raise_comparison_failure with custom message."""
        vt = VisualTest()

        with pytest.raises(AssertionError, match="Custom error"):
            vt._raise_comparison_failure("Custom error")


class TestVisualTestMovementDetection:
    """Test movement detection methods."""

    def test_find_partial_image_position_template(self):
        """Test template matching method selection."""
        vt = VisualTest(movement_detection="template")

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        template = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch.object(
            vt, "find_partial_image_distance_with_matchtemplate"
        ) as mock_method:
            mock_method.return_value = (10, 20, 0.8)

            result = vt.find_partial_image_position(img, template, threshold=0.1)

            assert result == (10, 20, 0.8)
            mock_method.assert_called_once()

    def test_find_partial_image_position_orb(self):
        """Test ORB method selection."""
        vt = VisualTest(movement_detection="orb")

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        template = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch.object(vt, "find_partial_image_distance_with_orb") as mock_method:
            mock_method.return_value = (15, 25, 0.7)

            result = vt.find_partial_image_position(img, template, threshold=0.1)

            assert result == (15, 25, 0.7)
            mock_method.assert_called_once()

    def test_find_partial_image_position_sift(self):
        """Test SIFT method selection."""
        vt = VisualTest(movement_detection="sift")

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        template = np.zeros((50, 50, 3), dtype=np.uint8)

        with patch.object(vt, "find_partial_image_distance_with_sift") as mock_method:
            mock_method.return_value = (5, 15, 0.9)

            result = vt.find_partial_image_position(img, template, threshold=0.1)

            assert result == (5, 15, 0.9)
            mock_method.assert_called_once()


class TestVisualTestScreenshots:
    """Test screenshot functionality."""

    def test_add_screenshot_disabled(self):
        """Test screenshot when disabled."""
        vt = VisualTest(take_screenshots=False)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = vt.add_screenshot_to_log(img, "_test")

        assert result is None

    @patch("cv2.imwrite")
    @patch("os.makedirs")
    def test_add_screenshot_enabled(self, mock_makedirs, mock_imwrite):
        """Test screenshot when enabled."""
        vt = VisualTest(take_screenshots=True)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        vt.add_screenshot_to_log(img, "_test")

        mock_imwrite.assert_called_once()
        # Check that the filename contains the suffix
        args = mock_imwrite.call_args[0]
        assert "_test" in args[0]


class TestVisualTestTextComparison:
    """Test text comparison functionality."""

    def test_compare_text_content_same(self):
        """Test text comparison with same content."""
        vt = VisualTest()

        mock_ref_doc = MagicMock()
        mock_cand_doc = MagicMock()

        mock_ref_doc.get_text_content.return_value = "Hello World"
        mock_cand_doc.get_text_content.return_value = "Hello World"

        # Should not raise exception
        vt._compare_text_content(mock_ref_doc, mock_cand_doc)

    def test_compare_text_content_different(self):
        """Test text comparison with different content."""
        vt = VisualTest()

        mock_ref_doc = MagicMock()
        mock_cand_doc = MagicMock()

        mock_ref_doc.get_text_content.return_value = "Hello World"
        mock_cand_doc.get_text_content.return_value = "Hello Mars"

        with pytest.raises(AssertionError, match="Text content differs"):
            vt._compare_text_content(mock_ref_doc, mock_cand_doc)


class TestVisualTestWatermarks:
    """Test watermark functionality."""

    def test_load_watermarks_empty_result(self):
        """Test load_watermarks with non-existent files."""
        vt = VisualTest()

        with patch("os.path.exists", return_value=False):
            result = vt.load_watermarks(["nonexistent.pdf"])
            assert result == []

    def test_load_watermarks_string_input(self):
        """Test load_watermarks with string input."""
        vt = VisualTest()

        with patch("os.path.isdir", return_value=False):
            with patch("os.path.exists", return_value=False):
                result = vt.load_watermarks("nonexistent.pdf")
                assert result == []


class TestVisualTestKeypoints:
    """Test keypoint detection methods."""

    def test_get_orb_keypoints_success(self):
        """Test successful ORB keypoint detection."""
        vt = VisualTest()

        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("cv2.ORB_create") as mock_orb_create:
            mock_orb = MagicMock()
            mock_orb_create.return_value = mock_orb

            # Return sufficient keypoints (>=4 required for ORB)
            mock_keypoints = [MagicMock() for _ in range(5)]
            mock_descriptors = np.array([[1, 2, 3]] * 5)
            mock_orb.detectAndCompute.side_effect = [
                (mock_keypoints, mock_descriptors),
                (mock_keypoints, mock_descriptors),
            ]

            kp1, desc1, kp2, desc2 = vt.get_orb_keypoints_and_descriptors(img1, img2)

            assert len(kp1) == 5
            assert len(kp2) == 5
            assert desc1.shape == (5, 3)
            assert desc2.shape == (5, 3)

    def test_get_sift_keypoints_success(self):
        """Test successful SIFT keypoint detection."""
        vt = VisualTest()

        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("cv2.SIFT_create") as mock_sift_create:
            mock_sift = MagicMock()
            mock_sift_create.return_value = mock_sift

            # Return sufficient keypoints (>=2 required for SIFT)
            mock_keypoints = [MagicMock() for _ in range(3)]
            mock_descriptors = np.array(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
            )
            mock_sift.detectAndCompute.side_effect = [
                (mock_keypoints, mock_descriptors),
                (mock_keypoints, mock_descriptors),
            ]

            kp1, desc1, kp2, desc2 = vt.get_sift_keypoints_and_descriptors(img1, img2)

            assert len(kp1) == 3
            assert len(kp2) == 3
            assert desc1.shape == (3, 2)
            assert desc2.shape == (3, 2)

    def test_get_sift_keypoints_insufficient(self):
        """Test SIFT with insufficient keypoints."""
        vt = VisualTest()

        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch("cv2.SIFT_create") as mock_sift_create:
            mock_sift = MagicMock()
            mock_sift_create.return_value = mock_sift

            # Return insufficient keypoints (only 1, need 2)
            mock_keypoints = [MagicMock()]
            mock_descriptors = np.array([[1.0, 2.0]], dtype=np.float32)
            mock_sift.detectAndCompute.side_effect = [
                (mock_keypoints, mock_descriptors),
                (mock_keypoints, mock_descriptors),
            ]

            kp1, desc1, kp2, desc2 = vt.get_sift_keypoints_and_descriptors(img1, img2)

            # Should return None when insufficient keypoints
            assert kp1 is None
            assert desc1 is None
            assert kp2 is None
            assert desc2 is None


class TestVisualTestImageProcessing:
    """Test image processing utilities."""

    def test_concatenate_different_heights(self):
        """Test concatenating images with different heights."""
        vt = VisualTest()

        img1 = np.zeros((100, 50, 3), dtype=np.uint8)
        img2 = np.zeros((80, 50, 3), dtype=np.uint8)

        result = vt.concatenate_images_safely(
            img1, img2, axis=1, fill_color=(128, 128, 128)
        )

        # Should pad to max height
        assert result.shape[0] == 100  # max height
        assert result.shape[1] == 100  # combined width
        assert result.shape[2] == 3  # channels

    def test_get_images_with_highlighted_differences(self):
        """Test highlighting differences in images."""
        vt = VisualTest()

        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        diff = np.zeros((100, 100), dtype=np.uint8)

        # Create a small difference area
        diff[30:35, 30:35] = 255

        ref_img, cand_img, contours = vt.get_images_with_highlighted_differences(
            diff, img1, img2
        )

        assert ref_img.shape == img1.shape
        assert cand_img.shape == img2.shape
        assert isinstance(contours, (list, tuple))
