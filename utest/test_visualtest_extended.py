"""Extended unit tests for VisualTest module to improve coverage."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from DocTest.VisualTest import VisualTest


class TestVisualTestInitialization:
    """Test cases for VisualTest initialization and configuration."""

    def test_init_default_values(self):
        """Test VisualTest initialization with default values."""
        vt = VisualTest()

        assert vt.threshold == 0.0
        assert vt.dpi == 200
        assert vt.take_screenshots is False
        assert vt.show_diff is False
        assert vt.ocr_engine == "ocrs"
        assert vt.screenshot_format == "jpg"
        assert vt.embed_screenshots is False
        assert vt.force_ocr is False
        assert vt.watermark_file is None
        assert vt.movement_detection == "template"

    def test_init_custom_values(self):
        """Test VisualTest initialization with custom values."""
        vt = VisualTest(
            threshold=0.5,
            dpi=300,
            take_screenshots=True,
            show_diff=True,
            ocr_engine="east",
            screenshot_format="png",
            embed_screenshots=True,
            force_ocr=True,
            watermark_file="test.pdf",
            movement_detection="orb",
            partial_image_threshold=0.2,
            sift_ratio_threshold=0.8,
            sift_min_matches=5,
            orb_max_matches=20,
            orb_min_matches=5,
            ransac_threshold=10.0,
        )

        assert vt.threshold == 0.5
        assert vt.dpi == 300
        assert vt.take_screenshots is True
        assert vt.show_diff is True
        assert vt.ocr_engine == "east"
        assert vt.screenshot_format == "png"
        assert vt.embed_screenshots is True
        assert vt.force_ocr is True
        assert vt.watermark_file == "test.pdf"
        assert vt.movement_detection == "orb"
        assert vt.partial_image_threshold == 0.2
        assert vt.sift_ratio_threshold == 0.8
        assert vt.sift_min_matches == 5
        assert vt.orb_max_matches == 20
        assert vt.orb_min_matches == 5
        assert vt.ransac_threshold == 10.0

    def test_init_invalid_screenshot_format(self):
        """Test VisualTest initialization with invalid screenshot format defaults to jpg."""
        vt = VisualTest(screenshot_format="invalid")
        assert vt.screenshot_format == "jpg"

    @patch("DocTest.VisualTest.BuiltIn")
    def test_init_robot_framework_variables(self, mock_builtin):
        """Test initialization with Robot Framework variables."""
        mock_builtin_instance = MagicMock()
        mock_builtin.return_value = mock_builtin_instance
        mock_builtin_instance.get_variable_value.side_effect = [
            "/output/dir",  # OUTPUT_DIR
            True,  # REFERENCE_RUN
            "1",  # PABOTQUEUEINDEX
        ]

        vt = VisualTest()

        assert vt.output_directory == "/output/dir"
        assert vt.reference_run is True
        assert vt.PABOTQUEUEINDEX == "1"


class TestVisualTestSetterMethods:
    """Test cases for VisualTest setter methods."""

    def test_set_ocr_engine(self):
        """Test set_ocr_engine method."""
        vt = VisualTest()

        vt.set_ocr_engine("east")
        assert vt.ocr_engine == "east"

        vt.set_ocr_engine("tesseract")
        assert vt.ocr_engine == "tesseract"

    def test_set_dpi(self):
        """Test set_dpi method."""
        vt = VisualTest()

        vt.set_dpi(300)
        assert vt.dpi == 300

        vt.set_dpi(150)
        assert vt.dpi == 150

    def test_set_threshold(self):
        """Test set_threshold method."""
        vt = VisualTest()

        vt.set_threshold(0.5)
        assert vt.threshold == 0.5

        vt.set_threshold(0.1)
        assert vt.threshold == 0.1

    def test_set_screenshot_format(self):
        """Test set_screenshot_format method."""
        vt = VisualTest()

        vt.set_screenshot_format("png")
        assert vt.screenshot_format == "png"

        vt.set_screenshot_format("jpg")
        assert vt.screenshot_format == "jpg"

    def test_set_embed_screenshots(self):
        """Test set_embed_screenshots method."""
        vt = VisualTest()

        vt.set_embed_screenshots(True)
        assert vt.embed_screenshots is True

        vt.set_embed_screenshots(False)
        assert vt.embed_screenshots is False

    def test_set_take_screenshots(self):
        """Test set_take_screenshots method."""
        vt = VisualTest()

        vt.set_take_screenshots(True)
        assert vt.take_screenshots is True

        vt.set_take_screenshots(False)
        assert vt.take_screenshots is False

    def test_set_show_diff(self):
        """Test set_show_diff method."""
        vt = VisualTest()

        vt.set_show_diff(True)
        assert vt.show_diff is True

        vt.set_show_diff(False)
        assert vt.show_diff is False

    def test_set_screenshot_dir(self):
        """Test set_screenshot_dir method."""
        vt = VisualTest()

        vt.set_screenshot_dir("custom_screenshots")
        assert vt.screenshot_dir == Path("custom_screenshots")

        vt.set_screenshot_dir("/absolute/path")
        assert vt.screenshot_dir == Path("/absolute/path")

    def test_set_reference_run(self):
        """Test set_reference_run method."""
        vt = VisualTest()

        vt.set_reference_run(True)
        assert vt.reference_run is True

        vt.set_reference_run(False)
        assert vt.reference_run is False

    def test_set_force_ocr(self):
        """Test set_force_ocr method."""
        vt = VisualTest()

        vt.set_force_ocr(True)
        assert vt.force_ocr is True

        vt.set_force_ocr(False)
        assert vt.force_ocr is False


class TestVisualTestUtilityMethods:
    """Test cases for VisualTest utility methods."""

    def test_concatenate_images_safely_same_height(self):
        """Test concatenating images with same height."""
        vt = VisualTest()
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 150, 3), dtype=np.uint8)

        result = vt.concatenate_images_safely(img1, img2, axis=1)

        assert result.shape == (100, 250, 3)

    def test_concatenate_images_safely_different_heights(self):
        """Test concatenating images with different heights."""
        vt = VisualTest()
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((150, 100, 3), dtype=np.uint8)

        result = vt.concatenate_images_safely(
            img1, img2, axis=1, fill_color=(255, 0, 0)
        )

        assert result.shape == (150, 200, 3)

    def test_concatenate_images_safely_vertical(self):
        """Test concatenating images vertically."""
        vt = VisualTest()
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((100, 150, 3), dtype=np.uint8)

        result = vt.concatenate_images_safely(img1, img2, axis=0)

        assert result.shape == (200, 150, 3)

    def test_blend_two_images(self):
        """Test blending two images."""
        vt = VisualTest()
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 128

        result = vt.blend_two_images(img1, img2)

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_blend_two_images_with_ignore_color(self):
        """Test blending two images with ignore color."""
        vt = VisualTest()
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 255  # All white

        result = vt.blend_two_images(img1, img2, ignore_color=[255, 255, 255])

        # White pixels should be ignored, so result should be mostly black
        assert np.array_equal(result, img1)

    def test_is_bounding_box_reasonable_valid(self):
        """Test is_bounding_box_reasonable with valid corners."""
        vt = VisualTest()
        corners = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)

        result = vt.is_bounding_box_reasonable(corners)

        assert result is True

    def test_get_diff_rectangles(self):
        """Test get_diff_rectangles method."""
        vt = VisualTest()
        # Create a difference image with a white rectangle in the center
        diff = np.zeros((100, 100), dtype=np.uint8)
        diff[30:70, 30:70] = 255

        rectangles = vt.get_diff_rectangles(diff)

        assert len(rectangles) == 1
        rect = rectangles[0]
        assert rect["x"] == 30
        assert rect["y"] == 30
        assert rect["width"] == 40
        assert rect["height"] == 40

    def test_get_diff_rectangles_empty(self):
        """Test get_diff_rectangles with no differences."""
        vt = VisualTest()
        diff = np.zeros((100, 100), dtype=np.uint8)

        rectangles = vt.get_diff_rectangles(diff)

        assert len(rectangles) == 0

    def test_get_diff_rectangles_multiple(self):
        """Test get_diff_rectangles with multiple difference areas."""
        vt = VisualTest()
        diff = np.zeros((100, 100), dtype=np.uint8)
        diff[10:30, 10:30] = 255  # First rectangle
        diff[70:90, 70:90] = 255  # Second rectangle

        rectangles = vt.get_diff_rectangles(diff)

        assert len(rectangles) == 2


class TestVisualTestCompareImages:
    """Test cases for compare_images method edge cases."""

    @patch("DocTest.VisualTest.is_url")
    @patch("DocTest.VisualTest.download_file_from_url")
    @patch("DocTest.VisualTest.DocumentRepresentation")
    def test_compare_images_url_download(self, mock_doc, mock_download, mock_is_url):
        """Test compare_images with URL inputs."""
        vt = VisualTest()
        mock_is_url.side_effect = [True, True]  # Both are URLs
        mock_download.side_effect = ["ref_local.pdf", "cand_local.pdf"]

        # Mock DocumentRepresentation
        mock_ref_doc = MagicMock()
        mock_cand_doc = MagicMock()
        mock_doc.side_effect = [mock_ref_doc, mock_cand_doc]

        # Mock pages with identical images
        mock_ref_page = MagicMock()
        mock_cand_page = MagicMock()
        mock_ref_page.image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cand_page.image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_ref_page.compare_with.return_value = (True, None, None, None, 1.0)

        mock_ref_doc.pages = [mock_ref_page]
        mock_cand_doc.pages = [mock_cand_page]

        vt.compare_images("http://example.com/ref.pdf", "http://example.com/cand.pdf")

        mock_download.assert_any_call("http://example.com/ref.pdf")
        mock_download.assert_any_call("http://example.com/cand.pdf")

    @patch("DocTest.VisualTest.DocumentRepresentation")
    def test_compare_images_custom_dpi_threshold(self, mock_doc):
        """Test compare_images with custom DPI and threshold."""
        vt = VisualTest()

        # Mock DocumentRepresentation
        mock_ref_doc = MagicMock()
        mock_cand_doc = MagicMock()
        mock_doc.side_effect = [mock_ref_doc, mock_cand_doc]

        # Mock pages
        mock_ref_page = MagicMock()
        mock_cand_page = MagicMock()
        mock_ref_page.image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cand_page.image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_ref_page.compare_with.return_value = (True, None, None, None, 1.0)

        mock_ref_doc.pages = [mock_ref_page]
        mock_cand_doc.pages = [mock_cand_page]

        vt.compare_images("ref.pdf", "cand.pdf", DPI=300, threshold=0.1)

        # Verify DocumentRepresentation was called with custom DPI
        mock_doc.assert_any_call(
            "ref.pdf",
            dpi=300,
            ocr_engine="ocrs",
            ignore_area_file=None,
            ignore_area=None,
        )
        mock_doc.assert_any_call(
            "cand.pdf",
            dpi=300,
            ocr_engine="ocrs",
            ignore_area_file=None,
            ignore_area=None,
        )

    @patch("DocTest.VisualTest.DocumentRepresentation")
    def test_compare_images_resize_candidate(self, mock_doc):
        """Test compare_images with resize_candidate option."""
        vt = VisualTest()

        # Mock DocumentRepresentation
        mock_ref_doc = MagicMock()
        mock_cand_doc = MagicMock()
        mock_doc.side_effect = [mock_ref_doc, mock_cand_doc]

        # Mock pages with different sizes
        mock_ref_page = MagicMock()
        mock_cand_page = MagicMock()
        mock_ref_page.image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cand_page.image = np.zeros((150, 150, 3), dtype=np.uint8)
        mock_ref_page.compare_with.return_value = (True, None, None, None, 1.0)

        mock_ref_doc.pages = [mock_ref_page]
        mock_cand_doc.pages = [mock_cand_page]

        with patch("cv2.resize") as mock_resize:
            mock_resize.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

            vt.compare_images("ref.pdf", "cand.pdf", resize_candidate=True)

            mock_resize.assert_called_once()

    @patch.dict(os.environ, {"IGNORE_WATERMARKS": "True"})
    @patch("DocTest.VisualTest.DocumentRepresentation")
    def test_compare_images_ignore_watermarks_env(self, mock_doc):
        """Test compare_images with IGNORE_WATERMARKS environment variable."""
        vt = VisualTest()

        # Mock DocumentRepresentation
        mock_ref_doc = MagicMock()
        mock_cand_doc = MagicMock()
        mock_doc.side_effect = [mock_ref_doc, mock_cand_doc]

        # Mock pages
        mock_ref_page = MagicMock()
        mock_cand_page = MagicMock()
        mock_ref_page.image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cand_page.image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_ref_page.compare_with.return_value = (
            False,
            None,
            None,
            np.ones((100, 100), dtype=np.uint8),
            0.5,
        )

        mock_ref_doc.pages = [mock_ref_page]
        mock_cand_doc.pages = [mock_cand_page]

        vt.get_diff_rectangles = MagicMock(
            return_value=[{"x": 45, "y": 45, "width": 10, "height": 10}]
        )

        # This should pass due to watermark detection
        vt.compare_images("ref.pdf", "cand.pdf")


class TestVisualTestTextAndBarcodes:
    """Test cases for text and barcode extraction methods."""

    @patch("DocTest.VisualTest.is_url")
    @patch("DocTest.VisualTest.download_file_from_url")
    @patch("DocTest.VisualTest.DocumentRepresentation")
    def test_get_text_from_document_url(self, mock_doc, mock_download, mock_is_url):
        """Test get_text_from_document with URL input."""
        vt = VisualTest()
        mock_is_url.return_value = True
        mock_download.return_value = "local_file.pdf"

        mock_doc_instance = MagicMock()
        mock_doc.return_value = mock_doc_instance
        mock_doc_instance.get_text.return_value = "Sample text"

        result = vt.get_text_from_document("http://example.com/doc.pdf")

        assert result == "Sample text"
        mock_download.assert_called_once_with("http://example.com/doc.pdf")

    @patch("DocTest.VisualTest.DocumentRepresentation")
    def test_get_text_from_document_local(self, mock_doc):
        """Test get_text_from_document with local file."""
        vt = VisualTest()

        mock_doc_instance = MagicMock()
        mock_doc.return_value = mock_doc_instance
        mock_doc_instance.get_text.return_value = "Local text content"

        result = vt.get_text_from_document("local_file.pdf")

        assert result == "Local text content"
        mock_doc.assert_called_once_with("local_file.pdf", dpi=200, ocr_engine="ocrs")

    @patch("DocTest.VisualTest.is_url")
    @patch("DocTest.VisualTest.download_file_from_url")
    @patch("DocTest.VisualTest.DocumentRepresentation")
    def test_get_barcodes_url(self, mock_doc, mock_download, mock_is_url):
        """Test get_barcodes with URL input."""
        vt = VisualTest()
        mock_is_url.return_value = True
        mock_download.return_value = "local_barcode.pdf"

        mock_doc_instance = MagicMock()
        mock_doc.return_value = mock_doc_instance
        mock_doc_instance.get_barcodes.return_value = [
            {"x": 10, "y": 20, "width": 50, "height": 30, "value": "12345"}
        ]

        result = vt.get_barcodes("http://example.com/barcode.pdf")

        assert len(result) == 1
        assert result[0]["value"] == "12345"
        mock_download.assert_called_once_with("http://example.com/barcode.pdf")

    @patch("DocTest.VisualTest.DocumentRepresentation")
    def test_get_barcodes_with_assertion(self, mock_doc):
        """Test get_barcodes with assertion parameters."""
        vt = VisualTest()

        mock_doc_instance = MagicMock()
        mock_doc.return_value = mock_doc_instance
        mock_doc_instance.get_barcodes.return_value = [
            {"value": "ABC123"},
            {"value": "XYZ789"},
        ]

        with patch("DocTest.VisualTest.verify_assertion") as mock_verify:
            mock_verify.return_value = ["ABC123", "XYZ789"]

            vt.get_barcodes(
                "test.pdf", assertion_operator=None, assertion_expected="ABC"
            )

            # Since assertion_operator is None, verify_assertion should be called
            # but the assertion_operator will be checked within the function

    @patch("DocTest.VisualTest.DocumentRepresentation")
    def test_get_barcodes_from_document_alias(self, mock_doc):
        """Test get_barcodes_from_document as alias method."""
        vt = VisualTest()

        mock_doc_instance = MagicMock()
        mock_doc.return_value = mock_doc_instance
        mock_doc_instance.get_barcodes.return_value = [{"value": "test"}]

        with patch.object(vt, "get_barcodes") as mock_get_barcodes:
            mock_get_barcodes.return_value = [{"value": "test"}]

            vt.get_barcodes_from_document("test.pdf")

            mock_get_barcodes.assert_called_once_with("test.pdf", None, None, None)


class TestVisualTestMovementDetection:
    """Test cases for movement detection functionality."""

    def test_init_movement_detection_stats(self):
        """Test initialization of movement detection statistics."""
        vt = VisualTest()

        # The method initializes internal stats, doesn't return them
        vt._init_movement_detection_stats()

        # Check that the stats were initialized
        assert hasattr(vt, "_movement_stats")
        stats = vt._movement_stats
        assert "total_attempts" in stats
        assert "successful_detections" in stats
        assert "method_success" in stats
        assert stats["total_attempts"] == 0
        assert stats["successful_detections"] == 0
        assert "template" in stats["method_success"]
        assert "orb" in stats["method_success"]
        assert "sift" in stats["method_success"]

    def test_validate_homography_matrix_valid(self):
        """Test validation of valid homography matrix."""
        vt = VisualTest()

        # Valid homography matrix (identity)
        H = np.eye(3, dtype=np.float32)

        result = vt._validate_homography_matrix(H)

        assert result is True

    def test_validate_homography_matrix_invalid_determinant(self):
        """Test validation of homography matrix with invalid determinant."""
        vt = VisualTest()

        # Invalid homography matrix (zero determinant)
        H = np.zeros((3, 3), dtype=np.float32)

        result = vt._validate_homography_matrix(H)

        assert result is False

    def test_validate_homography_matrix_extreme_scale(self):
        """Test validation of homography matrix with extreme scaling."""
        vt = VisualTest()

        # Homography with extreme scaling
        H = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]], dtype=np.float32)

        result = vt._validate_homography_matrix(H)

        assert result is False


class TestVisualTestErrorHandling:
    """Test cases for error handling and edge cases."""

    @patch("DocTest.VisualTest.DocumentRepresentation")
    def test_compare_images_different_dimensions(self, mock_doc):
        """Test compare_images with different image dimensions."""
        vt = VisualTest()

        # Mock DocumentRepresentation
        mock_ref_doc = MagicMock()
        mock_cand_doc = MagicMock()
        mock_doc.side_effect = [mock_ref_doc, mock_cand_doc]

        # Mock pages with different dimensions
        mock_ref_page = MagicMock()
        mock_cand_page = MagicMock()
        mock_ref_page.image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cand_page.image = np.zeros((150, 200, 3), dtype=np.uint8)

        mock_ref_doc.pages = [mock_ref_page]
        mock_cand_doc.pages = [mock_cand_page]

        vt.add_screenshot_to_log = MagicMock()

        with pytest.raises(AssertionError, match="The compared images are different."):
            vt.compare_images("ref.pdf", "cand.pdf")

    def test_get_diff_rectangles_with_noise(self):
        """Test get_diff_rectangles with noisy differences."""
        vt = VisualTest()

        # Create difference image with noise (small scattered white pixels)
        diff = np.zeros((100, 100), dtype=np.uint8)
        # Add some single pixel noise
        diff[10, 10] = 255
        diff[20, 30] = 255
        diff[50, 60] = 255

        rectangles = vt.get_diff_rectangles(diff)

        # Should filter out very small rectangles
        assert len(rectangles) == 0 or all(
            r["width"] * r["height"] > 1 for r in rectangles
        )
