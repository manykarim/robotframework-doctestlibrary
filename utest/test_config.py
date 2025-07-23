"""Unit tests for config module."""

from DocTest.config import (
    ADD_PIXELS_TO_IGNORE_AREA,
    DEFAULT_CONFIDENCE,
    DEFAULT_DPI,
    EAST_CONFIDENCE,
    MINIMUM_OCR_RESOLUTION,
    OCR_ENGINE_DEFAULT,
    SCREENSHOT_FORMAT,
    TESSERACT_CONFIG,
)


class TestConfigConstants:
    """Test cases for configuration constants."""

    def test_default_confidence_value(self):
        """Test that DEFAULT_CONFIDENCE has expected value."""
        assert DEFAULT_CONFIDENCE == 20
        assert isinstance(DEFAULT_CONFIDENCE, int)

    def test_minimum_ocr_resolution_value(self):
        """Test that MINIMUM_OCR_RESOLUTION has expected value."""
        assert MINIMUM_OCR_RESOLUTION == 300
        assert isinstance(MINIMUM_OCR_RESOLUTION, int)

    def test_screenshot_format_value(self):
        """Test that SCREENSHOT_FORMAT has expected value."""
        assert SCREENSHOT_FORMAT == "jpg"
        assert isinstance(SCREENSHOT_FORMAT, str)

    def test_default_dpi_value(self):
        """Test that DEFAULT_DPI has expected value."""
        assert DEFAULT_DPI == 200
        assert isinstance(DEFAULT_DPI, int)

    def test_ocr_engine_default_value(self):
        """Test that OCR_ENGINE_DEFAULT has expected value."""
        assert OCR_ENGINE_DEFAULT == "tesseract"
        assert isinstance(OCR_ENGINE_DEFAULT, str)

    def test_east_confidence_value(self):
        """Test that EAST_CONFIDENCE has expected value."""
        assert EAST_CONFIDENCE == 0.5
        assert isinstance(EAST_CONFIDENCE, float)

    def test_add_pixels_to_ignore_area_value(self):
        """Test that ADD_PIXELS_TO_IGNORE_AREA has expected value."""
        assert ADD_PIXELS_TO_IGNORE_AREA == 2
        assert isinstance(ADD_PIXELS_TO_IGNORE_AREA, int)

    def test_tesseract_config_value(self):
        """Test that TESSERACT_CONFIG has expected value."""
        assert TESSERACT_CONFIG == "--psm 11 -l eng"
        assert isinstance(TESSERACT_CONFIG, str)

    def test_all_constants_are_defined(self):
        """Test that all expected constants are defined and not None."""
        constants = [
            DEFAULT_CONFIDENCE,
            MINIMUM_OCR_RESOLUTION,
            SCREENSHOT_FORMAT,
            DEFAULT_DPI,
            OCR_ENGINE_DEFAULT,
            EAST_CONFIDENCE,
            ADD_PIXELS_TO_IGNORE_AREA,
            TESSERACT_CONFIG,
        ]

        for constant in constants:
            assert constant is not None

    def test_positive_numeric_values(self):
        """Test that numeric constants have positive values."""
        assert DEFAULT_CONFIDENCE > 0
        assert MINIMUM_OCR_RESOLUTION > 0
        assert DEFAULT_DPI > 0
        assert EAST_CONFIDENCE > 0
        assert ADD_PIXELS_TO_IGNORE_AREA >= 0

    def test_string_values_not_empty(self):
        """Test that string constants are not empty."""
        assert len(SCREENSHOT_FORMAT) > 0
        assert len(OCR_ENGINE_DEFAULT) > 0
        assert len(TESSERACT_CONFIG) > 0
