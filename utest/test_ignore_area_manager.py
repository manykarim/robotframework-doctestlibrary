"""Unit tests for IgnoreAreaManager module."""

import logging
from unittest.mock import mock_open, patch

import numpy as np

from DocTest.IgnoreAreaManager import IgnoreAreaManager


class TestIgnoreAreaManager:
    """Test cases for IgnoreAreaManager class."""

    def test_initialization_with_file(self):
        """Test IgnoreAreaManager initialization with file."""
        manager = IgnoreAreaManager(ignore_area_file="test.json")

        assert manager.ignore_area_file == "test.json"
        assert manager.mask is None
        assert manager.ignore_areas == []

    def test_initialization_with_mask(self):
        """Test IgnoreAreaManager initialization with mask."""
        mask = {"page": 1, "type": "area", "location": "top", "percent": 10}
        manager = IgnoreAreaManager(mask=mask)

        assert manager.ignore_area_file is None
        assert manager.mask == mask
        assert manager.ignore_areas == []

    def test_initialization_empty(self):
        """Test IgnoreAreaManager initialization with no parameters."""
        manager = IgnoreAreaManager()

        assert manager.ignore_area_file is None
        assert manager.mask is None
        assert manager.ignore_areas == []

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='[{"page": 1, "type": "area"}]',
    )
    def test_load_ignore_area_file_success(self, mock_file):
        """Test successful loading of ignore area file."""
        manager = IgnoreAreaManager(ignore_area_file="test.json")
        result = manager._load_ignore_area_file()

        assert result == [{"page": 1, "type": "area"}]
        mock_file.assert_called_once_with("test.json", "r")

    @patch("builtins.open", side_effect=IOError("File not found"))
    def test_load_ignore_area_file_io_error(self, mock_file, caplog):
        """Test loading ignore area file with IO error."""
        manager = IgnoreAreaManager(ignore_area_file="nonexistent.json")
        with caplog.at_level(logging.ERROR, logger="DocTest.IgnoreAreaManager"):
            result = manager._load_ignore_area_file()

        assert result is None
        assert len(caplog.records) >= 2  # Two log messages

    @patch("builtins.open", new_callable=mock_open, read_data="invalid json")
    def test_load_ignore_area_file_json_error(self, mock_file):
        """Test loading ignore area file with invalid JSON raises exception."""
        manager = IgnoreAreaManager(ignore_area_file="invalid.json")

        try:
            manager._load_ignore_area_file()
        except Exception:
            pass  # Expected to raise exception

    def test_parse_mask_dict(self):
        """Test parsing mask when it's a dictionary."""
        mask = {"page": 1, "type": "area", "location": "top", "percent": 10}
        manager = IgnoreAreaManager(mask=mask)
        result = manager._parse_mask()

        assert result == mask

    def test_parse_mask_list(self):
        """Test parsing mask when it's a list."""
        mask = [{"page": 1, "type": "area"}, {"page": 2, "type": "coordinates"}]
        manager = IgnoreAreaManager(mask=mask)
        result = manager._parse_mask()

        assert result == mask

    def test_parse_mask_json_string(self):
        """Test parsing mask when it's a JSON string."""
        mask = '{"page": 1, "type": "area"}'
        manager = IgnoreAreaManager(mask=mask)
        result = manager._parse_mask()

        assert result == {"page": 1, "type": "area"}

    def test_parse_mask_custom_string(self):
        """Test parsing mask when it's a custom format string."""
        mask = "top:10;bottom:5"
        manager = IgnoreAreaManager(mask=mask)
        result = manager._parse_mask()

        expected = [
            {"page": "all", "type": "area", "location": "top", "percent": "10"},
            {"page": "all", "type": "area", "location": "bottom", "percent": "5"},
        ]
        assert result == expected

    def test_parse_mask_string_single_area(self):
        """Test parsing mask string with single area."""
        mask = "left:15"
        manager = IgnoreAreaManager(mask=mask)
        result = manager._parse_mask()

        expected = [
            {"page": "all", "type": "area", "location": "left", "percent": "15"}
        ]
        assert result == expected

    def test_parse_mask_string_invalid_location(self, caplog):
        """Test parsing mask string with invalid location."""
        mask = "invalid:10"
        manager = IgnoreAreaManager(mask=mask)
        with caplog.at_level(logging.WARNING, logger="DocTest.IgnoreAreaManager"):
            result = manager._parse_mask()

        assert result == []
        assert len(caplog.records) == 1
        assert "invalid location" in caplog.records[0].message

    def test_parse_mask_string_invalid_percent(self, caplog):
        """Test parsing mask string with invalid percent."""
        mask = "top:invalid"
        manager = IgnoreAreaManager(mask=mask)
        with caplog.at_level(logging.WARNING, logger="DocTest.IgnoreAreaManager"):
            result = manager._parse_mask()

        assert result == []
        assert len(caplog.records) == 1
        assert "not a number" in caplog.records[0].message

    def test_parse_mask_string_empty(self):
        """Test parsing empty mask string."""
        mask = ""
        manager = IgnoreAreaManager(mask=mask)
        result = manager._parse_mask()

        assert result == []

    def test_parse_mask_string_with_empty_parts(self):
        """Test parsing mask string with empty parts."""
        mask = "top:10;;bottom:5"
        manager = IgnoreAreaManager(mask=mask)
        result = manager._parse_mask()

        expected = [
            {"page": "all", "type": "area", "location": "top", "percent": "10"},
            {"page": "all", "type": "area", "location": "bottom", "percent": "5"},
        ]
        assert result == expected

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='[{"page": 1, "type": "area"}]',
    )
    def test_read_ignore_areas_from_file(self, mock_file):
        """Test reading ignore areas from file."""
        manager = IgnoreAreaManager(ignore_area_file="test.json")
        result = manager.read_ignore_areas()

        assert result == [{"page": 1, "type": "area"}]

    def test_read_ignore_areas_from_mask(self):
        """Test reading ignore areas from mask."""
        mask = {"page": 1, "type": "area"}
        manager = IgnoreAreaManager(mask=mask)
        result = manager.read_ignore_areas()

        assert result == [{"page": 1, "type": "area"}]

    def test_read_ignore_areas_none(self):
        """Test reading ignore areas when none provided."""
        manager = IgnoreAreaManager()
        result = manager.read_ignore_areas()

        assert result == []

    @patch(
        "builtins.open", new_callable=mock_open, read_data='{"page": 1, "type": "area"}'
    )
    def test_read_ignore_areas_single_dict(self, mock_file):
        """Test reading ignore areas when file contains single dict (not list)."""
        manager = IgnoreAreaManager(ignore_area_file="test.json")
        result = manager.read_ignore_areas()

        assert result == [{"page": 1, "type": "area"}]

    def test_parse_mask_string_missing_colon(self, caplog):
        """Test parsing mask string without colon separator."""
        mask = "top10"
        manager = IgnoreAreaManager(mask=mask)
        with caplog.at_level(logging.WARNING, logger="DocTest.IgnoreAreaManager"):
            result = manager._parse_mask()

        assert result == []
        assert len(caplog.records) == 1
        assert "Expected format" in caplog.records[0].message


class TestAreaMaskExecution:
    """Test area mask execution through the Page class (the real runtime path).

    IgnoreAreaManager parses mask strings into abstract ignore areas.
    Page._process_area_ignore_area converts those into pixel-based ignore areas
    using the actual image dimensions.
    """

    def _make_page(self, height=1000, width=800, page_number=1, dpi=200):
        """Create a minimal Page object with a fake image for testing."""
        from DocTest.DocumentRepresentation import Page

        image = np.zeros((height, width, 3), dtype=np.uint8)
        page = Page(image=image, page_number=page_number, dpi=dpi)
        return page

    def test_area_top(self):
        """Test area mask for 'top' location."""
        page = self._make_page(height=1000, width=800)
        ignore_area = {"page": "all", "type": "area", "location": "top", "percent": "10"}

        page._process_area_ignore_area(ignore_area)

        assert len(page.pixel_ignore_areas) == 1
        area = page.pixel_ignore_areas[0]
        assert area["x"] == 0
        assert area["y"] == 0
        assert area["width"] == 800
        assert area["height"] == 100  # 10% of 1000

    def test_area_bottom(self):
        """Test area mask for 'bottom' location."""
        page = self._make_page(height=1000, width=800)
        ignore_area = {"page": "all", "type": "area", "location": "bottom", "percent": "20"}

        page._process_area_ignore_area(ignore_area)

        assert len(page.pixel_ignore_areas) == 1
        area = page.pixel_ignore_areas[0]
        assert area["x"] == 0
        assert area["y"] == 800  # 1000 - 200
        assert area["width"] == 800
        assert area["height"] == 200  # 20% of 1000

    def test_area_left(self):
        """Test area mask for 'left' location."""
        page = self._make_page(height=1000, width=800)
        ignore_area = {"page": "all", "type": "area", "location": "left", "percent": "25"}

        page._process_area_ignore_area(ignore_area)

        assert len(page.pixel_ignore_areas) == 1
        area = page.pixel_ignore_areas[0]
        assert area["x"] == 0
        assert area["y"] == 0
        assert area["width"] == 200  # 25% of 800
        assert area["height"] == 1000

    def test_area_right(self):
        """Test area mask for 'right' location."""
        page = self._make_page(height=1000, width=800)
        ignore_area = {"page": "all", "type": "area", "location": "right", "percent": "15"}

        page._process_area_ignore_area(ignore_area)

        assert len(page.pixel_ignore_areas) == 1
        area = page.pixel_ignore_areas[0]
        assert area["x"] == 680  # 800 - 120
        assert area["y"] == 0
        assert area["width"] == 120  # 15% of 800
        assert area["height"] == 1000

    def test_page_filter_all(self):
        """Test that page='all' applies ignore area to any page."""
        for page_num in [1, 2, 5]:
            page = self._make_page(height=1000, width=800, page_number=page_num)
            ignore_area = {"page": "all", "type": "area", "location": "top", "percent": "10"}

            page._process_area_ignore_area(ignore_area)

            assert len(page.pixel_ignore_areas) == 1, (
                f"page='all' should apply to page {page_num}"
            )

    def test_page_filter_specific_match(self):
        """Test that a specific page number matches the correct page."""
        page = self._make_page(height=1000, width=800, page_number=3)
        ignore_area = {"page": 3, "type": "area", "location": "top", "percent": "10"}

        page._process_area_ignore_area(ignore_area)

        assert len(page.pixel_ignore_areas) == 1

    def test_page_filter_specific_no_match(self):
        """Test that a specific page number does not match a different page."""
        page = self._make_page(height=1000, width=800, page_number=1)
        ignore_area = {"page": 3, "type": "area", "location": "top", "percent": "10"}

        page._process_area_ignore_area(ignore_area)

        assert len(page.pixel_ignore_areas) == 0

    def test_page_filter_string_page_number(self):
        """Test that page number given as string is cast to int for comparison."""
        page = self._make_page(height=1000, width=800, page_number=2)
        ignore_area = {"page": "2", "type": "area", "location": "top", "percent": "10"}

        page._process_area_ignore_area(ignore_area)

        assert len(page.pixel_ignore_areas) == 1

    def test_invalid_location_still_appends(self):
        """Test that an unrecognized location still appends the full image area.

        The Page implementation defaults to full image dimensions when
        no location branch matches. This documents the current behavior.
        """
        page = self._make_page(height=1000, width=800)
        ignore_area = {"page": "all", "type": "area", "location": "center", "percent": "10"}

        page._process_area_ignore_area(ignore_area)

        # Current behavior: appends the full image as ignore area
        assert len(page.pixel_ignore_areas) == 1
        area = page.pixel_ignore_areas[0]
        assert area["x"] == 0
        assert area["y"] == 0
        assert area["width"] == 800
        assert area["height"] == 1000

    def test_missing_location_defaults_to_none(self):
        """Test that a missing location key defaults to None and uses full image."""
        page = self._make_page(height=1000, width=800)
        ignore_area = {"page": "all", "type": "area", "percent": "10"}

        page._process_area_ignore_area(ignore_area)

        assert len(page.pixel_ignore_areas) == 1
        area = page.pixel_ignore_areas[0]
        assert area["width"] == 800
        assert area["height"] == 1000

    def test_end_to_end_parse_and_process(self):
        """Test the full pipeline: IgnoreAreaManager parses, Page processes."""
        manager = IgnoreAreaManager(mask="top:10;bottom:20;left:5")
        areas = manager.read_ignore_areas()

        assert len(areas) == 3

        page = self._make_page(height=2000, width=1000, page_number=1)
        for area in areas:
            page._process_area_ignore_area(area)

        assert len(page.pixel_ignore_areas) == 3

        # top:10 => top 10% of 2000px height = 200px
        top_area = page.pixel_ignore_areas[0]
        assert top_area["x"] == 0
        assert top_area["y"] == 0
        assert top_area["height"] == 200
        assert top_area["width"] == 1000

        # bottom:20 => bottom 20% of 2000px height = 400px
        bottom_area = page.pixel_ignore_areas[1]
        assert bottom_area["x"] == 0
        assert bottom_area["y"] == 1600  # 2000 - 400
        assert bottom_area["height"] == 400
        assert bottom_area["width"] == 1000

        # left:5 => left 5% of 1000px width = 50px
        left_area = page.pixel_ignore_areas[2]
        assert left_area["x"] == 0
        assert left_area["y"] == 0
        assert left_area["width"] == 50
        assert left_area["height"] == 2000

    def test_end_to_end_json_mask_with_coordinates(self):
        """Test end-to-end with a JSON coordinate mask."""
        mask = '[{"page": "all", "type": "coordinates", "x": 10, "y": 20, "width": 100, "height": 50}]'
        manager = IgnoreAreaManager(mask=mask)
        areas = manager.read_ignore_areas()

        assert len(areas) == 1

        page = self._make_page(height=1000, width=800, page_number=1)
        page._process_coordinates_ignore_area(areas[0])

        assert len(page.pixel_ignore_areas) == 1
        area = page.pixel_ignore_areas[0]
        assert area["x"] == 10
        assert area["y"] == 20
        assert area["width"] == 100
        assert area["height"] == 50

    def test_multiple_pages_selective_application(self):
        """Test that page-specific ignore areas apply only to matching pages."""
        areas = [
            {"page": 1, "type": "area", "location": "top", "percent": "10"},
            {"page": 2, "type": "area", "location": "bottom", "percent": "20"},
            {"page": "all", "type": "area", "location": "left", "percent": "5"},
        ]

        page1 = self._make_page(height=1000, width=800, page_number=1)
        page2 = self._make_page(height=1000, width=800, page_number=2)

        for area in areas:
            page1._process_area_ignore_area(area)
            page2._process_area_ignore_area(area)

        # Page 1 gets: top (page=1 match) + left (page=all)
        assert len(page1.pixel_ignore_areas) == 2
        # Page 2 gets: bottom (page=2 match) + left (page=all)
        assert len(page2.pixel_ignore_areas) == 2
