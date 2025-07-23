"""Unit tests for IgnoreAreaManager module."""

from unittest.mock import mock_open, patch

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
    @patch("builtins.print")
    def test_load_ignore_area_file_io_error(self, mock_print, mock_file):
        """Test loading ignore area file with IO error."""
        manager = IgnoreAreaManager(ignore_area_file="nonexistent.json")
        result = manager._load_ignore_area_file()

        assert result is None
        assert mock_print.call_count >= 2  # Two print statements

    @patch("builtins.open", new_callable=mock_open, read_data="invalid json")
    @patch("builtins.print")
    def test_load_ignore_area_file_json_error(self, mock_print, mock_file):
        """Test loading ignore area file with invalid JSON."""
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

    @patch("builtins.print")
    def test_parse_mask_string_invalid_location(self, mock_print):
        """Test parsing mask string with invalid location."""
        mask = "invalid:10"
        manager = IgnoreAreaManager(mask=mask)
        result = manager._parse_mask()

        assert result == []
        mock_print.assert_called()

    @patch("builtins.print")
    def test_parse_mask_string_invalid_percent(self, mock_print):
        """Test parsing mask string with invalid percent."""
        mask = "top:invalid"
        manager = IgnoreAreaManager(mask=mask)
        result = manager._parse_mask()

        assert result == []
        mock_print.assert_called()

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

    def test_convert_to_pixels_px_unit(self):
        """Test converting coordinates to pixels with px unit."""
        manager = IgnoreAreaManager()
        ignore_area = {"x": 10, "y": 20, "height": 30, "width": 40}
        x, y, h, w = manager._convert_to_pixels(ignore_area, "px", 200)

        assert x == 10
        assert y == 20
        assert h == 30
        assert w == 40

    def test_convert_to_pixels_mm_unit(self):
        """Test converting coordinates to pixels with mm unit."""
        manager = IgnoreAreaManager()
        ignore_area = {"x": 10, "y": 20, "height": 30, "width": 40}
        x, y, h, w = manager._convert_to_pixels(
            ignore_area, "mm", 254
        )  # 254 DPI = 10 px/mm

        assert x == 100  # 10 * (254/25.4) = 100
        assert y == 200  # 20 * (254/25.4) = 200
        assert h == 300  # 30 * (254/25.4) = 300
        assert w == 400  # 40 * (254/25.4) = 400

    def test_convert_to_pixels_cm_unit(self):
        """Test converting coordinates to pixels with cm unit."""
        manager = IgnoreAreaManager()
        ignore_area = {"x": 1, "y": 2, "height": 3, "width": 4}
        x, y, h, w = manager._convert_to_pixels(
            ignore_area, "cm", 254
        )  # 254 DPI = 100 px/cm

        assert x == 100  # 1 * (254/2.54) = 100
        assert y == 200  # 2 * (254/2.54) = 200
        assert h == 300  # 3 * (254/2.54) = 300
        assert w == 400  # 4 * (254/2.54) = 400
