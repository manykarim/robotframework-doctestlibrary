import json
import re
import sys
import cv2
from DocTest.config import DEFAULT_CONFIDENCE, MINIMUM_OCR_RESOLUTION
from typing import List, Dict, Optional

class IgnoreAreaManager:
    def __init__(self, ignore_area_file=None, mask=None):
        self.ignore_area_file = ignore_area_file
        self.mask = mask
        self.ignore_areas = []

    def read_ignore_areas(self):
        """
        Read ignore areas from the provided file or mask and return a list of ignore areas.
        """
        ignore_areas = None
        if self.ignore_area_file:
            ignore_areas = self._load_ignore_area_file()
        elif self.mask:
            ignore_areas = self._parse_mask()

        if ignore_areas:
            if not isinstance(ignore_areas, list):
                ignore_areas = [ignore_areas]
            return ignore_areas
        return []
    
    
    def _load_ignore_area_file(self):
        """Load ignore areas from the provided JSON file."""
        try:
            with open(self.ignore_area_file, 'r') as f:
                return json.load(f)
        except IOError as err:
            print(f"Ignore Area File {self.ignore_area_file} is not accessible.")
            print(f"I/O error: {err}")
        except Exception:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def _parse_mask(self):
        """Parse mask if provided as a string, dict, or list."""
        if isinstance(self.mask, dict) or isinstance(self.mask, list):
            return self.mask
        elif isinstance(self.mask, str):
            try:
                return json.loads(self.mask)
            except json.JSONDecodeError:
                return self._parse_mask_string(self.mask)

    def _parse_mask_string(self, mask_str):
        """Parse a mask string that uses a custom format, e.g., 'top:10;bottom:10'."""
        ignore_areas = []
        mask_list = mask_str.split(';')
        for mask in mask_list:
            if mask:
                location, percent = mask.split(':', 1)
                if location in ['top', 'bottom', 'left', 'right'] and percent.isnumeric():
                    ignore_areas.append({'page': 'all', 'type': 'area', 'location': location, 'percent': percent})
                else:
                    print(f'The mask {mask} is not valid. The percent value is not a number.')
        return ignore_areas

    def _process_ignore_area(self, ignore_area, dpi, text_content):
        """Process each ignore area based on its type and convert it into pixel-based coordinates."""
        ignore_area_type = ignore_area.get('type')
        
        if ignore_area_type in ['pattern', 'line_pattern', 'word_pattern'] and text_content:
            self._process_pattern_ignore_area(ignore_area, dpi, text_content)
        elif ignore_area_type == 'coordinates':
            self._process_coordinates_ignore_area(ignore_area, dpi)
        elif ignore_area_type == 'area':
            self._process_area_ignore_area(ignore_area, dpi)

    def _process_pattern_ignore_area(self, ignore_area, dpi, text_content):
        """Handle pattern-based ignore areas by searching the page for text patterns."""
        pattern = ignore_area.get('pattern')
        xoffset = int(ignore_area.get('xoffset', 0))
        yoffset = int(ignore_area.get('yoffset', 0))

        # Iterate through text data to identify matching patterns and mark as ignore areas
        for i, page_text in enumerate(text_content):
            n_boxes = len(page_text['text'])
            for j in range(n_boxes):
                if int(page_text.get('conf', [0])[j]) > DEFAULT_CONFIDENCE and re.match(pattern, page_text['text'][j]):
                    x, y, w, h = page_text['left'][j], page_text['top'][j], page_text['width'][j], page_text['height'][j]
                    pixel_factor = dpi / MINIMUM_OCR_RESOLUTION
                    text_mask = {"page": i + 1, "x": int(x * pixel_factor) - xoffset, "y": int(y * pixel_factor) - yoffset,
                                 "width": int(w * pixel_factor) + 2 * xoffset, "height": int(h * pixel_factor) + 2 * yoffset}
                    self.ignore_areas.append(text_mask)

    def _process_coordinates_ignore_area(self, ignore_area, dpi):
        """Convert coordinate-based ignore areas into pixel-wise ignore areas."""
        page = ignore_area.get('page', 'all')
        unit = ignore_area.get('unit', 'px')
        x, y, h, w = self._convert_to_pixels(ignore_area, unit, dpi)
        self.ignore_areas.append({"page": page, "x": x, "y": y, "height": h, "width": w})

    def _convert_to_pixels(self, ignore_area, unit, dpi):
        """Convert dimensions from cm, mm, or px to pixel units."""
        x, y, h, w = int(ignore_area['x']), int(ignore_area['y']), int(ignore_area['height']), int(ignore_area['width'])
        if unit == 'mm':
            constant = dpi / 25.4
            x, y, h, w = int(x * constant), int(y * constant), int(h * constant), int(w * constant)
        elif unit == 'cm':
            constant = dpi / 2.54
            x, y, h, w = int(x * constant), int(y * constant), int(h * constant), int(w * constant)
        return x, y, h, w

    def _process_area_ignore_area(self, ignore_area, dpi):
        """Handle area-based ignore areas (e.g., 'top', 'bottom', 'left', 'right') as percentages."""
        page = ignore_area.get('page', 'all')
        location = ignore_area.get('location', None)
        percent = int(ignore_area.get('percent', 10))
        
        # Assuming that opencv_images is accessible here
        image_height = self.opencv_images[0].shape[0]  # Just for demo, need to handle for different pages
        image_width = self.opencv_images[0].shape[1]

        # Process based on location (top, bottom, left, right)
        if location == 'top':
            height = int(image_height * percent / 100)
            self.ignore_areas.append({"page": page, "x": 0, "y": 0, "width": image_width, "height": height})
        elif location == 'bottom':
            height = int(image_height * percent / 100)
            self.ignore_areas.append({"page": page, "x": 0, "y": image_height - height, "width": image_width, "height": height})
        # Handle other locations (left, right)...
