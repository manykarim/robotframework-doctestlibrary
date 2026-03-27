import json
import logging
from typing import List, Dict, Optional, Union

logger = logging.getLogger(__name__)


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
            logger.error("Ignore Area File %s is not accessible.", self.ignore_area_file)
            logger.error("I/O error: %s", err)

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
        valid_locations = ('top', 'bottom', 'left', 'right')
        for mask in mask_list:
            if not mask:
                continue
            parts = mask.split(':', 1)
            if len(parts) != 2:
                logger.warning(
                    "The mask '%s' is not valid. Expected format 'location:percent' "
                    "(e.g., 'top:10').", mask
                )
                continue
            location, percent = parts
            if location not in valid_locations:
                logger.warning(
                    "The mask '%s' has an invalid location '%s'. "
                    "Valid locations are: %s.",
                    mask, location, ', '.join(valid_locations)
                )
                continue
            if not percent.isnumeric():
                logger.warning(
                    "The mask '%s' is not valid. The percent value '%s' is not a number.",
                    mask, percent
                )
                continue
            ignore_areas.append({
                'page': 'all',
                'type': 'area',
                'location': location,
                'percent': percent,
            })
        return ignore_areas
