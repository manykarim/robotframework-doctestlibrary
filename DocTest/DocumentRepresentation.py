import os
import cv2
import pytesseract
import numpy as np
import json
import logging
import re
from pathlib import Path
from skimage import metrics
from typing import List, Dict, Optional, Tuple, Union, Literal
from concurrent.futures import ThreadPoolExecutor
from pytesseract import Output
from DocTest.IgnoreAreaManager import IgnoreAreaManager
from DocTest.config import DEFAULT_DPI, OCR_ENGINE_DEFAULT, DEFAULT_CONFIDENCE, MINIMUM_OCR_RESOLUTION, ADD_PIXELS_TO_IGNORE_AREA, TESSERACT_CONFIG
import tempfile

# Constants


class Page:
    def __init__(self, image: np.ndarray, page_number: int, dpi: int = DEFAULT_DPI):
        self.page_number = page_number
        self.image = image  # Image as NumPy array (OpenCV format)
        self.dpi = dpi
        self.ocr_text_data: Optional[Dict] = None
        self.pdf_text_data: Optional[Dict] = None
        self.pdf_text_dict: Optional[Dict] = None
        self.pdf_text_blocks: Optional[Dict] = None
        self.pdf_text_words: Optional[Dict] = None
        self.text: str = ""
        self.barcodes: List[Dict] = []
        self.ocr_performed = False
        self.pixel_ignore_areas = []
        self.image_rescaled_for_ocr = False
        self.ocr_tokens: List[str] = []

    def apply_ocr(self, ocr_engine: str = OCR_ENGINE_DEFAULT, tesseract_config: str = TESSERACT_CONFIG, confidence: int = DEFAULT_CONFIDENCE):
        """Perform OCR on the page image."""
        # re-scale the image to a standard resolution for OCR if needed
        self.image_rescaled_for_ocr = False
        original_image = None
        target_height, target_width = self.image.shape[:2]
        scale_factor = 1.0
        if self.dpi < MINIMUM_OCR_RESOLUTION:
            # Rescale the image to a higher resolution for better OCR results
            scale_factor = MINIMUM_OCR_RESOLUTION / self.dpi
            if self.image.shape[0] * scale_factor < 32767 and self.image.shape[1] * scale_factor < 32767:
                original_image = self.image.copy()
                target_height, target_width = self.image.shape[:2]
                self.image = cv2.resize(
                    self.image,
                    (0, 0),
                    fx=scale_factor,
                    fy=scale_factor,
                    interpolation=cv2.INTER_CUBIC,
                )
                self.image_rescaled_for_ocr = True
        if ocr_engine == "tesseract":
            config = tesseract_config
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            self.ocr_text_data = pytesseract.image_to_data(thresholded_image, config=config, output_type=Output.DICT)
            self._normalize_ocr_coordinates(scale_factor if self.image_rescaled_for_ocr else 1.0, target_height, target_width)
            self.ocr_performed = True
        elif ocr_engine == "east":
            from DocTest.Ocr import EastTextExtractor
            self.east_text_extractor = EastTextExtractor()
            text = self.east_text_extractor.get_image_text(
                self.image,
                scale_back=scale_factor if self.image_rescaled_for_ocr else 1.0,
                target_size=(target_height, target_width),
            )
            self.ocr_text_data = text
            self._normalize_ocr_coordinates(scale_factor if self.image_rescaled_for_ocr else 1.0, target_height, target_width)
            self._refine_east_tokens(original_image if original_image is not None else self.image)
            self.ocr_performed = True
        if self.image_rescaled_for_ocr and original_image is not None:
            self.image = original_image
            self.image_rescaled_for_ocr = False

    def get_area(self, area: Dict):
        """Gets the area of the image specified by the coordinates."""
        x, y, w, h = area['x'], area['y'], area['width'], area['height']
        return self.image[y:y+h, x:x+w]

    def rescale_image_for_ocr(self):
        pass

    def get_text_content(self):
        """Return the OCR text content."""
        return self.ocr_text_data['text'] if self.ocr_text_data else ""

    def _normalize_ocr_coordinates(self, scale_factor: float, target_height: int, target_width: int):
        """Bring OCR bounding boxes back to the original image scale and clamp them."""
        if not self.ocr_text_data:
            return

        lefts = self.ocr_text_data.get('left') or []
        tops = self.ocr_text_data.get('top') or []
        widths = self.ocr_text_data.get('width') or []
        heights = self.ocr_text_data.get('height') or []

        if not (lefts and tops and widths and heights):
            return

        for idx in range(len(lefts)):
            left = self._coerce_to_float(lefts[idx])
            top = self._coerce_to_float(tops[idx])
            width = self._coerce_to_float(widths[idx])
            height = self._coerce_to_float(heights[idx])

            if scale_factor not in (0, 1):
                left /= scale_factor
                top /= scale_factor
                width /= scale_factor
                height /= scale_factor

            left = max(0, min(target_width - 1, int(round(left))))
            top = max(0, min(target_height - 1, int(round(top))))
            width = max(0, min(target_width - left, int(round(width))))
            height = max(0, min(target_height - top, int(round(height))))

            lefts[idx] = left
            tops[idx] = top
            widths[idx] = width
            heights[idx] = height

    @staticmethod
    def _coerce_to_float(value: Union[str, int, float]) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _normalized_ocr_tokens(self) -> List[str]:
        if self.ocr_tokens:
            return [token for token in (self._normalize_token(t) for t in self.ocr_tokens) if token]

        if not self.ocr_text_data:
            return []
        tokens = []
        for token in self.ocr_text_data.get('text', []):
            normalized = self._normalize_token(token)
            if normalized:
                tokens.append(normalized)
        return tokens

    def _normalize_token(self, token: Optional[str]) -> str:
        if not token:
            return ""
        token = token.strip()
        if not token:
            return ""
        token = token.translate(str.maketrans({
            "»": "",
            "«": "",
            "“": "",
            "”": "",
            "„": "",
            "’": "",
            "‘": "",
        }))
        token = token.strip().strip(".,;:!\"'`")
        if not token:
            return ""
        token = re.sub(r"\s+", " ", token)

        return token

    def _refine_east_tokens(self, base_image: np.ndarray):
        if not self.ocr_text_data:
            return
        tokens = []
        lefts = self.ocr_text_data.get('left') or []
        tops = self.ocr_text_data.get('top') or []
        widths = self.ocr_text_data.get('width') or []
        heights = self.ocr_text_data.get('height') or []

        for left, top, width, height in zip(lefts, tops, widths, heights):
            if width <= 0 or height <= 0:
                continue
            roi = base_image[top:top+height, left:left+width]
            if roi.size == 0:
                continue
            scale_factor = 3 if min(roi.shape[0], roi.shape[1]) < 80 else 1
            roi_for_ocr = roi
            if scale_factor > 1:
                roi_for_ocr = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            text = pytesseract.image_to_string(roi_for_ocr, config='--oem 1 --psm 7 -l eng').strip()
            if text:
                tokens.append(text)

        self.ocr_tokens = tokens

    def get_image_with_ignore_areas(self):
        """Return the image with ignore areas highlighted."""
        image_with_ignore_areas = self.image.copy()
        for ignore_area in self.pixel_ignore_areas:
            x, y, w, h = ignore_area.get('x'), ignore_area.get('y'), ignore_area.get('width'), ignore_area.get('height')
            # Add ADD_PIXELS_TO_IGNORE_AREA to the ignore area dimensions
            # Ensure that the ignore area is within the image bounds

            x -= ADD_PIXELS_TO_IGNORE_AREA
            y -= ADD_PIXELS_TO_IGNORE_AREA
            w += 2 * ADD_PIXELS_TO_IGNORE_AREA
            h += 2 * ADD_PIXELS_TO_IGNORE_AREA

            x = max(0, x)
            y = max(0, y)
            w = min(w, image_with_ignore_areas.shape[1] - x)
            h = min(h, image_with_ignore_areas.shape[0] - y)

            # Draw a filled blue rectangle to cover the ignore area
            cv2.rectangle(image_with_ignore_areas, (x, y), (x + w, y + h), (255, 0, 0), -1)
        return image_with_ignore_areas

    def compare_with(self, other_page: 'Page', threshold: float = 0.0, resize: bool = False, block_based_ssim: bool = False, block_size: int = 32, **kwargs):
        """
        Compare this page with another page and return a tuple of (similarity_result, diff_image).
        
        | =Arguments= | =Description= |
        | ``other_page`` | Another Page object to compare against. |
        | ``threshold`` | The SSIM threshold to determine similarity. |
        """
        if self.dpi != other_page.dpi:
            raise ValueError(f"Page DPI mismatch: {self.dpi} vs {other_page.dpi}")        
        
        blur = kwargs.get('blur', False)

        if self.pixel_ignore_areas:
            other_page.pixel_ignore_areas = self.pixel_ignore_areas
            self.image = self.get_image_with_ignore_areas()
            other_page.image = other_page.get_image_with_ignore_areas()
    
        # Quick check: if the images are of different sizes, they are not similar
        if self.image.shape != other_page.image.shape:
            return False, None, None, None, None
        
        # Quick check: if the images are identical, they are similar
        if np.array_equal(self.image, other_page.image):
            return True, None, None, None, None

        # Perform SSIM (Structural Similarity Index) comparison and get the diff image
        gray_self = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray_other = cv2.cvtColor(other_page.image, cv2.COLOR_BGR2GRAY)

        if blur:
            kernel_size = int(gray_self.shape[1]/30)
            # must be odd if median
            kernel_size += kernel_size%2-1
            gray_self = cv2.GaussianBlur(gray_self, (kernel_size, kernel_size), 1.5)
            gray_other = cv2.GaussianBlur(gray_other, (kernel_size, kernel_size), 1.5)

        score, diff = metrics.structural_similarity(gray_self, gray_other, full=True)

        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        absolute_diff = cv2.absdiff(gray_self, gray_other)

        if block_based_ssim:
            block_based_ssim_result, block_based_ssim_score = self.block_based_ssim_comparison(other_page.image, threshold=threshold, block_size=block_size)            
            if not block_based_ssim_result:
                return False, diff, thresh, absolute_diff, 1.0 - block_based_ssim_score
        # Return a tuple: whether the pages are similar, and the difference image
        return score >= (1.0 - threshold), diff, thresh, absolute_diff, 1.0 - score


    def block_based_ssim_comparison(self, other_image, threshold: float = 0.0, block_size: int = 32) -> Tuple[bool, float]:
        """
        Perform a block-based SSIM comparison between this page's image and another image,
        including partial blocks at the edges.

        :param other_image: The image of the other Page to compare against.
        :param threshold: The minimum SSIM score for a block to be considered similar.
        :param block_size: The size of the blocks for block-based SSIM.
        :return: 
            - A tuple of (similarity_result, lowest_score) 
            where similarity_result is a boolean indicating whether the pages are similar
            - lowest_score is the lowest SSIM score found in any block
        """

        # Dimensions of the reference image
        height, width = self.image.shape[:2]

        # Initialize the lowest score to 1.0 (maximum SSIM)
        lowest_score = 1.0

        # Go row by row in increments of `block_size`
        for y in range(0, height, block_size):
            # Figure out how tall this block really should be (including partial blocks)
            block_height = min(block_size, height - y)

            # Go column by column in increments of `block_size`
            for x in range(0, width, block_size):
                # Figure out how wide this block really should be
                block_width = min(block_size, width - x)

                # Extract block from reference and candidate images
                block_ref = self.image[y:y+block_height, x:x+block_width]
                block_cand = other_image[y:y+block_height, x:x+block_width]

                if np.array_equal(block_ref, block_cand):
                    continue

                # Convert to grayscale
                block_ref_gray = cv2.cvtColor(block_ref, cv2.COLOR_BGR2GRAY)
                block_cand_gray = cv2.cvtColor(block_cand, cv2.COLOR_BGR2GRAY)

                # If either block is zero-sized (shouldn't happen unless images mismatch), skip
                if block_ref_gray.size == 0 or block_cand_gray.size == 0:
                    continue

                # Determine a `win_size` that does not exceed the block's dimensions.
                # structural_similarity requires:
                #    - an odd integer
                #    - <= the smallest dimension of the image/block
                # Default is 7, so reduce only if needed.
                min_side = min(block_ref_gray.shape[0], block_ref_gray.shape[1])
                if min_side >= 7:
                    # Use default (7) if block is at least 7×7
                    win_size = 7
                else:
                    # If block is smaller than 7 in any dimension, pick the largest odd integer <= min_side
                    # (This ensures we do not exceed the partial block dimensions)
                    if min_side % 2 == 0:
                        win_size = max(1, min_side - 1)
                    else:
                        win_size = min_side

                # Compute SSIM score for the current block using the chosen window size
                block_score = metrics.structural_similarity(
                    block_ref_gray,
                    block_cand_gray,
                    win_size=win_size
                )

                # Track the lowest block SSIM
                lowest_score = abs(min(lowest_score, block_score))

                # If any block's SSIM falls below (1.0 - threshold), return immediately
                if lowest_score < (1.0 - threshold):
                    return False, lowest_score

        # If we reach here, all blocks (including partials) are above the threshold
        return True, lowest_score

    def identify_barcodes(self):
        """Detect and store barcodes for this page."""
        self.identify_barcodes_with_zbar()
        self.identify_datamatrices()

    def identify_barcodes_with_zbar(self):
        try:
            from pyzbar import pyzbar
        except:
            logging.debug('Failed to import pyzbar', exc_info=True)
            return
        image_height = self.image.shape[0]
        image_width = self.image.shape[1]
        barcodes = pyzbar.decode(self.image)
        #Add barcode as placehoder
        for barcode in barcodes:
            logging.debug(barcode)
            x = barcode.rect.left
            y = barcode.rect.top
            h = barcode.rect.height
            w = barcode.rect.width
            value = barcode.data.decode("utf-8")
            self.barcodes.append({"x":x, "y":y, "height":h, "width":w, "value":value})
            self.pixel_ignore_areas.append({"x": x, "y": y, "height": h, "width": w})

    def identify_datamatrices(self):
        try:
            from pylibdmtx import pylibdmtx
        except:
            logging.debug('Failed to import pylibdmtx', exc_info=True)
            return
        logging.debug("Identify datamatrices")
        image_height = self.image.shape[0]
        try:
            barcodes = pylibdmtx.decode(self.image, timeout=5000)
        except:
            logging.debug("pylibdmtx could not be loaded",exc_info=True)
            return
        #Add barcode as placehoder
        for barcode in barcodes:
            print(barcode)
            x = barcode.rect.left
            y = image_height - barcode.rect.top - barcode.rect.height
            h = barcode.rect.height
            w = barcode.rect.width
            value = barcode.data.decode("utf-8")
            self.barcodes.append({"x":x, "y":y, "height":h, "width":w, "value":value})
            self.pixel_ignore_areas.append({"x": x, "y:": y, "height": h, "width": w})

    def _process_ignore_area(self, ignore_area: Dict):
        """Process each ignore area based on its type and convert it into pixel-based coordinates."""
        ignore_area_type = ignore_area.get('type')
        
        if ignore_area_type in ['pattern', 'line_pattern', 'word_pattern']:
            self._process_pattern_ignore_area(ignore_area)
        elif ignore_area_type == 'coordinates':
            self._process_coordinates_ignore_area(ignore_area)
        elif ignore_area_type == 'area':
            self._process_area_ignore_area(ignore_area)

    def _process_pattern_ignore_area_from_ocr(self, ignore_area: Dict):
        """Handle pattern-based ignore areas by searching the OCR text for text patterns."""
        import re
        pattern = ignore_area.get('pattern')
        xoffset = int(ignore_area.get('xoffset', 0))
        yoffset = int(ignore_area.get('yoffset', 0))

        # Iterate through text data to identify matching patterns and mark as ignore areas
        n_boxes = len(self.ocr_text_data['text'])
        for j in range(n_boxes):
            raw_text = self.ocr_text_data['text'][j]
            normalized_text = self._normalize_token(raw_text)
            if not normalized_text:
                continue
            match_target = normalized_text.upper()
            if not re.match(pattern, match_target):
                continue

            x, y, w, h = (
                self.ocr_text_data['left'][j],
                self.ocr_text_data['top'][j],
                self.ocr_text_data['width'][j],
                self.ocr_text_data['height'][j],
            )
            text_mask = {
                "x": int(x) - xoffset,
                "y": int(y) - yoffset,
                "width": int(w) + 2 * xoffset,
                "height": int(h) + 2 * yoffset,
            }
            self.pixel_ignore_areas.append(text_mask)

    def _process_pattern_ignore_area_from_pdf(self, ignore_area: Dict):
        import re
        pattern_type = ignore_area.get('type') or ignore_area.get('pattern_type')
        pattern = ignore_area.get('pattern')
        xoffset = int(ignore_area.get('xoffset', 0))
        yoffset = int(ignore_area.get('yoffset', 0))
        search_pattern = re.compile(pattern)
        if pattern_type == "word_pattern":
            if self.pdf_text_words:
                for word in self.pdf_text_words:
                    if search_pattern.match(word[4]):
                        x0, y0, x1, y1 = word[:4]
                        x = x0 * self.dpi / 72
                        y = y0 * self.dpi / 72
                        w = (x1 - x0) * self.dpi / 72
                        h = (y1 - y0) * self.dpi / 72
                        text_mask = {"x": int(x) - xoffset, "y": int(y) - yoffset, "width": int(w) + 2 * xoffset, "height": int(h) + 2 * yoffset}
                        self.pixel_ignore_areas.append(text_mask)
        else:
            if self.pdf_text_dict:
                for block in self.pdf_text_dict['blocks']:
                    if block['type'] == 0:
                        for line in block['lines']:
                            if len(line['spans']) != 0 and search_pattern.match(line['spans'][0]['text']):
                                (x, y, w, h) = (line['bbox'][0]*self.dpi/72, line['bbox'][1]*self.dpi/72,(line['bbox'][2]-line['bbox'][0])*self.dpi/72, (line['bbox'][3]-line['bbox'][1])*self.dpi/72)
                                text_mask = {"x": int(x) - xoffset, "y": int(y) - yoffset, "width": int(w) + 2 * xoffset, "height": int(h) + 2 * yoffset}
                                self.pixel_ignore_areas.append(text_mask)


    def _process_pattern_ignore_area(self, ignore_area: Dict):
        """Handle pattern-based ignore areas by searching the page for text patterns."""
        # import re
        # pattern = ignore_area.get('pattern')
        # xoffset = int(ignore_area.get('xoffset', 0))
        # yoffset = int(ignore_area.get('yoffset', 0))

        if self.ocr_performed:
            self._process_pattern_ignore_area_from_ocr(ignore_area)
        elif self.pdf_text_data or self.pdf_text_dict or self.pdf_text_words:
            self._process_pattern_ignore_area_from_pdf(ignore_area)

    def _process_coordinates_ignore_area(self, ignore_area: Dict):
        """Convert coordinate-based ignore areas into pixel-wise ignore areas."""
        unit = ignore_area.get('unit', 'px')
        x, y, w, h = self._convert_to_pixels(ignore_area, unit)
        self.pixel_ignore_areas.append({"x": x, "y": y, "height": h, "width": w})
    
    def _convert_to_pixels(self, area: Dict, unit: str):
        """Convert dimensions from cm, mm, or px to pixel units."""
        x, y, w, h = int(area['x']), int(area['y']), int(area['width']), int(area['height'])
        if unit == 'mm':
            constant = self.dpi / 25.4
            x, y, w, h = int(x * constant), int(y * constant), int(w * constant), int(h * constant)
        elif unit == 'cm':
            constant = self.dpi / 2.54
            x, y, w, h = int(x * constant), int(y * constant), int(w * constant), int(h * constant)
        return x, y, w, h
        
    def _process_area_ignore_area(self, ignore_area: Dict):
        """Handle area-based ignore areas (e.g., 'top', 'bottom', 'left', 'right') as percentages."""
        page = ignore_area.get('page', 'all')
        # Cast the page number to an integer if it is not 'all'
        if page != 'all':
            page = int(page)
        if page == 'all' or page == self.page_number:
            location = ignore_area.get('location', None)
            percent = int(ignore_area.get('percent', 10))
            x, y, w, h = 0, 0, self.image.shape[1], self.image.shape[0]
            if location == 'top':
                h = int(self.image.shape[0] * percent / 100)
            elif location == 'bottom':
                h = int(self.image.shape[0] * percent / 100)
                y = self.image.shape[0] - h
            elif location == 'left':
                w = int(self.image.shape[1] * percent / 100)
            elif location == 'right':
                w = int(self.image.shape[1] * percent / 100)
                x = self.image.shape[1] - w
            self.pixel_ignore_areas.append({"x": x, "y": y, "width": w, "height": h})

    def _get_text(self, force_ocr: bool = False, ocr_engine: Literal['tesseract', 'east'] = 'tesseract', confidence: int = DEFAULT_CONFIDENCE, tesseract_config: str = TESSERACT_CONFIG, **kwargs):
        """Extract text content from the page image."""
        if force_ocr and not self.ocr_performed:
            self.apply_ocr(ocr_engine=ocr_engine, tesseract_config=tesseract_config)
            return " ".join(self._normalized_ocr_tokens())
        if self.ocr_performed:
            return " ".join(self._normalized_ocr_tokens())
        elif self.pdf_text_words:
            return make_text(self.pdf_text_words).split()
        else:
            self.apply_ocr(ocr_engine=ocr_engine, tesseract_config=tesseract_config)
            return " ".join(self._normalized_ocr_tokens())

    def _get_text_from_area(self, area: Dict, force_ocr: bool = False):
        """Extract text content from a specific area of the page:
        Returns the text content within the specified area.
        An area is defined by a dictionary with keys 'x', 'y', 'width', and 'height'.
        It can also be a tuple with the format (x, y, width, height).
        Units are in pixels by default.
        Optional: Units can be specified as 'mm' or 'cm' via the 'unit' key.
        """
        if force_ocr:
            pdf_text = self._get_text_from_area_using_pdf(area)
            if pdf_text:
                return pdf_text
            return self._get_text_from_area_with_tesseract(area)
        
        try:
            unit = area.get('unit', 'px')
        except:
            unit = 'px'
        area_x, area_y, area_w, area_h  = self._convert_to_pixels(area, unit)


        if self.ocr_performed:
            text = ""
            for i, box in enumerate(self.ocr_text_data['text']):
                x, y, w, h = self.ocr_text_data['left'][i], self.ocr_text_data['top'][i], self.ocr_text_data['width'][i], self.ocr_text_data['height'][i]
                if x >= area_x and y >= area_y and x + w <= area_x + area_w and y + h <= area_y + area_h:
                    text += box + " "
            return text.strip()
        
        elif self.pdf_text_words:
            return self._get_text_from_area_using_pdf(area)
        else:
            return self._get_text_from_area_with_tesseract(area)
        
    def _get_text_from_area_using_pdf(self, area: Dict) -> str:
        if not self.pdf_text_words:
            return ""

        area_x, area_y, area_w, area_h = self._convert_to_pixels(area, area.get('unit', 'px'))
        import fitz
        fitz.TOOLS.set_aa_level(0)
        rect = fitz.Rect(
            area_x * 72 / self.dpi,
            area_y * 72 / self.dpi,
            (area_x + area_w) * 72 / self.dpi,
            (area_y + area_h) * 72 / self.dpi,
        )
        diff_area_ref_words = [
            w for w in self.pdf_text_words if fitz.Rect(w[:4]).intersects(rect)
        ]
        normalized_words = [self._normalize_token(word[4]) for word in diff_area_ref_words]
        normalized_words = [word for word in normalized_words if word]
        return " ".join(normalized_words)

    def _get_text_from_area_with_tesseract(self, area: Dict):
        """Extract text content from a specific area of the page using Tesseract OCR."""
        x, y, w, h = self._convert_to_pixels(area, area.get('unit', 'px'))
        image = self.image[y:y+h, x:x+w]
        # upscale the image if the resolution is too low for OCR
        if self.dpi < MINIMUM_OCR_RESOLUTION:
            scale_factor = MINIMUM_OCR_RESOLUTION / self.dpi
            image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # Add a white border around the image to improve OCR accuracy
        thresholded_image = cv2.copyMakeBorder(thresholded_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        config = f'--psm 11 -l eng'
        text = pytesseract.image_to_string(thresholded_image, config=config)
        return text.strip()

    def _compare_text_content_in_area_with(self, other_page: 'Page', area: Dict, force_ocr: bool = False):
        """Compare text content in a specific area of the page with another page.
        Returns True if the text content in the specified area is the same between the two pages.
        Also returns the text content from the area in both pages.
        Area can be defined by a dictionary with keys 'x', 'y', 'width', and 'height'.
        Or it can be a tuple like (124, 337, 287, 121) where the values are in pixels.

        """
        text_self = self._get_text_from_area(area, force_ocr)
        text_other = other_page._get_text_from_area(area, force_ocr)
        return text_self == text_other, text_self, text_other
    
    def _get_area(self, area: Dict):
        """Extract the specified area from the page image.
        Area is a rectangle object from cv2
        """
        x, y, w, h = self._convert_to_pixels(area, area.get('unit', 'px'))
        return self.image[y:y+h, x:x+w]
    
class DocumentRepresentation:
    def __init__(self, file_path: str, dpi: int = DEFAULT_DPI, ocr_engine: Literal['tesseract', 'east'] = OCR_ENGINE_DEFAULT, tesseract_config: str = TESSERACT_CONFIG, ignore_area_file: Union[str, dict, list] = None, ignore_area: Union[str, dict, list] = None, force_ocr: bool = False, contains_barcodes: bool = False, **kwargs):
        self.file_path = Path(file_path)
        self.dpi = dpi
        self.contains_barcodes = contains_barcodes
        self.pages: List[Page] = []
        self.ocr_engine = ocr_engine
        self.abstract_ignore_areas = []
        self.barcodes = []
        self.path, self.filename= os.path.split(self.file_path)
        self.filename_without_extension, self.extension = os.path.splitext(self.filename)
        self.tmp_directory = tempfile.gettempdir()
        self.ignore_printfactory_envelope=kwargs.get('ignore_printfactory_envelope', False)
        self.printing_papersize=kwargs.get('printing_papersize', 'a4')
        self.load_document()
        self.create_abstract_ignore_areas(ignore_area_file, ignore_area)
        self.create_pixel_based_ignore_areas(force_ocr)
        
        
        if self.contains_barcodes:
            self.identify_barcodes()

    def load_document(self):
        """Load the document, either as an image or a multi-page PDF, into Page objects."""
        if self.file_path.suffix == '.pdf':
            self._load_pdf()
        elif self.file_path.suffix == '.pcl':
            self._load_pcl()
        elif self.file_path.suffix == '.ps':
            self._load_ps()
        else:
            self._load_image()

    def _load_image(self):
        """Load a single image file as a Page object."""
        image = cv2.imread(str(self.file_path))
        # For images, the dpi is always 72 (default for OpenCV)
        self.dpi = 72
        if image is None:
            raise ValueError(f"Cannot load image from {self.file_path}")
        page = Page(image, page_number=1, dpi=self.dpi)
        self.pages.append(page)

    def _load_pdf(self):
        """Load a PDF document, converting each page into a Page object."""
        try:
            import fitz
            fitz.TOOLS.set_aa_level(0)  # PyMuPDF
            doc = fitz.open(str(self.file_path))
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=self.dpi)
                img_data = pix.tobytes("png")
                image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                self.pages.append(Page(image, page_number=page_num + 1, dpi=self.dpi))
                self.pages[-1].pdf_text_data = page.get_text("text")
                self.pages[-1].pdf_text_dict = page.get_text("dict")
                self.pages[-1].pdf_text_words = page.get_text("words")
                self.pages[-1].pdf_text_blocks = page.get_text("blocks")
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required for PDF processing.")

    def _load_pcl(self):
        import subprocess
        import shutil
        import random
        import time
        import tempfile
        from os.path import splitext, split
        import subprocess
        try:
            command = shutil.which('pcl6') or shutil.which('gpcl6win64') or shutil.which('gpcl6win32') or shutil.which('gpcl6')
        except Exception:
            command = None
        if not command:
            raise AssertionError("No pcl6 executable found in path. Please install ghostPCL")
        self.opencv_images = []
        
        resolution = self.dpi
        tic = time.perf_counter()
        output_image_directory = os.path.join(self.tmp_directory, self.filename_without_extension+str(random.randint(100, 999)))
        
        is_exist = os.path.exists(output_image_directory)
        if not is_exist:
            os.makedirs(output_image_directory)
        else:
            shutil.rmtree(output_image_directory)
            os.makedirs(output_image_directory)
        Output_filepath = os.path.join(output_image_directory,'output-%d.png')

        args = [
            command,
            f'-J"@PJL SET PAPER = {self.printing_papersize}"',
            '-dNOPAUSE',
            "-sDEVICE=png16m",
            f"-r{resolution}",
            f"-sOutputFile={Output_filepath}",
            self.file_path
        ]
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        toc = time.perf_counter()
        print(f"Rendering pcl document to Image with ghostPCL performed in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()
        file_num =len(os.listdir(output_image_directory))
        for index in range(file_num):
            if (index == 0 or index == int(file_num-1)) and self.ignore_printfactory_envelope is True:
                print("The printfactory envelope is ignored in page {}".format(index+1))
                continue
            else:
                filename = 'output-' + str(index+1)+'.png'
                image_file =os.path.join(output_image_directory, filename)
                data = cv2.imread(image_file)
                page = Page(data, page_number=str(index+1), dpi=self.dpi)
                
            
                if page is None:
                    raise AssertionError("No OpenCV Image could be created for file {} . Maybe the file is corrupt?".format(self.file_path))
                #self.opencv_images.append(data)
                self.pages.append(page)

        toc = time.perf_counter()
        print(f"Conversion from Image to OpenCV Image performed in {toc - tic:0.4f} seconds")
        shutil.rmtree(output_image_directory) 

    def _load_ps(self):
        import subprocess
        import shutil
        import random
        import tempfile
        import time
        from os.path import splitext, split 
        try:
            command = shutil.which('gs') or shutil.which('gswin64c') or shutil.which('gswin32c') or shutil.which('ghostscript')
        except Exception:
            command = None
        if not command:
            raise AssertionError("No ghostscript executable found in path. Please install ghostscript")
        self.opencv_images = []
        
        resolution = self.dpi
        tic = time.perf_counter()
        output_image_directory = os.path.join(self.tmp_directory, self.filename_without_extension+str(random.randint(100, 999)))
        is_exist = os.path.exists(output_image_directory)
        if not is_exist:
            os.makedirs(output_image_directory)
        else:
            shutil.rmtree(output_image_directory)
            os.makedirs(output_image_directory)
        Output_filepath = os.path.join(output_image_directory, 'output-%d.png')
        args = [
            command,
            f'-sPAPERSIZE={self.printing_papersize}',
            '-dNOPAUSE',
            "-dBATCH",
            "-dSAFER",
            "-sDEVICE=png16m",
            f"-r{resolution}",
            f"-sOutputFile={Output_filepath}",
            self.file_path
        ]
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        toc = time.perf_counter()
        print(f"Rendering ps document to Image with ghostscript performed in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()
        file_num =len(os.listdir(output_image_directory))
        for index in range(file_num):
            filename = 'output-' + str(index+1)+'.png'
            image_file =os.path.join(output_image_directory, filename)
            data = cv2.imread(image_file)
            page = Page(data, page_number=str(index+1), dpi=self.dpi)
            if page is None:
                raise AssertionError("No OpenCV Image could be created for file {} . Maybe the file is corrupt?".format(self.file_path))
            # self.opencv_images.append(data)
            self.pages.append(page)
        
        toc = time.perf_counter()
        print(f"Conversion from Image to OpenCV Image performed in {toc - tic:0.4f} seconds")
        shutil.rmtree(output_image_directory)

    def get_barcodes(self):
        """Return a list of barcodes detected in the document."""
        barcodes = []
        for page in self.pages:
            barcodes.extend(page.barcodes)
        return barcodes
    
    def apply_ocr(self, parallel: bool = False):
        """Apply OCR to each page of the document."""
        if parallel:
            with ThreadPoolExecutor() as executor:
                executor.map(lambda page: page.apply_ocr(self.ocr_engine), self.pages)
        else:
            for page in self.pages:
                page.apply_ocr(self.ocr_engine)

    def extract_text_from_pdf(self) -> str:
        """
        Attempt to extract text directly from a PDF document.
        Returns the text content if available, otherwise returns an empty string.
        """
        if not self.file_path.suffix == '.pdf':
            return ""

        try:
            import fitz
            fitz.TOOLS.set_aa_level(0)  # PyMuPDF
            with fitz.open(str(self.file_path)) as pdf:
                text_content = ""
                for page_num in range(len(pdf)):
                    page = pdf.load_page(page_num)
                    text_content += page.get_text("text")
                return text_content if text_content.strip() else ""
        except Exception as e:
            print(f"Failed to extract text from PDF: {e}")
            return ""

    def compare_with(self, other_doc: 'DocumentRepresentation') -> bool:
        """Compare this document with another document."""
        if len(self.pages) != len(other_doc.pages):
            raise ValueError("Documents have different number of pages.")
        
        # Compare page by page
        for page_self, page_other in zip(self.pages, other_doc.pages):
            if not page_self.compare_with(page_other):
                return False
        return True

    def create_abstract_ignore_areas(self, ignore_area_file: Union[str, dict, list], ignore_area: Union[str, dict, list]):
        """Read ignore areas from the provided file or mask and return a list of ignore areas."""
        if ignore_area_file:
            ignore_area_manager = IgnoreAreaManager(ignore_area_file=ignore_area_file)
            self.abstract_ignore_areas = ignore_area_manager.read_ignore_areas()
        elif ignore_area:
            ignore_area_manager = IgnoreAreaManager(mask=ignore_area)
            self.abstract_ignore_areas = ignore_area_manager.read_ignore_areas()

    def create_pixel_based_ignore_areas(self, force_ocr: bool = False):
        """Apply ignore areas to each page of the document."""

        for page in self.pages:
            # If ignore area is of type pattern, line_pattern, or word_pattern
            # and if filetype is PDF, read text directly from PDF
            # to identify the pattern-based ignore areas
            # If force_ocr is True or page.text_content is not available, apply OCR

            for ignore_area in self.abstract_ignore_areas:
                if ignore_area.get('type') in ['pattern', 'line_pattern', 'word_pattern']:
                    if (force_ocr or not page.pdf_text_data) and not page.ocr_performed:
                        page.apply_ocr(ocr_engine=self.ocr_engine)
                page._process_ignore_area(ignore_area)    
            

    def get_text_from_area(self, area: Dict, force_ocr: bool = False):
        """Extract text content from a specific area of the document."""
        text_content = ""
        for page in self.pages:
            text_content += page._get_text_from_area(area, force_ocr) + " "
        return text_content.strip()

    def get_text(self, force_ocr: bool = False, tesseract_config: str = TESSERACT_CONFIG):
        """Extract text content from the document."""
        # If doc is pdf, extract text directly from pdf
        text_content = ""
        if force_ocr:
            for page in self.pages:
                text_content += page._get_text(force_ocr, ocr_engine=self.ocr_engine, tesseract_config=tesseract_config) + " "
            return text_content.strip()
        if self.file_path.suffix == '.pdf':
            text_content = self.extract_text_from_pdf()
            return text_content
        else:
            # If OCR is not forced, extract text from the OCR data
            for page in self.pages:
                text_content += page._get_text(force_ocr, ocr_engine=self.ocr_engine, tesseract_config=tesseract_config) + " "
        return text_content.strip()

    def identify_barcodes(self):
        """Detect barcodes in all pages."""
        for page in self.pages:
            page.identify_barcodes()

    def save_images(self, output_dir: str):
        """Save all pages as images to the specified directory."""
        os.makedirs(output_dir, exist_ok=True)
        for page in self.pages:
            output_path = os.path.join(output_dir, f'page_{page.page_number}.png')
            cv2.imwrite(output_path, page.image)

    def compare_visuals(self, other_doc: 'DocumentRepresentation'):
        """Compare images between two documents."""
        return self.compare_with(other_doc)

    def assign_ignore_areas_to_pages(self, ignore_areas: List[Dict]):
        """Assign each ignore area to the corresponding page."""
        for page in self.pages:
            for ignore_area in ignore_areas:
                if ignore_area.get('page') == page.page_number or ignore_area.get('page') == 'all':
                    page.ignore_areas.append(ignore_area)
    

    def identify_barcodes(self):
        """Detect barcodes in all pages."""
        for page in self.pages:
            page.identify_barcodes()

def make_text(words):
    """Return textstring output of get_text("words").
    Word items are sorted for reading sequence left to right,
    top to bottom.
    """
    line_dict = {}  # key: vertical coordinate, value: list of words
    words.sort(key=lambda w: w[0])  # sort by horizontal coordinate
    for w in words:  # fill the line dictionary
        y1 = round(w[3], 1)  # bottom of a word: don't be too picky!
        word = w[4]  # the text of the word
        line = line_dict.get(y1, [])  # read current line content
        line.append(word)  # append new word
        line_dict[y1] = line  # write back to dict
    lines = list(line_dict.items())
    lines.sort()  # sort vertically
    return "\n".join([" ".join(line[1]) for line in lines])
