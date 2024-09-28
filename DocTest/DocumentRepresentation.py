import os
import cv2
import pytesseract
import numpy as np
import json
from skimage import metrics
from typing import List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from pytesseract import Output
from DocTest.IgnoreAreaManager import IgnoreAreaManager
from DocTest.config import DEFAULT_DPI, OCR_ENGINE_DEFAULT, DEFAULT_CONFIDENCE, MINIMUM_OCR_RESOLUTION, ADD_PIXELS_TO_IGNORE_AREA
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

    def apply_ocr(self, ocr_engine: str = OCR_ENGINE_DEFAULT, confidence: int = DEFAULT_CONFIDENCE):
        """Perform OCR on the page image."""
        # re-scale the image to a standard resolution for OCR if needed
        if self.dpi < MINIMUM_OCR_RESOLUTION:
            original_image = self.image.copy()
            scale_factor = MINIMUM_OCR_RESOLUTION / self.dpi
            self.image = cv2.resize(self.image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            self.image_rescaled_for_ocr = True
        if ocr_engine == "tesseract":
            config = f'--psm 11 -l eng'
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            self.ocr_text_data = pytesseract.image_to_data(thresholded_image, config=config, output_type=Output.DICT)
            self.ocr_performed = True
        elif ocr_engine == "east":
            # Placeholder for EAST OCR logic
            pass
        if self.image_rescaled_for_ocr:
            self.image = original_image

    def get_area(self, area: Dict):
        """Gets the area of the image specified by the coordinates."""
        x, y, w, h = area['x'], area['y'], area['width'], area['height']
        return self.image[y:y+h, x:x+w]

    def rescale_image_for_ocr(self):
        pass

    def get_text_content(self):
        """Return the OCR text content."""
        return self.ocr_text_data['text'] if self.ocr_text_data else ""

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

    def compare_with(self, other_page: 'Page', threshold: float = 0.99):
        """
        Compare this page with another page and return a tuple of (similarity_result, diff_image).
        
        | =Arguments= | =Description= |
        | ``other_page`` | Another Page object to compare against. |
        | ``threshold`` | The SSIM threshold to determine similarity. |
        """
        if self.dpi != other_page.dpi:
            raise ValueError(f"Page DPI mismatch: {self.dpi} vs {other_page.dpi}")
        
        if self.pixel_ignore_areas:
            other_page.pixel_ignore_areas = self.pixel_ignore_areas
            self.image = self.get_image_with_ignore_areas()
            other_page.image = other_page.get_image_with_ignore_areas()

        # Quick check: if the images are of different sizes, they are not similar
        if self.image.shape != other_page.image.shape:
            return False, None, None, None
        
        # Quick check: if the images are identical, they are similar
        if np.array_equal(self.image, other_page.image):
            return True, None, None, None

        # Perform SSIM (Structural Similarity Index) comparison and get the diff image
        gray_self = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray_other = cv2.cvtColor(other_page.image, cv2.COLOR_BGR2GRAY)
        score, diff = metrics.structural_similarity(gray_self, gray_other, full=True)

        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        absolute_diff = cv2.absdiff(gray_self, gray_other)

        # Return a tuple: whether the pages are similar, and the difference image
        return score >= (1.0 - threshold), diff, thresh, absolute_diff

    def identify_barcodes(self):
        """Detect and store barcodes for this page."""
        # Placeholder for barcode detection logic using OpenCV or pyzbar
        pass

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
            if int(self.ocr_text_data.get('conf', [0])[j]) > DEFAULT_CONFIDENCE and re.match(pattern, self.ocr_text_data['text'][j]):
                x, y, w, h = self.ocr_text_data['left'][j], self.ocr_text_data['top'][j], self.ocr_text_data['width'][j], self.ocr_text_data['height'][j]
                if self.image_rescaled_for_ocr:
                    pixel_factor = self.dpi / MINIMUM_OCR_RESOLUTION
                    (x, y, w, h) = (int(x * pixel_factor), int(y * pixel_factor), int(w * pixel_factor), int(h * pixel_factor))
                text_mask = {"x": int(x) - xoffset, "y": int(y) - yoffset, "width": int(w) + 2 * xoffset, "height": int(h) + 2 * yoffset}
                self.pixel_ignore_areas.append(text_mask)

    def _process_pattern_ignore_area_from_pdf(self, ignore_area: Dict):
        import re
        pattern_type = ignore_area.get('pattern_type')
        pattern = ignore_area.get('pattern')
        xoffset = int(ignore_area.get('xoffset', 0))
        yoffset = int(ignore_area.get('yoffset', 0))
        search_pattern = re.compile(pattern)
        if pattern_type == "word_pattern":
            if self.pdf_text_words:
                for word in self.pdf_text_words:
                    if search_pattern.match(word[4]):
                        (x, y, w, h) = (word[0]*self.dpi/72, word[1]*self.dpi/72, word[2]*self.dpi/72, word[3]*self.dpi/72)
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

    def _get_text_from_area(self, area: Dict, force_ocr: bool = False):
        """Extract text content from a specific area of the page:
        Returns the text content within the specified area.
        An area is defined by a dictionary with keys 'x', 'y', 'width', and 'height'.
        It can also be a tuple with the format (x, y, width, height).
        Units are in pixels by default.
        Optional: Units can be specified as 'mm' or 'cm' via the 'unit' key.
        """
        if force_ocr:
        # Shortcut: if OCR is forced, extract text using Tesseract
            return self._get_text_from_area_with_tesseract(area)
        
        try:
            unit = area.get('unit', 'px')
        except:
            unit = 'px'
        area_x, area_y, area_h, area_w  = self._convert_to_pixels(area, unit)


        if self.ocr_performed:
            text = ""
            for i, box in enumerate(self.ocr_text_data['text']):
                x, y, w, h = self.ocr_text_data['left'][i], self.ocr_text_data['top'][i], self.ocr_text_data['width'][i], self.ocr_text_data['height'][i]
                if x >= area_x and y >= area_y and x + w <= area_x + area_w and y + h <= area_y + area_h:
                    text += box + " "
            return text.strip()
        
        elif self.pdf_text_words:
            text = ""
            import fitz
            rect = fitz.Rect(
                            area_x*72/self.dpi, area_y*72/self.dpi, (area_x+area_w)*72/self.dpi, (area_y+area_h)*72/self.dpi)
            diff_area_ref_words = [
                            w for w in self.pdf_text_words if fitz.Rect(w[:4]).intersects(rect)]
            return make_text(diff_area_ref_words).split()
        else:
            self.apply_ocr()
            text = ""
            for i, box in enumerate(self.ocr_text_data['text']):
                x, y, w, h = self.ocr_text_data['left'][i], self.ocr_text_data['top'][i], self.ocr_text_data['width'][i], self.ocr_text_data['height'][i]
                if x >= area_x and y >= area_y and x + w <= area_x + area_w and y + h <= area_y + area_h:
                    text += box + " "
            return text.strip()
        
    def _get_text_from_area_with_tesseract(self, area: Dict):
        """Extract text content from a specific area of the page using Tesseract OCR."""
        x, y, w, h = self._convert_to_pixels(area, area.get('unit', 'px'))
        image = self.image[y:y+h, x:x+w]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
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
    def __init__(self, file_path: str, dpi: int = DEFAULT_DPI, ocr_engine: str = OCR_ENGINE_DEFAULT, ignore_area_file: Union[str, dict, list] = None, ignore_area: Union[str, dict, list] = None, force_ocr: bool = False):
        self.file_path = file_path
        self.dpi = dpi
        self.pages: List[Page] = []
        self.ocr_engine = ocr_engine
        self.abstract_ignore_areas = []
        self.pixel_ignore_areas = []
        self.load_document()
        self.create_abstract_ignore_areas(ignore_area_file, ignore_area)
        self.create_pixel_based_ignore_areas(force_ocr)

    def load_document(self):
        """Load the document, either as an image or a multi-page PDF, into Page objects."""
        if self.file_path.endswith('.pdf'):
            self._load_pdf()
        else:
            self._load_image()

    def _load_image(self):
        """Load a single image file as a Page object."""
        image = cv2.imread(self.file_path)
        # For images, the dpi is always 72 (default for OpenCV)
        self.dpi = 72
        if image is None:
            raise ValueError(f"Cannot load image from {self.file_path}")
        page = Page(image, page_number=1, dpi=self.dpi)
        self.pages.append(page)

    def _load_pdf(self):
        """Load a PDF document, converting each page into a Page object."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(self.file_path)
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
        if not self.file_path.endswith('.pdf'):
            return ""

        try:
            import fitz  # PyMuPDF
            with fitz.open(self.file_path) as pdf:
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
            

    def get_text_content(self) -> str:
        """Extract the text content from all pages."""
        return "\n".join([page.get_text_content() for page in self.pages])

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
