import base64
from typing import Union, Optional, Any, Literal
import uuid
from DocTest.DocumentRepresentation import DocumentRepresentation, Page
from DocTest.Downloader import is_url, download_file_from_url
from robot.api.deco import keyword, library
from pathlib import Path
import os
import json
import cv2
import shutil
import numpy as np
import imutils
from robot.libraries.BuiltIn import BuiltIn
from DocTest.IgnoreAreaManager import IgnoreAreaManager  # Import IgnoreAreaManager
from assertionengine import verify_assertion, AssertionOperator, Formatter
import logging

LOG = logging.getLogger(__name__)

@library
class VisualTest:
    ROBOT_LIBRARY_VERSION = 1.0
    DPI_DEFAULT = 200
    OCR_ENGINE_DEFAULT = 'tesseract'

    def __init__(self, threshold: float = 0.0, dpi: int = DPI_DEFAULT, take_screenshots: bool = False, show_diff: bool = False, 
                 ocr_engine: Literal["tesseract", "east"] = OCR_ENGINE_DEFAULT, screenshot_format: str = 'jpg', embed_screenshots: bool = False, force_ocr: bool = False, watermark_file: str =None,   **kwargs):
        self.threshold = threshold
        self.dpi = dpi
        self.take_screenshots = take_screenshots
        self.show_diff = show_diff
        self.ocr_engine = ocr_engine
        self.screenshot_format = screenshot_format if screenshot_format in ['jpg', 'png'] else 'jpg'
        self.embed_screenshots = embed_screenshots
        self.screenshot_dir = Path('screenshots')
        self.watermark_file = watermark_file
        built_in = BuiltIn()
        self.force_ocr = force_ocr
        try:
            self.output_directory = built_in.get_variable_value('${OUTPUT DIR}')
            self.reference_run = built_in.get_variable_value('${REFERENCE_RUN}', False)
            self.PABOTQUEUEINDEX = built_in.get_variable_value('${PABOTQUEUEINDEX}')
        except:
            print("Robot Framework is not running")
            self.output_directory = Path.cwd()
            self.reference_run = False
            self.PABOTQUEUEINDEX = None

        self.screenshot_path = self.output_directory / self.screenshot_dir
        

    @keyword
    def compare_images(self, reference_image: str, candidate_image: str, placeholder_file: Union[str, dict, list] = None, 
                       check_text_content: bool = False, move_tolerance: int = None, contains_barcodes: bool = False, 
                       watermark_file: str = None, ignore_watermarks: bool=None, force_ocr: bool = False, DPI: int = None, resize_candidate: bool = False, 
                       blur: bool = False, threshold: float = None, mask: Union[str, dict, list] = None, get_pdf_content: bool = False,  **kwargs):
        """Compares the documents/images ``reference_image`` and ``test_image``.

        Result is passed if no visual differences are detected.

        | =Arguments= | =Description= |
        | ``reference_image`` | Path or URL of the Reference Image/Document, your expected result. May be .pdf, .ps, .pcl or image files |
        | ``test_image`` | Path or URL of the Candidate Image/Document, that's the one you want to test. May be .pdf, .ps, .pcl or image files |
        | ``placeholder_file`` | Path to a ``.json`` which defines areas that shall be ignored for comparison. Those parts will be replaced with solid placeholders  |
        | ``mask`` | Same purpose as ``placeholder_file`` but instead of a file path, this is either ``json`` , a ``dict`` , a ``list`` or a ``string`` which defines the areas to be ignored  |
        | ``check_text_content`` | In case of visual differences: Is it acceptable, if only the text content in the different areas is equal |
        | ``move_tolerance`` | In case of visual differences: Is is acceptable, if only parts in the different areas are moved by ``move_tolerance`` pixels  |
        | ``contains_barcodes`` | Shall the image be scanned for barcodes and shall their content be checked (currently only data matrices are supported) |
        | ``get_pdf_content`` | Only relevant in case of using ``move_tolerance`` and ``check_text_content``: Shall the PDF Content like Texts and Boxes be used for calculations |
        | ``force_ocr`` | Always use OCR to find Texts in Images, even for PDF Documents |
        | ``DPI`` | Resolution in which documents are rendered before comparison |
        | ``watermark_file`` | Path to an image/document or a folder containing multiple images. They shall only contain a ```solid black`` area of the parts that shall be ignored for visual comparisons |
        | ``ignore_watermarks`` | Ignores a very special watermark in the middle of the document |
        | ``ocr_engine`` | Use ``tesseract`` or ``east`` for Text Detection and OCR |
        | ``resize_candidate`` | Allow visual comparison, even of documents have different sizes |
        | ``blur`` | Blur the image before comparison to reduce visual difference caused by noise |
        | ``threshold`` | Threshold for visual comparison between 0.0000 and 1.0000 . Default is 0.0000. Higher values mean more tolerance for visual differences. |
        | ``**kwargs`` | Everything else |
        

        Examples:
        | `Compare Images`   reference.pdf   candidate.pdf                                  #Performs a pixel comparison of both files
        | `Compare Images`   reference.pdf (not existing)    candidate.pdf                  #Will always return passed and save the candidate.pdf as reference.pdf
        | `Compare Images`   reference.pdf   candidate.pdf   placeholder_file=mask.json     #Performs a pixel comparison of both files and excludes some areas defined in mask.json
        | `Compare Images`   reference.pdf   candidate.pdf   contains_barcodes=${true}      #Identified barcodes in documents and excludes those areas from visual comparison. The barcode data will be checked instead
        | `Compare Images`   reference.pdf   candidate.pdf   check_text_content${true}      #In case of visual differences, the text content in the affected areas will be identified using OCR. If text content it equal, the test is considered passed
        | `Compare Images`   reference.pdf   candidate.pdf   move_tolerance=10              #In case of visual differences, it is checked if difference is caused only by moved areas. If the move distance is within 10 pixels the test is considered as passed. Else it is failed
        | `Compare Images`   reference.pdf   candidate.pdf   check_text_content=${true}   get_pdf_content=${true}   #In case of visual differences, the text content in the affected areas will be read directly from  PDF (not OCR). If text content it equal, the test is considered passed
        | `Compare Images`   reference.pdf   candidate.pdf   watermark_file=watermark.pdf     #Provides a watermark file as an argument. In case of visual differences, watermark content will be subtracted
        | `Compare Images`   reference.pdf   candidate.pdf   watermark_file=${CURDIR}${/}watermarks     #Provides a watermark folder as an argument. In case of visual differences, all watermarks in folder will be subtracted
        | `Compare Images`   reference.pdf   candidate.pdf   move_tolerance=10   get_pdf_content=${true}   #In case of visual differences, it is checked if difference is caused only by moved areas. Move distance is identified directly from PDF data. If the move distance is within 10 pixels the test is considered as passed. Else it is failed
        
        Special Examples with ``mask``:
        | `Compare Images`   reference.pdf   candidate.pdf   mask={"page": "all", type: "coordinate", "x": 0, "y": 0, "width": 100, "height": 100}     #Excludes a rectangle from comparison

        | ${top_mask}    Create Dictionary    page=1    type=area    location=top    percent=10
        | ${bottom_mask}    Create Dictionary    page=all    type=area    location=bottom    percent=10
        | ${masks}    Create List    ${top_mask}    ${bottom_mask}
        | `Compare Images`     reference.pdf    candidate.pdf    mask=${masks}      #Excludes an area and a rectangle from comparison

        | ${mask}    Create Dictionary    page=1    type=coordinate    x=0    y=0    width=100    height=100
        | `Compare Images`    reference.pdf    candidate.pdf    mask=${mask}    #Excludes a rectangle from comparison

        | `Compare Images`    reference.pdf    candidate.pdf    mask=top:10;bottom:10   #Excludes two areas top and bottom with 10% from comparison
        """
        # Download files if URLs are provided
        if is_url(reference_image):
            reference_image = download_file_from_url(reference_image)
        if is_url(candidate_image):
            candidate_image = download_file_from_url(candidate_image)

        # Set DPI and threshold if provided
        dpi = DPI if DPI else self.dpi
        threshold = threshold if threshold is not None else self.threshold

        if watermark_file is None:
            watermark_file = self.watermark_file
        if ignore_watermarks is None:
            ignore_watermarks = os.getenv('IGNORE_WATERMARKS', False)

        # Load reference and candidate documents
        reference_doc = DocumentRepresentation(reference_image, dpi=dpi, ocr_engine=self.ocr_engine, ignore_area_file=placeholder_file, ignore_area=mask)
        candidate_doc = DocumentRepresentation(candidate_image, dpi=dpi, ocr_engine=self.ocr_engine)            

        watermarks = []

        # Apply ignore areas if provided
        abstract_ignore_areas = None
       
        detected_differences = []
        # Compare visual content through the Page class
        for ref_page, cand_page in zip(reference_doc.pages, candidate_doc.pages):

            # Resize the candidate page if needed
            if resize_candidate and ref_page.image.shape != cand_page.image.shape:
                cand_page.image = cv2.resize(cand_page.image, (ref_page.image.shape[1], ref_page.image.shape[0]))
            
            # Check if dimensions are different
            if ref_page.image.shape != cand_page.image.shape:
                detected_differences.append((ref_page, cand_page, "Image dimensions are different."))
                continue

            similar, diff, thresh, absolute_diff, score = ref_page.compare_with(cand_page, threshold=threshold, blur=blur)

            if self.take_screenshots:
                # Save original images to the screenshot directory and add them to the Robot Framework log
                # But add them next to each other in the log
                # Do a np.concatenate with axis=1 to add them next to each other
                combined_image = np.concatenate((ref_page.image, cand_page.image), axis=1)
                self.add_screenshot_to_log(combined_image, suffix='_combined', original_size=False)

            if not similar and watermark_file:
                if watermarks == []:
                    watermarks = load_watermarks(watermark_file)
                for mask in watermarks:    
                    if (mask.shape[0] != ref_page.image.shape[0] or mask.shape[1] != ref_page.image.shape[1]):
                        # Resize mask to match thresh
                        mask = cv2.resize(mask, (ref_page.image.shape[1], ref_page.image.shape[0]))

                    mask_inv = cv2.bitwise_not(mask)
                    # dilate the mask to account for slight misalignments
                    mask_inv = cv2.dilate(mask_inv, None, iterations=2)
                    result = cv2.subtract(absolute_diff, mask_inv)
                    if cv2.countNonZero(cv2.subtract(absolute_diff, mask_inv)) == 0 or cv2.countNonZero(cv2.subtract(thresh, mask_inv)) == 0:
                        similar = True
                        print(
                            "A watermark file was provided. After removing watermark area, both images are equal")

                

            if check_text_content and not similar:
                similar = True
                # Create two new Page objects which only contain absolute differences
                # Do a simple cv2.absdiff to get the absolute differences between the two images

                



                # If the images are not similar, we need to compare text content
                # Only compare the text content in the areas that have differences
                # For that, the rectangles around the differences are needed
                diff_rectangles = self.get_diff_rectangles(absolute_diff)
                # Compare text content only in the areas that have differences
                for rect in diff_rectangles:

                    same_text, ref_area_text, cand_area_text = ref_page._compare_text_content_in_area_with(cand_page, rect, force_ocr)
                    # Save the reference and candidate areas as images and add them to the log
                    reference_area = ref_page.get_area(rect)
                    candidate_area = cand_page.get_area(rect)
                    self.add_screenshot_to_log(reference_area, suffix='_reference_area', original_size=False)
                    self.add_screenshot_to_log(candidate_area, suffix='_candidate_area', original_size=False)
                    
                    if not same_text:
                        similar = False
                        # Add log message with the text content differences
                        # Add screenshots to the log of the reference and candidate areas

                        print(f"Text content in the area {rect} differs:\n\nReference Text:\n{ref_area_text}\n\nCandidate Text:\n{cand_area_text}")
  
                    else:
                        print(f"Visual differences in the area {rect} but text content is the same.")
                        print(f"Reference Text:\n{ref_area_text}\n\nCandidate Text:\n{cand_area_text}")

            if move_tolerance and not similar:
                
                if get_pdf_content:
                    import fitz
                    similar = True
                    ref_words = ref_page.pdf_text_words
                    cand_words = cand_page.pdf_text_words

                    # If no words are fount, proceed with nornmal tolerance check and set check_pdf_content to False
                    if len(ref_words) == 0 or len(cand_words) == 0:
                        check_pdf_content = False
                        print("No pdf layout elements found. Proceeding with normal tolerance check.")

                    diff_rectangles = self.get_diff_rectangles(absolute_diff)
                    c = 0
                    for diff_rect in diff_rectangles:
                        c += 1
                        # Get Values for x, y, w, h
                        (x, y, w, h) = diff_rect['x'], diff_rect['y'], diff_rect['width'], diff_rect['height']

                        rect = fitz.Rect(
                            x*72/self.dpi, y*72/self.dpi, (x+w)*72/self.dpi, (y+h)*72/self.dpi)
                        diff_area_ref_words = [
                            w for w in ref_words if fitz.Rect(w[:4]).intersects(rect)]
                        diff_area_cand_words = [
                            w for w in cand_words if fitz.Rect(w[:4]).intersects(rect)]
                        # diff_area_ref_words = make_text(diff_area_ref_words)
                        # diff_area_cand_words = make_text(diff_area_cand_words)
                        diff_area_reference = ref_page.get_area(diff_rect)
                        diff_area_candidate = cand_page.get_area(diff_rect)
                        self.add_screenshot_to_log(
                            diff_area_reference, "_page_" + str(ref_page.page_number+1) + "_diff_area_reference_"+str(c))
                        self.add_screenshot_to_log(
                            diff_area_candidate, "_page_" + str(ref_page.page_number+1) + "_diff_area_test_"+str(c))

                        if len(diff_area_ref_words) != len(diff_area_cand_words):
                            similar = False
                            print("The identified pdf layout elements are different",
                                  diff_area_ref_words, diff_area_cand_words)
                            raise AssertionError('The compared images are different.')
                        else:
                            for ref_Item, cand_Item in zip(diff_area_ref_words, diff_area_cand_words):
                                if ref_Item == cand_Item:
                                    pass

                                elif str(ref_Item[4]).strip() == str(cand_Item[4]).strip():
                                    left_moved = abs(
                                        ref_Item[0]-cand_Item[0])*self.dpi/72
                                    top_moved = abs(
                                        ref_Item[1]-cand_Item[1])*self.dpi/72
                                    right_moved = abs(
                                        ref_Item[2]-cand_Item[2])*self.dpi/72
                                    bottom_moved = abs(
                                        ref_Item[3]-cand_Item[3])*self.dpi/72
                                    print("Checking pdf elements",
                                          ref_Item, cand_Item)

                                    if int(left_moved) > int(move_tolerance) or int(top_moved) > int(move_tolerance) or int(right_moved) > int(move_tolerance) or int(bottom_moved) > int(move_tolerance):
                                        print("Image section moved ", left_moved,
                                              top_moved, right_moved, bottom_moved, " pixels")
                                        print(
                                            "This is outside of the allowed range of ", move_tolerance, " pixels")
                                        similar = False
                                        self.add_screenshot_to_log(self.blend_two_images(
                                            diff_area_reference, diff_area_candidate), "_diff_area_blended")
                                        raise AssertionError('The compared images are different.')
                                    else:
                                        print("Image section moved ", left_moved,
                                              top_moved, right_moved, bottom_moved, " pixels")
                                        print(
                                            "This is within the allowed range of ", move_tolerance, " pixels")
                                        self.add_screenshot_to_log(self.blend_two_images(
                                            diff_area_reference, diff_area_candidate), "_diff_area_blended")

                
                else:
                    # If the images are not similar, check if the different areas are only moved within the move_tolerance
                    # If the areas are moved within the tolerance, the images are considered similar
                    # If the areas are moved outside the tolerance, the images are considered different
                    # The move_tolerance is the maximum number of pixels the areas can be moved
                    similar = True
                    diff_rectangles = self.get_diff_rectangles(absolute_diff)
                    for rect in diff_rectangles:
                        # Check if the area is moved within the tolerance
                        reference_area = ref_page.get_area(rect)
                        candidate_area = cand_page.get_area(rect)
                        
                        try:
                            # Find the position of the candidate area in the reference area
                            # Use multiple detection methods to find the position
                            # First use the simple template matching method
                            # If no result is found, use the ORB or SIFT method

    #                        result = self.find_partial_image_position(reference_area, candidate_area, threshold=0.1, detection="template")
                            result = self.find_partial_image_position(reference_area, candidate_area, threshold=0.1, detection="sift")

                            if result:
                                if 'distance' in result:
                                    distance = int(result["distance"])
                                # Check if result is a dictuinory with pt1 and pt2
                                if "pt1" in result and "pt2" in result:
                                    pt1 = result["pt1"]
                                    pt2 = result["pt2"]
                                    distance = int(np.sqrt(np.sum((np.array(pt1) - np.array(pt2))**2)))
                                    
                                if distance > move_tolerance:
                                    similar = False
                                    print(f"Area {rect} is moved more than {move_tolerance} pixels.")
                                    self.add_screenshot_to_log(self.blend_two_images(reference_area, candidate_area), suffix='_moved_area', original_size=False)
                                else:
                                    print(f"Area {rect} is moved {distance} pixels.")
                        except Exception as e:
                            print(f"Could not compare areas: {e}")
                            similar = False
                            self.add_screenshot_to_log(self.blend_two_images(reference_area, candidate_area), suffix='_moved_area', original_size=False)
                            break
                
                            
                        



            if not similar:
                # Save original images to the screenshot directory and add them to the Robot Framework log
                # But add them next to each other in the log
                # Do a np.concatenate with axis=1 to add them next to each other
                combined_image = np.concatenate((ref_page.image, cand_page.image), axis=1)
                self.add_screenshot_to_log(combined_image, suffix='_combined', original_size=False)

                # Generate side-by-side image highlighting differences using the SSIM diff image
                reference_img, candidate_img, _ = self.get_images_with_highlighted_differences(thresh, ref_page.image, cand_page.image)
                
                combined_image_with_differeces = np.concatenate((reference_img, candidate_img), axis=1)
                # Add the side-by-side comparison with differences to the Robot Framework log
                self.add_screenshot_to_log(combined_image_with_differeces, suffix='_combined_with_diff', original_size=False)

                # Add absolute difference image to the log
                self.add_screenshot_to_log(absolute_diff, suffix='_absolute_diff', original_size=False)

                detected_differences.append((ref_page, cand_page, f"Visual differences detected. SSIM score: {score:.20f}"))

        for ref_page, cand_page, message in detected_differences:
            print(message)
            self._raise_comparison_failure()

        print("Images/Document comparison passed.")

    @keyword
    def get_text_from_area(self, document: str, area: Union[str, dict, list] = None, assertion_operator: Optional[AssertionOperator] = None,
        assertion_expected: Any = None,
        message: str = None):
        """Get the text content of a specific area in a document.

        The area can be defined as a string, a dictionary or a list of dictionaries.
        If the area is a string, it must be a JSON string.
        If the area is a dictionary, it must have the keys 'x', 'y', 'width' and 'height'.
        If the area is a list of dictionaries, each dictionary must have the keys 'x', 'y', 'width' and 'height'.
        The area is defined in pixels.

        | =Arguments= | =Description= |
        | ``document`` | Path or URL of the document |
        | ``area`` | Area to extract the text from. Can be a string, a dictionary or a list of dictionaries. |
        | ``assertion_operator`` | Assertion operator to be used. |
        | ``assertion_expected`` | Expected value for the assertion. |
        | ``message`` | Message to be displayed in the log. |

        Examples:
        | `Get Text From Area` | document.pdf | {"x": 100, "y": 100, "width": 200, "height": 300} | == | "Expected text" | # Get the text content of the area |
        | `Get Text From Area` | document.pdf | [{"x": 100, "y": 100, "width": 200, "height": 300}, {"x": 300, "y": 300, "width": 200, "height": 300}] | == | ["Expected text 1", "Expected text 2"] | # Get the text content of multiple areas |

        
        """
        if is_url(document):
            document = download_file_from_url(document)
        
        # Load the document
        doc = DocumentRepresentation(document, dpi=self.dpi, ocr_engine=self.ocr_engine)

        # Convert area to dictionary if it's a string
        if isinstance(area, str):
            area = json.loads(area)

        # Get the text content of the area
        text = doc.get_text_from_area(area=area)

    @keyword
    def get_text_from_document(self, document: str, assertion_operator: Optional[AssertionOperator] = None,
        assertion_expected: Any = None,
        message: str = None):
        """Get the text content of a document.

        | =Arguments= | =Description= |
        | ``document`` | Path or URL of the document |
        | ``assertion_operator`` | Assertion operator to be used. |
        | ``assertion_expected`` | Expected value for the assertion. |
        | ``message`` | Message to be displayed in the log. |

        Examples:
        | `Get Text From Document` | document.pdf | == | "Expected text" | # Get the text content of the document |
        | `Get Text From Document` | document.pdf | != | "Unexpected text" | # Get the text content of the document and check if it's not equal to the expected text |
        | ${text} | `Get Text From Document` | document.pdf | # Get the text content of the document and store it in a variable |
        | `Should Be Equal` | ${text} | "Expected text" | # Check if the text content is equal to the expected text |
        """
        return self.get_text(document, assertion_operator, assertion_expected, message)

    @keyword
    def get_text(self, document: str, assertion_operator: Optional[AssertionOperator] = None,
        assertion_expected: Any = None,
        message: str = None):
        
        """Get the text content of a document.

        | =Arguments= | =Description= |
        | ``document`` | Path or URL of the document |
        | ``assertion_operator`` | Assertion operator to be used. |
        | ``assertion_expected`` | Expected value for the assertion. |
        | ``message`` | Message to be displayed in the log. |

        Examples:
        | `Get Text` | document.pdf | == | "Expected text" | # Get the text content of the document |
        | `Get Text` | document.pdf | != | "Unexpected text" | # Get the text content of the document and check if it's not equal to the expected text |
        | ${text} | `Get Text` | document.pdf | # Get the text content of the document and store it in a variable |
        | `Should Be Equal` | ${text} | "Expected text" | # Check if the text content is equal to the expected text |
        """
        if is_url(document):
            document = download_file_from_url(document)
        
        # Load the document
        doc = DocumentRepresentation(document, dpi=self.dpi, ocr_engine=self.ocr_engine)

        # Get the text content of the document
        text = doc.get_text(force_ocr=self.force_ocr)
        return verify_assertion(text, assertion_operator, assertion_expected, message)

    @keyword    
    def set_ocr_engine(self, ocr_engine: Literal["tesseract", "east"]):
        """Set the OCR engine to be used for text extraction.

        | =Arguments= | =Description= |
        | ``ocr_engine`` | OCR engine to be used. Options are ``tesseract`` and ``east``. |

        Examples:
        | `Set OCR Engine` | tesseract | # Set the OCR engine to Tesseract |
        | `Set OCR Engine` | east | # Set the OCR engine to EAST |
        
        """
        self.ocr_engine = ocr_engine

    @keyword
    def set_dpi(self, dpi: int):
        """Set the DPI to be used for image processing
        
        | =Arguments= | =Description= |
        | ``dpi`` | DPI to be used for image processing. |

        Examples:
        | `Set DPI` | 300 | # Set the DPI to 300 |

        """
        self.dpi = dpi
    
    @keyword
    def set_threshold(self, threshold: float):
        """Set the threshold for image comparison.
        
        | =Arguments= | =Description= |
        | ``threshold`` | Threshold for image comparison. |

        Examples:
        | `Set Threshold` | 0.1 | # Set the threshold to 0.1 |
        
        """
        self.threshold = threshold
    
    @keyword
    def set_screenshot_format(self, screenshot_format: str):
        """Set the format of the screenshots to be saved.
        
        | =Arguments= | =Description= |
        | ``screenshot_format`` | Format of the screenshots to be saved. Options are ``jpg`` and ``png``. |

        Examples:
        | `Set Screenshot Format` | jpg | # Set the screenshot format to jpg |
        | `Set Screenshot Format` | png | # Set the screenshot format to png |
        """
        self.screenshot_format = screenshot_format
    
    @keyword
    def set_embed_screenshots(self, embed_screenshots: bool):
        """Set whether to embed screenshots in the log."""
        self.embed_screenshots = embed_screenshots
    
    @keyword
    def set_take_screenshots(self, take_screenshots: bool):
        """Set whether to take screenshots during image comparison.
        
        | =Arguments= | =Description= |
        | ``take_screenshots`` | Whether to take screenshots during image comparison. |

        Examples:
        | `Set Take Screenshots` | True | # Set to take screenshots during image comparison |
        | `Set Take Screenshots` | False | # Set not to take screenshots during image comparison |
        
        """
        self.take_screenshots = take_screenshots
    
    @keyword
    def set_show_diff(self, show_diff: bool):
        """Set whether to show diff screenshot in the images during comparison.
        
        | =Arguments= | =Description= |
        | ``show_diff`` | Whether to show diff screenshot in the images during comparison. |

        Examples:
        | `Set Show Diff` | True | # Set to show diff screenshot in the images during comparison |
        | `Set Show Diff` | False | # Set not to show diff screenshot in the images during comparison |
        
        """
        self.show_diff = show_diff
    
    @keyword
    def set_screenshot_dir(self, screenshot_dir: str):
        """Set the directory to save screenshots."""
        self.screenshot_dir = Path(screenshot_dir)
    
    @keyword
    def set_reference_run(self, reference_run: bool):
        """Set whether the run is a reference run."""
        self.reference_run = reference_run
    
    @keyword
    def set_force_ocr(self, force_ocr: bool):
        """Set whether to force OCR during image comparison."""
        self.force_ocr = force_ocr

    @keyword
    def get_barcodes(self, document: str, assertion_operator: Optional[AssertionOperator] = None, assertion_expected: Any = None, message: str = None):
        """Get the barcodes from a document. Returns the barcodes as a list of dictionaries.
        
        | =Arguments= | =Description= |
        | ``document`` | Path or URL of the document |
        | ``assertion_operator`` | Assertion operator to be used. |
        | ``assertion_expected`` | Expected value for the assertion. |
        | ``message`` | Message to be displayed in the log. |

        Optionally assert the barcodes using the ``assertion_operator`` and ``assertion_expected`` arguments.

        Examples:
        | ${data} | `Get Barcodes` | document.pdf | # Get the barcodes from the document |
        | `Length Should Be` | ${data} | 2 | # Check if the number of barcodes is 2 |
        | `Should Be True` | ${data[0]} == {'x': 5, 'y': 7, 'height': 95, 'width': 96, 'value': 'Stegosaurus'} | # Check if the first barcode is as expected |
        | `Should Be True` | ${data[1]} == {'x': 298, 'y': 7, 'height': 95, 'width': 95, 'value': 'Plesiosaurus'} | # Check if the second barcode is as expected |
        | `Should Be True` | ${data[0]['value']} == 'Stegosaurus' | # Check if the value of the first barcode is 'Stegosaurus' |
        | `Should Be True` | ${data[1]['value']} == 'Plesiosaurus' | # Check if the value of the second barcode is 'Plesiosaurus' |
        | `Get Barcodes` | document.pdf | contains | 'Stegosaurus' | # Check if the document contains a barcode with the value 'Stegosaurus' |
        """
        if is_url(document):
            document = download_file_from_url(document)
        
        # Load the document
        doc = DocumentRepresentation(document, dpi=self.dpi, contains_barcodes=True)

        # Get the barcodes from the document
        barcodes = doc.get_barcodes()
        if assertion_operator:
            barcodes = [barcode['value'] for barcode in barcodes]
        return verify_assertion(barcodes, assertion_operator, assertion_expected, message)

    @keyword
    def get_barcodes_from_document(self, document: str, assertion_operator: Optional[AssertionOperator] = None, assertion_expected: Any = None, message: str = None):
    
        """Get the barcodes from a document. Returns the barcodes as a list of dictionaries.
        
        | =Arguments= | =Description= |
        | ``document`` | Path or URL of the document |
        | ``assertion_operator`` | Assertion operator to be used. |
        | ``assertion_expected`` | Expected value for the assertion. |
        | ``message`` | Message to be displayed in the log. |

        Optionally assert the barcodes using the ``assertion_operator`` and ``assertion_expected`` arguments.

        Examples:
        | ${data} | `Get Barcodes From Document` | document.pdf | # Get the barcodes from the document |
        | `Length Should Be` | ${data} | 2 | # Check if the number of barcodes is 2 |
        | `Should Be True` | ${data[0]} == {'x': 5, 'y': 7, 'height': 95, 'width': 96, 'value': 'Stegosaurus'} | # Check if the first barcode is as expected |
        | `Should Be True` | ${data[1]} == {'x': 298, 'y': 7, 'height': 95, 'width': 95, 'value': 'Plesiosaurus'} | # Check if the second barcode is as expected |
        | `Should Be True` | ${data[0]['value']} == 'Stegosaurus' | # Check if the value of the first barcode is 'Stegosaurus' |
        | `Should Be True` | ${data[1]['value']} == 'Plesiosaurus' | # Check if the value of the second barcode is 'Plesiosaurus' |
        | `Get Barcodes From Document` | document.pdf | contains | 'Stegosaurus' | # Check if the document contains a barcode with the value 'Stegosaurus' |
       
        """
        return self.get_barcodes(document, assertion_operator, assertion_expected, message)

    @keyword
    def image_should_contain_template(self, image: str, template: str, threshold: float=0.0, 
                                      take_screenshots: bool=False, log_template: bool=False, 
                                      detection: str="template",
                                      tpl_crop_x1: int = None, tpl_crop_y1: int = None,
                                      tpl_crop_x2: int = None, tpl_crop_y2: int = None):
        """Verifies that ``image`` contains a ``template``.  

        Returns the coordinates of the template in the image if the template is found.  
        Can be used to find a smaller image ``template`` in a larger image ``image``.  
        ``image`` and ``template`` can be either a path to an image or a url.  
        The ``threshold`` can be used to set the minimum similarity between the two images.  
        If ``take_screenshots`` is set to ``True``, screenshots of the image with the template highlighted are added to the log.  

        | =Arguments= | =Description= |
        | ``image`` | Path of the Image/Document in which the template shall be found |
        | ``template`` | Path of the Image/Document which shall be found in the image |
        | ``threshold`` | Minimum similarity between the two images between ``0.0`` and ``1.0``. Default is ``0.0`` which is an exact match. Higher values allow more differences |
        | ``take_screenshots`` | If set to ``True``, a screenshot of the image with the template highlighted gets linked to the HTML log (if `embed_screenshots` is used during import, the image gets embedded). Default is ``False``. |
        | ``log_template`` | If set to ``True``, a screenshots of the template image gets linked to the HTML log (if `embed_screenshots` is used during import, the image gets embedded). Default is ``False``. |
        | ``detection`` | Detection method to be used. Options are ``template``, ``sift`` and ``orb``.  Default is ``template``. |
        | ``tpl_crop_x1`` | X1 coordinate of the rectangle to crop the template image to.  |
        | ``tpl_crop_y1`` | Y1 coordinate of the rectangle to crop the template image to.  |
        | ``tpl_crop_x2`` | X2 coordinate of the rectangle to crop the template image to.  |
        | ``tpl_crop_y2`` | Y2 coordinate of the rectangle to crop the template image to.  |

        Examples:
        | `Image Should Contain Template` | reference.jpg | template.jpg | #Checks if template is in image |
        | `Image Should Contain Template` | reference.jpg | template.jpg | threshold=0.9 | #Checks if template is in image with a higher threshold |
        | `Image Should Contain Template` | reference.jpg | template.jpg | take_screenshots=True | #Checks if template is in image and adds screenshots to log |
        | `Image Should Contain Template` | reference.jpg | template.jpg | tpl_crop_x1=50  tpl_crop_y1=50  tpl_crop_x2=100  tpl_crop_y2=100 | #Before image comparison, the template image gets cropped to that selection. |
        | `${coordinates}` | `Image Should Contain Template` | reference.jpg | template.jpg | #Checks if template is in image and returns coordinates of template |
        | `Should Be Equal As Numbers` | ${coordinates['pt1'][0]} | 100 | #Checks if x coordinate of found template is 100 |
        """
        all_crop_args = all((tpl_crop_x1, tpl_crop_y1, tpl_crop_x2, tpl_crop_y2))
        any_crop_args = any((tpl_crop_x1, tpl_crop_y1, tpl_crop_x2, tpl_crop_y2))
        if not all_crop_args and any_crop_args:
            raise ValueError("Either provide all crop arguments or none of them.")

        img = DocumentRepresentation(image).pages[0].image
        template = DocumentRepresentation(template).pages[0].image
        # Crop the template image if crop boundaries are provided
        if all_crop_args:
            template = template[tpl_crop_y1:tpl_crop_y2, tpl_crop_x1:tpl_crop_x2]

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        h, w = template.shape[0:2]

        if detection == "template":
            match = False
            res = cv2.matchTemplate(
                img_gray, template_gray, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = min_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            if min_val <= threshold:
                match = True
                cv2.rectangle(img, top_left, bottom_right, 255, 2)
            
            if take_screenshots:
                self.add_screenshot_to_log(img, "image_with_template")

            if log_template:
                self.add_screenshot_to_log(template, "template_original", original_size=True)

            if match:
                return {"pt1": top_left, "pt2": bottom_right}
            else:
                raise AssertionError('The Template was not found in the Image.')
        
        elif detection == "sift" or detection == "orb":
            img_kp, img_des, template_kp, template_des = self.get_sift_keypoints_and_descriptors(img_gray, template_gray)

            if img_kp is None or template_kp is None:
                raise AssertionError('The Template was not found in the Image.')

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(template_des, img_des, k=2)

            # Apply Lowe’s ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_matches.append(m)

            good_matches = [m for m in good_matches if m.distance < 50]
            
            if len(good_matches) >= 4:
                # Add screenshot with good matches to the log
                matches_for_drawing = [m for m in good_matches]
                img_matches = cv2.drawMatches(template, template_kp, img, img_kp, matches_for_drawing, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                self.add_screenshot_to_log(img_matches, "good_sift_matches")
                src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([img_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Use homography to find the template in the larger image
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
                if M is not None:
                    # Define corners of the template and transform them into the full image
                    template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(template_corners, M)

                    # Draw bounding box on the full image
                    cv2.polylines(img, [np.int32(transformed_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

                    # Add a screenshot with the detected area to the log
                    self.add_screenshot_to_log(img, "image_with_template")

                    # Return the coordinates of the detected area
                    top_left = (int(transformed_corners[0][0][0]), int(transformed_corners[0][0][1]))
                    bottom_right = (int(transformed_corners[2][0][0]), int(transformed_corners[2][0][1]))

                    return {"pt1": top_left, "pt2": bottom_right}
                else:
                    raise AssertionError('The Template was not found in the Image.')
            else:
                raise AssertionError('The Template was not found in the Image.')
        
        else:
            raise ValueError("Detection method must be 'template', 'orb' or 'sift'.")
        



    def _get_diff_rectangles(self, absolute_diff):
        """Get rectangles around differences in the page."""
        # Increase the size of the conturs to make sure the differences are covered and small differences are grouped
        # Every contour is a rectangle, overlapping rectangles are grouped together

        absolute_diff = cv2.dilate(absolute_diff, None, iterations=10)
        contours, _ = cv2.findContours(absolute_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = [cv2.boundingRect(contour) for contour in contours]
        return rectangles

    def _raise_comparison_failure(self, message: str = "The compared images are different."):
        """Handle failures in image comparison."""
        raise AssertionError(message)

    def _compare_text_content(self, reference_doc: DocumentRepresentation, candidate_doc: DocumentRepresentation):
        """Compare the text content of the two documents after OCR."""
        reference_text = reference_doc.get_text_content()
        candidate_text = candidate_doc.get_text_content()
        if reference_text != candidate_text:
            raise AssertionError(f"Text content differs:\n\nReference Text:\n{reference_text}\n\nCandidate Text:\n{candidate_text}")

    def _load_placeholders(self, placeholder_file: Union[str, dict, list]):
        """Load and return placeholders from file or dictionary."""
        if isinstance(placeholder_file, str) and os.path.exists(placeholder_file):
            with open(placeholder_file, 'r') as f:
                return json.load(f)
        return placeholder_file  # If it's already in dict/list format

    def get_images_with_highlighted_differences(self, thresh, reference, candidate, extension=10):

        #thresh = cv2.dilate(thresh, None, iterations=extension)
        thresh = cv2.dilate(thresh, None, iterations=extension)
        thresh = cv2.erode(thresh, None, iterations=extension)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(reference, (x, y), (x + w, y + h), (0, 0, 255), 4)
            cv2.rectangle(candidate, (x, y), (x + w, y + h), (0, 0, 255), 4)
        return reference, candidate, cnts

    def get_diff_rectangles(self, absolute_diff):
        """Get rectangles around differences in the page.
        absolute_diff is a np.array with the differences between the two images.
        """
        
        absolute_diff = cv2.dilate(absolute_diff, None, iterations=10)
        absolute_diff = cv2.erode(absolute_diff, None, iterations=10)
        contours, _ = cv2.findContours(absolute_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = [cv2.boundingRect(contour) for contour in contours]
        # Convert the rectangles to a list of dictionaries
        rectangles = [{'x': rect[0], 'y': rect[1], 'width': rect[2], 'height': rect[3]} for rect in rectangles]
        return rectangles

    def add_screenshot_to_log(self, image, suffix, original_size=False):
        if original_size:
            img_style = "width: auto; height: auto;"
        else:
            img_style = "width:50%; height: auto;"

        if self.embed_screenshots:
            import base64
            if self.screenshot_format == 'jpg':
                _, encoded_img  = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])  # im_arr: image in Numpy one-dim array format.
                im_b64 = base64.b64encode(encoded_img).decode()
                print("*HTML* " + f'{suffix}:<br><img alt="screenshot" src="data:image/jpeg;base64,{im_b64}" style="{img_style}">' )
            else:
                _, encoded_img  = cv2.imencode('.png', image)
                im_b64 = base64.b64encode(encoded_img).decode()
                print("*HTML* " + f'{suffix}:<br><img alt="screenshot" src="data:image/png;base64,{im_b64}" style="{img_style}">' )
        else:
            screenshot_name = str(str(uuid.uuid1()) + suffix +
                                '.{}'.format(self.screenshot_format))
            if self.PABOTQUEUEINDEX is not None:
                rel_screenshot_path = str(
                    self.screenshot_dir / '{}-{}'.format(self.PABOTQUEUEINDEX, screenshot_name))
            else:
                rel_screenshot_path = str(
                    self.screenshot_dir / screenshot_name)
            abs_screenshot_path = str(
                self.output_directory/self.screenshot_dir/screenshot_name)
            os.makedirs(os.path.dirname(abs_screenshot_path), exist_ok=True)
            if self.screenshot_format == 'jpg':
                cv2.imwrite(abs_screenshot_path, image, [
                            int(cv2.IMWRITE_JPEG_QUALITY), 70])
            else:
                cv2.imwrite(abs_screenshot_path, image)
            print("*HTML* " + f'{suffix}:<br><a href="{rel_screenshot_path}" target="_blank"><img src="{rel_screenshot_path}" style="{img_style}"></a>')

    def find_partial_image_position(self, img, template, threshold=0.1, detection="classic"):

        if detection == "template":
            result = self.find_partial_image_distance_with_matchtemplate(img, template, threshold)
            
        elif detection == "classic":
            result = self.find_partial_image_distance_with_classic_method(img, template, threshold)
            
        elif detection == "orb":
            result = self.find_partial_image_distance_with_orb(img, template)

        elif detection == "sift":
            result = self.find_partial_image_distance_with_sift(img, template)

        return result 

    def find_partial_image_distance_with_sift(self, img, template, threshold=0.1):
        # Convert both images to grayscale
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find non-white area bounding boxes
        template_nonwhite = cv2.findNonZero(cv2.threshold(template_gray, 254, 255, cv2.THRESH_BINARY_INV)[1])
        img_nonwhite = cv2.findNonZero(cv2.threshold(img_gray, 254, 255, cv2.THRESH_BINARY_INV)[1])
        template_x, template_y, template_w, template_h = cv2.boundingRect(template_nonwhite)
        img_x, img_y, img_w, img_h = cv2.boundingRect(img_nonwhite)

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect SIFT keypoints and descriptors
        keypoints_img, descriptors_img = sift.detectAndCompute(img_gray, None)
        keypoints_template, descriptors_template = sift.detectAndCompute(template_gray, None)

        # Use BFMatcher to match descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(descriptors_template, descriptors_img, k=2)

        # Apply Lowe's ratio test to filter good matches
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # If enough good matches, find homography and return the coordinates
        if len(good_matches) >= 4:
            # Extract the matched keypoints
            src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                # The translation vector (dx, dy) can be extracted directly from the homography matrix M
                dx = M[0, 2]  # Translation in x direction
                dy = M[1, 2]  # Translation in y direction

                # Calculate moved distance using the translation vector
                moved_distance = np.sqrt(dx**2 + dy**2)

                return {
                    "displacement_x": dx,
                    "displacement_y": dy,
                    "distance": moved_distance
                }

        # Return None if not enough good matches
        return None

    def find_partial_image_distance_with_matchtemplate(self, img, template, threshold=0.1):
        print("Find partial image position")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        h, w = template.shape[0:2]
        print("dev detection")
        mask = cv2.absdiff(img_gray, template_gray)
        mask[mask > 0] = 255

        # find contours in the mask and get the largest one
        cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # merge all contours into one
        merged_contour = np.zeros(mask.shape, np.uint8)
        for cnt in cnts:
            cv2.drawContours(merged_contour, [cnt], -1, 255, -1)
        
    

        # # get largest contour
        # largest_contour = max(cnts, key=cv2.contourArea)
        # contour_mask = np.zeros(mask.shape, np.uint8)
        # cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)

        masked_img =      cv2.bitwise_not(cv2.bitwise_and(merged_contour, cv2.bitwise_not(img_gray)))
        masked_template = cv2.bitwise_not(cv2.bitwise_and(merged_contour, cv2.bitwise_not(template_gray)))
        template_blur = cv2.GaussianBlur(masked_template, (3, 3), 0)
        template_thresh = cv2.threshold(
            template_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        temp_x, temp_y, temp_w, temp_h = cv2.boundingRect(template_thresh)
        res = cv2.matchTemplate(
            masked_img, masked_template[temp_y:temp_y + temp_h, temp_x:temp_x + temp_w], cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        res_temp = cv2.matchTemplate(masked_template, masked_template[temp_y:temp_y + temp_h, temp_x:temp_x + temp_w],
                                    cv2.TM_SQDIFF_NORMED)
        min_val_temp, max_val_temp, min_loc_temp, max_loc_temp = cv2.minMaxLoc(
            res_temp)

        if (min_val < threshold):
            return {"pt1": min_loc, "pt2": min_loc_temp}
        return

    def get_orb_keypoints_and_descriptors(self, img1, img2, edgeThreshold=5, patchSize=10):
        orb = cv2.ORB_create(nfeatures=1000, edgeThreshold=edgeThreshold, patchSize=patchSize)
        img1_kp, img1_des = orb.detectAndCompute(img1, None)
        img2_kp, img2_des = orb.detectAndCompute(img2, None)

        if len(img1_kp) == 0 or len(img2_kp) == 0:
            if patchSize > 4:
                patchSize -= 4
                edgeThreshold = int(patchSize/2)
                return self.get_orb_keypoints_and_descriptors(img1, img2, edgeThreshold, patchSize)
            else:
                return None, None, None, None

        return img1_kp, img1_des, img2_kp, img2_des
    
    def get_sift_keypoints_and_descriptors(self, img1, img2):
        sift = cv2.SIFT_create()
        img1_kp, img1_des = sift.detectAndCompute(img1, None)
        img2_kp, img2_des = sift.detectAndCompute(img2, None)

        if len(img1_kp) == 0 or len(img2_kp) == 0:
            return None, None, None, None

        return img1_kp, img1_des, img2_kp, img2_des
    
    def find_partial_image_distance_with_orb(self, img, template):
        print("Find partial image position")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        h, w = template.shape[0:2]
        print("dev detection")
        mask = cv2.absdiff(img_gray, template_gray)
        mask[mask > 0] = 255

        # find contours in the mask and get the largest one
        cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get largest contour
        largest_contour = max(cnts, key=cv2.contourArea)
        contour_mask = np.zeros(mask.shape, np.uint8)

        for cnt in cnts:
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)

        masked_img =      cv2.bitwise_not(cv2.bitwise_and(contour_mask, cv2.bitwise_not(img_gray)))
        masked_template = cv2.bitwise_not(cv2.bitwise_and(contour_mask, cv2.bitwise_not(template_gray)))

        edges_img = cv2.Canny(masked_img, 100, 200)
        edges_template = cv2.Canny(masked_template, 100, 200)

        # Find the keypoints and descriptors for the template image
        template_keypoints, template_descriptors, target_keypoints, target_descriptors = self.get_orb_keypoints_and_descriptors(edges_template, edges_img)

        if len(template_keypoints) == 0 or len(target_keypoints) == 0:
            return

        # Create a brute-force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the template image with the target image
        matches = bf.match(template_descriptors, target_descriptors)

        if len(matches) > 0:

            # Sort the matches based on their distance
            matches = sorted(matches, key=lambda x: x.distance)
            best_matches = matches[:10]
            # Estimate the transformation matrix between the two images
            src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([target_keypoints[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Calculate the amount of movement between the two images
            movement = np.sqrt(np.sum(M[:,2]**2))

            self.add_screenshot_to_log(cv2.drawMatches(masked_template, template_keypoints, masked_img, target_keypoints, best_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS), "ORB_matches")
            return {"distance": movement}
        
    def blend_two_images(self, image, overlay, ignore_color=[255, 255, 255]):
        ignore_color = np.asarray(ignore_color)
        mask = ~(overlay == ignore_color).all(-1)
        # Or mask = (overlay!=ignore_color).any(-1)
        out = image.copy()
        out[mask] = image[mask] * 0.5 + overlay[mask] * 0.5
        return out

    def is_bounding_box_reasonable(self, corners):
        """Check if the bounding box is spatially consistent (rectangular and not too skewed)."""
        tl, tr, br, bl = corners
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)

        # Ensure the width and height are approximately consistent (not too skewed)
        width_diff = abs(width_top - width_bottom)
        height_diff = abs(height_left - height_right)

        return width_diff < 20 and height_diff < 20  # Thresholds can be fine-tuned

def load_watermarks(watermark_file):
    if isinstance(watermark_file, str):
        watermarks = []
        if os.path.isdir(watermark_file):
            watermark_file = [str(os.path.join(watermark_file, f)) for f in os.listdir(
                watermark_file) if os.path.isfile(os.path.join(watermark_file, f))]
        else:
            watermark_file = [watermark_file]
        if isinstance(watermark_file, list):
            try:
                for single_watermark in watermark_file:
                    try:
                        watermark = DocumentRepresentation(single_watermark).pages[0].image
                        if watermark.shape[2] == 4:
                            watermark = watermark[:, :, :3]
                        watermark_gray = cv2.cvtColor(
                            watermark, cv2.COLOR_BGR2GRAY)
                        #watermark_gray = (watermark_gray * 255).astype("uint8")
                        mask = cv2.threshold(watermark_gray, 10, 255, cv2.THRESH_BINARY)[1]   

                        watermarks.append(mask)
                    except:
                        print(
                            f'Watermark file {single_watermark} could not be loaded. Continue with next item.')
                    continue
            except:
                print("Watermark file could not be loaded.")
            return watermarks