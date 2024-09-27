import base64
from typing import Union
import uuid
from DocTest.DocumentRepresentation import DocumentRepresentation
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

@library
class VisualTest:
    ROBOT_LIBRARY_VERSION = 1.0
    DPI_DEFAULT = 200
    OCR_ENGINE_DEFAULT = 'tesseract'

    def __init__(self, threshold: float = 0.0, dpi: int = DPI_DEFAULT, take_screenshots: bool = False, show_diff: bool = False, 
                 ocr_engine: str = OCR_ENGINE_DEFAULT, screenshot_format: str = 'jpg', embed_screenshots: bool = False):
        self.threshold = threshold
        self.dpi = dpi
        self.take_screenshots = take_screenshots
        self.show_diff = show_diff
        self.ocr_engine = ocr_engine
        self.screenshot_format = screenshot_format if screenshot_format in ['jpg', 'png'] else 'jpg'
        self.embed_screenshots = embed_screenshots
        self.screenshot_dir = Path('screenshots')
        built_in = BuiltIn()
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
                       watermark_file: str = None, force_ocr: bool = False, DPI: int = None, resize_candidate: bool = False, 
                       blur: bool = False, threshold: float = None, mask: Union[str, dict, list] = None):
        """
        Compare two images or documents visually and textually (if needed).
        If they are different, a side-by-side comparison image with highlighted differences will be saved and added to the log.
        """
        # Download files if URLs are provided
        if is_url(reference_image):
            reference_image = download_file_from_url(reference_image)
        if is_url(candidate_image):
            candidate_image = download_file_from_url(candidate_image)

        # Set DPI and threshold if provided
        dpi = DPI if DPI else self.dpi
        threshold = threshold if threshold is not None else self.threshold

        # Load reference and candidate documents
        reference_doc = DocumentRepresentation(reference_image, dpi=dpi, ocr_engine=self.ocr_engine, ignore_area_file=placeholder_file, ignore_area=mask)
        candidate_doc = DocumentRepresentation(candidate_image, dpi=dpi, ocr_engine=self.ocr_engine)

        # Apply ignore areas if provided
        abstract_ignore_areas = None
        requires_ocr = force_ocr  # Start by using the force_ocr flag

        # Compare visual content through the Page class
        for ref_page, cand_page in zip(reference_doc.pages, candidate_doc.pages):
            similar, diff, thresh, absolute_diff = ref_page.compare_with(cand_page, threshold=threshold)

            if self.take_screenshots:
                # Save original images to the screenshot directory and add them to the Robot Framework log
                # But add them next to each other in the log
                # Do a np.concatenate with axis=1 to add them next to each other
                combined_image = np.concatenate((ref_page.image, cand_page.image), axis=1)
                self.add_screenshot_to_log(combined_image, suffix='_combined', original_size=False)


            if check_text_content and not similar:
                similar = True
                # If the images are not similar, we need to compare text content
                # Only compare the text content in the areas that have differences
                # For that, the rectangles around the differences are needed
                diff_rectangles = self.get_diff_rectangles(absolute_diff)
                # Compare text content only in the areas that have differences
                for rect in diff_rectangles:
                    
                    same_text, ref_area_text, cand_area_text = ref_page._compare_text_content_in_area_with(cand_page, rect)
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


                # Raise an assertion error if the images are not similar
                self._raise_comparison_failure(reference_doc, candidate_doc)

        print("Images/Document comparison passed.")

    def _get_diff_rectangles(self, absolute_diff):
        """Get rectangles around differences in the page."""
        # Increase the size of the conturs to make sure the differences are covered and small differences are grouped
        # Every contour is a rectangle, overlapping rectangles are grouped together

        absolute_diff = cv2.dilate(absolute_diff, None, iterations=10)
        contours, _ = cv2.findContours(absolute_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = [cv2.boundingRect(contour) for contour in contours]
        return rectangles

    def _raise_comparison_failure(self, reference_doc: DocumentRepresentation, candidate_doc: DocumentRepresentation):
        """Handle failures in image comparison."""
        print("Visual comparison failed.")
        raise AssertionError("The compared images are different.")

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