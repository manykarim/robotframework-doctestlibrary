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
                 ocr_engine: Literal["tesseract", "east"] = OCR_ENGINE_DEFAULT, screenshot_format: str = 'jpg', embed_screenshots: bool = False, force_ocr: bool = False, **kwargs):
        self.threshold = threshold
        self.dpi = dpi
        self.take_screenshots = take_screenshots
        self.show_diff = show_diff
        self.ocr_engine = ocr_engine
        self.screenshot_format = screenshot_format if screenshot_format in ['jpg', 'png'] else 'jpg'
        self.embed_screenshots = embed_screenshots
        self.screenshot_dir = Path('screenshots')
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
                       watermark_file: str = None, force_ocr: bool = False, DPI: int = None, resize_candidate: bool = False, 
                       blur: bool = False, threshold: float = None, mask: Union[str, dict, list] = None, **kwargs):
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

        watermarks = []

        # Apply ignore areas if provided
        abstract_ignore_areas = None
       
        detected_differences = []
        # Compare visual content through the Page class
        for ref_page, cand_page in zip(reference_doc.pages, candidate_doc.pages):

            # Resize the candidate page if needed
            if resize_candidate:
                cand_page.image = cv2.resize(cand_page.image, (ref_page.image.shape[1], ref_page.image.shape[0]))
            
            # Check if dimensions are different
            if ref_page.image.shape != cand_page.image.shape:
                detected_differences.append((ref_page, cand_page, "Image dimensions are different."))
                continue

            similar, diff, thresh, absolute_diff, score = ref_page.compare_with(cand_page, threshold=threshold)

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
                        result = self.find_partial_image_position(reference_area, candidate_area, threshold=0.1, detection="template")
                        if result:
                            if 'distance' in result:
                                distance = result["distance"]
                            # Check if result is a dictuinory with pt1 and pt2
                            if "pt1" in result and "pt2" in result:
                                pt1 = result["pt1"]
                                pt2 = result["pt2"]
                                distance = np.sqrt(np.sum((np.array(pt1) - np.array(pt2))**2))
                                
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
        """Get the text content of a specific area in a document."""
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
        """Get the text content of a document."""
        return self.get_text(document, assertion_operator, assertion_expected, message)

    @keyword
    def get_text(self, document: str, assertion_operator: Optional[AssertionOperator] = None,
        assertion_expected: Any = None,
        message: str = None):
        """Get the text content of a document."""
        if is_url(document):
            document = download_file_from_url(document)
        
        # Load the document
        doc = DocumentRepresentation(document, dpi=self.dpi, ocr_engine=self.ocr_engine)

        # Get the text content of the document
        text = doc.get_text(force_ocr=self.force_ocr)
        return verify_assertion(text, assertion_operator, assertion_expected, message)

    @keyword    
    def set_ocr_engine(self, ocr_engine: Literal["tesseract", "east"]):
        """Set the OCR engine to be used for text extraction."""
        self.ocr_engine = ocr_engine

    @keyword
    def set_dpi(self, dpi: int):
        """Set the DPI to be used for image processing."""
        self.dpi = dpi
    
    @keyword
    def set_threshold(self, threshold: float):
        """Set the threshold for image comparison."""
        self.threshold = threshold
    
    @keyword
    def set_screenshot_format(self, screenshot_format: str):
        """Set the format of the screenshots to be saved."""
        self.screenshot_format = screenshot_format
    
    @keyword
    def set_embed_screenshots(self, embed_screenshots: bool):
        """Set whether to embed screenshots in the log."""
        self.embed_screenshots = embed_screenshots
    
    @keyword
    def set_take_screenshots(self, take_screenshots: bool):
        """Set whether to take screenshots during image comparison."""
        self.take_screenshots = take_screenshots
    
    @keyword
    def set_show_diff(self, show_diff: bool):
        """Set whether to show differences in the images during comparison."""
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

        return result 

    def find_partial_image_distance_with_classic_method(self, img, template, threshold=0.1):
        print("Find partial image position")
        rectangles = []
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        h, w = template.shape[0:2]
        print("Old detection")
        template_blur = cv2.GaussianBlur(template_gray, (3, 3), 0)
        template_thresh = cv2.threshold(
            template_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Obtain bounding rectangle and extract ROI
        temp_x, temp_y, temp_w, temp_h = cv2.boundingRect(template_thresh)

        res = cv2.matchTemplate(
            img_gray, template_gray[temp_y:temp_y + temp_h, temp_x:temp_x + temp_w], cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        res_temp = cv2.matchTemplate(template_gray, template_gray[temp_y:temp_y + temp_h, temp_x:temp_x + temp_w],
                                    cv2.TM_SQDIFF_NORMED)
        min_val_temp, max_val_temp, min_loc_temp, max_loc_temp = cv2.minMaxLoc(
            res_temp)

        if (min_val < threshold):
            return {"pt1": min_loc, "pt2": min_loc_temp}
        return

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

    def get_orb_keypoints_and_descriptors(self, img1, img2, edgeThreshold = 5, patchSize = 10):
        orb = cv2.ORB_create(nfeatures=250, edgeThreshold=edgeThreshold, patchSize=patchSize)
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