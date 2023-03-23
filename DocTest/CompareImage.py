from skimage import io, measure, metrics, util, img_as_ubyte
from os.path import splitext, split
from decimal import *
from skimage.draw import rectangle
import json
import time
import re
import cv2
import pytesseract
from pytesseract import Output
import os
from io import BytesIO
import tempfile
from skimage.util import img_as_ubyte
from imutils.object_detection import non_max_suppression
import numpy as np
import sys
from concurrent import futures
import fitz
import logging
from DocTest.Ocr import EastTextExtractor
from DocTest.Downloader import is_url, download_file_from_url

EAST_CONFIDENCE=0.5

class CompareImage(object):

    ROBOT_LIBRARY_VERSION = 1.0
    DPI=200
    PYTESSERACT_CONFIDENCE=20
    EAST_CONFIDENCE=0
    MINIMUM_OCR_RESOLUTION = 300
    
    def __init__(self, image, **kwargs):
        tic = time.perf_counter()

        self.placeholder_file = kwargs.pop('placeholder_file', None)
        self.mask = kwargs.pop('mask', None)
        self.contains_barcodes = kwargs.pop('contains_barcodes', False)
        self.get_pdf_content = kwargs.pop('get_pdf_content', False)
        self.force_ocr = kwargs.pop('force_ocr', False)
        self.ocr_engine = kwargs.pop('ocr_engine', 'tesseract')
        self.DPI = int(kwargs.pop('DPI', 200))
        if is_url(image):
            self.image = download_file_from_url(image)
        else:
            self.image = str(image)
        self.path, self.filename= split(self.image)
        self.filename_without_extension, self.extension = splitext(self.filename)
        self.opencv_images = []
        self.placeholders = []
        self.placeholder_mask = None
        self.text_content = []
        #self.pdf_content = []
        self.placeholder_frame_width = 10
        self.tmp_directory = tempfile.TemporaryDirectory()
        self.diff_images = []
        self.threshold_images = []
        self.barcodes = []
        self.rerendered_for_ocr = False
        self.mupdfdoc= None
        self.load_image_into_array()
        self.load_text_content_and_identify_masks()
        
    
        toc = time.perf_counter()
        print(f"Compare Image Object created in {toc - tic:0.4f} seconds")

    def convert_mupdf_to_opencv_image(self, resolution=None):
        self.opencv_images = []
        if resolution == None:
            resolution = self.DPI
        tic = time.perf_counter()
        try:
            self.mupdfdoc = fitz.open(self.image)
            toc = time.perf_counter()
            print(f"Rendering document to PyMuPDF Image performed in {toc - tic:0.4f} seconds")
            #split pages
            tic = time.perf_counter()
            for i, page in enumerate(self.mupdfdoc.pages()):
                zoom = resolution/72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix = mat)
                imgData = pix.tobytes("png")
                nparr = np.frombuffer(imgData, np.uint8)
                opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self.opencv_images.append(opencv_image)
                pass
            toc = time.perf_counter()
            print(f"Conversion from PyMuPDF Image to OpenCV Image performed in {toc - tic:0.4f} seconds")
        except:
            raise AssertionError("File could not be converted by ImageMagick to OpenCV Image: {}".format(self.image))

        
    def get_text_content(self):
        for i in range(len(self.opencv_images)):
            cv_image = self.opencv_images[i]
            text = pytesseract.image_to_string(cv_image)
            self.text_content.append(text)
        return self.text_content

    def get_ocr_text_data(self, ocr_config: str='--psm 11', ocr_lang: str='eng'):
        self.increase_resolution_for_ocr()
        for i in range(len(self.opencv_images)):
            text_list = []
            left_list = []
            top_list = []
            width_list = []
            height_list = []
            conf_list = []

            cv_image = self.opencv_images[i]
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            ocr_config = ocr_config + f' -l {ocr_lang}'
            d = pytesseract.image_to_data(threshold_image, output_type=Output.DICT, config=ocr_config)
            n_boxes = len(d['text'])         

            # For each detected part
            for j in range(n_boxes):

                # If the prediction accuracy greater than %50
                if int(float(d['conf'][j])) > self.PYTESSERACT_CONFIDENCE:
                    text_list.append(d['text'][j])
                    left_list.append(d['left'][j])
                    top_list.append(d['top'][j])
                    width_list.append(d['width'][j])
                    height_list.append(d['height'][j])
                    conf_list.append(d['conf'][j])
            self.text_content.append({'text': text_list, 'left': left_list, 'top': top_list, 'width': width_list, 'height': height_list, 'conf': conf_list})

    def increase_resolution_for_ocr(self):
        # experimental: IF OCR is used and DPI is lower than self.MINIMUM_OCR_RESOLUTION DPI, re-render with self.MINIMUM_OCR_RESOLUTION DPI
        if (self.DPI < self.MINIMUM_OCR_RESOLUTION):
            self.rerendered_for_ocr = True
            print("Re-Render document for OCR at {} DPI as current resolution is only {} DPI".format(self.MINIMUM_OCR_RESOLUTION, self.DPI))
            if self.extension == '.pdf':
                self.convert_mupdf_to_opencv_image(resolution=self.MINIMUM_OCR_RESOLUTION)
            elif (self.extension == '.ps') or (self.extension == '.pcl'):
                self.convert_pywand_to_opencv_image(resolution=self.MINIMUM_OCR_RESOLUTION)
            else:
                scale = self.MINIMUM_OCR_RESOLUTION / self.DPI # percent of original size
                width = int(self.opencv_images[0].shape[1] * scale)
                height = int(self.opencv_images[0].shape[0] * scale)
                dim = (width, height)
                # resize image
                self.opencv_images[0] = cv2.resize(self.opencv_images[0], dim, interpolation = cv2.INTER_CUBIC)

    def get_text_content_with_east(self):
        self.increase_resolution_for_ocr()
        self.east_text_extractor = EastTextExtractor()
        for frame in self.opencv_images:
            text = self.east_text_extractor.get_image_text(frame)
            self.text_content.append(text)

    def identify_placeholders(self):
        placeholders = None
        if self.placeholder_file is not None:
            try:
                with open(self.placeholder_file, 'r') as f:
                    placeholders = json.load(f)
            except IOError as err:
                print("Placeholder File %s is not accessible", self.placeholder_file)
                print("I/O error: {0}".format(err))
            except:
                print("Unexpected error:", sys.exc_info()[0])
                raise
        elif self.mask is not None:
            if isinstance(self.mask, dict):
                placeholders = self.mask
            elif isinstance(self.mask, list):
                placeholders = self.mask
            elif isinstance(self.mask, str):
                try:
                    placeholders = json.loads(self.mask)
                except:
                    print('The mask {} could not be read as JSON'.format(self.mask))
                    # Split the mask at ;
                    mask_list = self.mask.split(';')
                    for mask in mask_list:
                        if len(mask) > 0:
                            # Split the mask at : but only once
                            location, percent = mask.split(':',1)
                            # Check if location is top, bottom, left, right
                            if location in ['top', 'bottom', 'left', 'right']:
                                if percent.isnumeric():
                                    if placeholders is None:
                                        placeholders = []
                                    placeholders.append({'page': 'all', 'type': 'area', 'location': location, 'percent': percent})
                                else:
                                    print('The mask {} is not valid. The percent value is not a number'.format(mask))
        if (placeholders is not None):
            if isinstance(placeholders, list) is not True:
                placeholders = [placeholders]
            for placeholder in placeholders:
                placeholder_type = str(placeholder.get('type'))
                if (placeholder_type == 'pattern' or placeholder_type == 'line_pattern' or placeholder_type == 'word_pattern'):
                    # print("Pattern placeholder identified:")
                    # print(placeholder)
                    pattern = str(placeholder.get('pattern'))
                    xoffset = int(placeholder.get('xoffset', 0))
                    yoffset = int(placeholder.get('yoffset', 0))
                    # print(pattern)

                    if self.mupdfdoc is None or self.force_ocr is True:
                        if self.ocr_engine == 'tesseract':
                            self.get_ocr_text_data()
                        elif self.ocr_engine == 'east':
                            self.get_text_content_with_east()
                        else:
                            self.get_ocr_text_data()
                        for i in range(len(self.opencv_images)):
                            d = self.text_content[i]
                            keys = list(d.keys())
                            n_boxes = len(d['text'])
                            for j in range(n_boxes):
                                if 'conf' not in keys or int(float(d['conf'][j])) > self.PYTESSERACT_CONFIDENCE:
                                    if re.match(pattern, d['text'][j]):
                                        (x, y, w, h) = (d['left'][j], d['top'][j], d['width'][j], d['height'][j])
                                        if self.rerendered_for_ocr:
                                            pixel_recalculation_factor = self.DPI / self.MINIMUM_OCR_RESOLUTION
                                            (x, y, w, h) = (int(pixel_recalculation_factor * x), int(pixel_recalculation_factor * y), int(pixel_recalculation_factor * w), int(pixel_recalculation_factor * h))
                                        text_pattern_mask = {"page":i+1, "x":x-xoffset, "y":y-yoffset, "height":h+2*yoffset, "width":w+2*xoffset}
                                        self.placeholders.append(text_pattern_mask)
                        if self.rerendered_for_ocr:
                            self.load_image_into_array()
                    else:
                        for i in range(len(self.opencv_images)):
                            if (placeholder_type == 'word_pattern'):
                                print("Searching word_pattern")
                                words = self.mupdfdoc[i].get_text("words")
                                search_pattern = re.compile(pattern)
                                for word in words:
                                    if search_pattern.match(word[4]):
                                        (x, y, w, h) = (word[0]*self.DPI/72, word[1]*self.DPI/72,(word[2]-word[0])*self.DPI/72, (word[3]-word[1])*self.DPI/72)
                                        text_pattern_mask = {"page":i+1, "x":x-xoffset, "y":y-yoffset, "height":h+2*yoffset, "width":w+2*xoffset}
                                        self.placeholders.append(text_pattern_mask)
                            if (placeholder_type == 'pattern' or placeholder_type == 'line_pattern'):
                                print("Searching line_pattern")
                                tdict = json.loads(self.mupdfdoc[i].get_text("json"))
                                search_pattern = re.compile(pattern)
                                for block in tdict['blocks']:
                                    if block['type'] == 0:
                                        for line in block['lines']:
                                            if len(line['spans']) != 0 and search_pattern.match(line['spans'][0]['text']):
                                                (x, y, w, h) = (line['bbox'][0]*self.DPI/72, line['bbox'][1]*self.DPI/72,(line['bbox'][2]-line['bbox'][0])*self.DPI/72, (line['bbox'][3]-line['bbox'][1])*self.DPI/72)
                                                text_pattern_mask = {"page":i+1, "x":x-xoffset, "y":y-yoffset, "height":h+2*yoffset, "width":w+2*xoffset}
                                                self.placeholders.append(text_pattern_mask)       
                        
                elif (placeholder_type == 'coordinates'):
                    # print("Coordinate placeholder identified:")
                    # print(placeholder)
                    page = placeholder.get('page', 'all')
                    unit = placeholder.get('unit', 'px')
                    if unit == 'px':
                        x, y, h, w = (int(placeholder['x']), int(placeholder['y']), int(placeholder['height']), int(placeholder['width']))                    
                    elif unit == 'mm':
                        constant = self.DPI / 25.4
                        x, y, h, w = (int(float(placeholder['x'])*constant), int(float(placeholder['y'])*constant), int(float(placeholder['height'])*constant), int(float(placeholder['width'])*constant))
                    elif unit == 'cm':
                        constant = self.DPI / 2.54
                        x, y, h, w = x, y, h, w = (int(float(placeholder['x'])*constant), int(float(placeholder['y'])*constant), int(float(placeholder['height'])*constant), int(float(placeholder['width'])*constant))
                    placeholder_coordinates = {"page":page, "x":x, "y":y, "height":h, "width":w}
                    self.placeholders.append(placeholder_coordinates)

                elif (placeholder_type == 'area'):
                    page = placeholder.get('page', 'all')
                    location = placeholder.get('location', None)
                    percent = int(placeholder.get('percent', 10))
                    if page == 'all':
                        image_height = self.opencv_images[0].shape[0]
                        image_width = self.opencv_images[0].shape[1]
                    elif page.isnumeric():
                        page = int(page)
                        image_height = self.opencv_images[page-1].shape[0]
                        image_width = self.opencv_images[page-1].shape[1]
                    else:
                        print("Invalid page number, will apply to all pages")
                        page = 'all'
                        image_height = self.opencv_images[0].shape[0]
                        image_width = self.opencv_images[0].shape[1]
                    if location == 'top':
                        height = int(image_height * percent / 100)
                        width = image_width
                        placeholder_coordinates = {"page":page, "x":0, "y":0, "height":height, "width":width}
                        pass
                    elif location == 'bottom':
                        height = int(image_height * percent / 100)
                        width = image_width
                        placeholder_coordinates = {"page":page, "x":0, "y":image_height - height, "height":height, "width":width}
                    elif location == 'left':
                        height = image_height
                        width = int(image_width * percent / 100)
                        placeholder_coordinates = {"page":page, "x":0, "y":0, "height":height, "width":width}
                    elif location == 'right':
                        height = image_height
                        width = int(image_width * percent / 100)
                        placeholder_coordinates = {"page":page, "x":image_width - width, "y":0, "height":height, "width":width}
                    self.placeholders.append(placeholder_coordinates)


    def identify_barcodes_with_opencv(self):
        try:
            qrcode_detector = cv2.QRCodeDetector()
            barcode_detector = cv2.barcode.BarcodeDetector()
        except:
            print("OpenCV contrib package is not installed, barcode detection is not available. Make sure to install opencv-contrib-python")
            return

        for i in range(len(self.opencv_images)):
            print("Identify barcodes")
            image_height = self.opencv_images[i].shape[0]
            image_width = self.opencv_images[i].shape[1]
            # Detect QR code
            retval, points, straight_qrcode = qrcode_detector.detectAndDecode(self.opencv_images[i])
            if retval:
                print("QR code detected")
                print(retval)
                print(points)
                x = points[0][0]
                y = points[0][1]
                h = points[2][1] - points[0][1]
                w = points[2][0] - points[0][0]
                barcode_placeholder = {"page":i+1, "x":x, "y":y, "height":h, "width":w}
                self.placeholders.append(barcode_placeholder)
            # Detect barcode
            retval, decoded_info, decoded_type, points = barcode_detector.detectAndDecode(self.opencv_images[i])
            if retval:
                print("Barcode detected")
                print(retval)
                print(points)
                
                for point in points:
                    x = point[1][0]
                    y = point[1][1]
                    h = point[3][1] - point[1][1]
                    w = point[3][0] - point[1][0]
                    barcode_placeholder = {"page":i+1, "x":x, "y":y, "height":h, "width":w}
                    self.placeholders.append(barcode_placeholder)


    def identify_barcodes_with_zbar(self):
        try:
            from pyzbar import pyzbar
        except:
            logging.debug('Failed to import pyzbar', exc_info=True)
            return
        for i in range(len(self.opencv_images)):
            print("Identify barcodes")
            image_height = self.opencv_images[i].shape[0]
            image_width = self.opencv_images[i].shape[1]
            barcodes = pyzbar.decode(self.opencv_images[i])
            #Add barcode as placehoder
            for barcode in barcodes:
                print(barcode)
                x = barcode.rect.left
                y = barcode.rect.top
                h = barcode.rect.height
                w = barcode.rect.width
                value = barcode.data.decode("utf-8")
                barcode_placeholder = {"page":i+1, "x":x, "y":y, "height":h, "width":w}
                self.placeholders.append(barcode_placeholder)
                self.barcodes.append({"page":i+1, "x":x, "y":y, "height":h, "width":w, "value":value})
    
    def identify_datamatrices(self):
        try:
            from pylibdmtx import pylibdmtx
        except:
            logging.debug('Failed to import pylibdmtx', exc_info=True)
            return
        for i in range(len(self.opencv_images)):
            print("Identify datamatrices")
            image_height = self.opencv_images[i].shape[0]
            try:
                barcodes = pylibdmtx.decode(self.opencv_images[i], timeout=5000)
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
                barcode_mask = {"page":i+1, "x":x, "y":y, "height":h, "width":w}
                self.placeholders.append(barcode_mask)
                self.barcodes.append({"page":i+1, "x":x, "y":y, "height":h, "width":w, "value":value})


    def identify_barcodes(self):
        #self.identify_barcodes_with_opencv()
        self.identify_barcodes_with_zbar()
        self.identify_datamatrices()
    
    @staticmethod
    def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    def get_image_with_placeholders(self, placeholders=None):
        if placeholders is None:
            placeholders = self.placeholders
        images_with_placeholders = self.opencv_images
        for placeholder in placeholders:
            if placeholder['page'] == 'all':
                for i in range(len(images_with_placeholders)):
                    start_point = (placeholder['x']-5, placeholder['y']-5)
                    end_point = (start_point[0]+placeholder['width']+10, start_point[1]+placeholder['height']+10)
                    try:
                        images_with_placeholders[i]=cv2.rectangle(images_with_placeholders[i], start_point, end_point, (255, 0, 0), -1)
                    except IndexError as err:
                        print("Page ", i, " does not exist in document")
                        print("Placeholder ", placeholder, " could not be applied")
            else:
                pagenumber = placeholder['page']-1
                start_point = (int(placeholder['x']-5), int(placeholder['y']-5))
                end_point = (int(start_point[0]+placeholder['width']+10), int(start_point[1]+placeholder['height']+10))
                try:
                    images_with_placeholders[pagenumber]=cv2.rectangle(images_with_placeholders[pagenumber], start_point, end_point, (255, 0, 0), -1)
                except IndexError as err:
                    print("Page ", pagenumber, " does not exist in document")
                    print("Placeholder ", placeholder, " could not be applied")
        return images_with_placeholders

    def load_image_into_array(self):
        if (os.path.isfile(self.image) is False):
            raise AssertionError('The file does not exist: {}'.format(self.image))
        if self.extension=='.pdf':
            self.convert_mupdf_to_opencv_image()
        elif (self.extension=='.ps') or (self.extension=='.pcl'):
            self.convert_pywand_to_opencv_image()
        else:
            self.DPI = 72
            img = cv2.imread(self.image)
            if img is None:
                raise AssertionError("No OpenCV Image could be created for file {} . Maybe the file is corrupt?".format(self.image))
            if self.opencv_images:
                self.opencv_images[0]= img
            else:
                self.opencv_images.append(img)

    def load_text_content_and_identify_masks(self):
        if (self.placeholder_file is not None) or (self.mask is not None):
            self.identify_placeholders()
        if (self.contains_barcodes==True):
            self.identify_barcodes()
        if self.placeholders != []:
            print('Identified Masks: {}'.format(self.placeholders))

    def get_text_content_from_mupdf(self):
        pass

    def convert_pywand_to_opencv_image(self, resolution=None):
        self.opencv_images = []
        if resolution == None:
            resolution = self.DPI
        tic = time.perf_counter()
        try:
            from wand.image import Image
            from wand.color import Color
            with(Image(filename=self.image, resolution=resolution)) as source:

                toc = time.perf_counter()
                print(f"Rendering document to pyWand Image performed in {toc - tic:0.4f} seconds")

                images = source.sequence
                pages = len(images)

                tic = time.perf_counter()

                for i in range(pages):
                    images[i].background_color = Color('white')  # Set white background.
                    images[i].alpha_channel = 'remove'  # Remove transparency and replace with bg.
                    opencv_image = np.array(images[i])
                    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
                    self.opencv_images.append(opencv_image)

                toc = time.perf_counter()
                print(f"Conversion from pyWand Image to OpenCV Image performed in {toc - tic:0.4f} seconds")
        except:
            raise AssertionError("File could not be converted by ImageMagick to OpenCV Image: {}".format(self.image))

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
