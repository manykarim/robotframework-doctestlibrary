from skimage import io, measure, metrics, util, img_as_ubyte
from os.path import splitext, split
from wand.image import Image
from wand.color import Color
from decimal import *
from skimage.draw import rectangle
import json
import time
import re
import cv2
import pytesseract
from pytesseract import Output
from wand.image import Image
import os
from io import BytesIO
import tempfile
from skimage.util import img_as_ubyte
from imutils.object_detection import non_max_suppression
import numpy as np
import sys
from DocTest.PdfDoc import PdfDoc
from concurrent import futures
import fitz
import logging
try:
    from pylibdmtx import pylibdmtx
except ImportError:
    logging.debug('Failed to import pylibdmtx', exc_info=True)
import urllib

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
        self.image = str(image)
        self.path, self.filename= split(image)
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
            with(fitz.open(self.image)) as doc:
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

    def get_ocr_text_data(self):
        self.increase_resolution_for_ocr()
        for i in range(len(self.opencv_images)):
            # print("Parse Image using tesseract")
            cv_image = self.opencv_images[i]
            #rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(self.opencv_images[i], cv2.COLOR_BGR2GRAY)
            threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            #ret, threshold_image = cv2.threshold(cv_image,127,255,cv2.THRESH_BINARY)
            #threshold_image = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
            #d = pytesseract.image_to_data(threshold_image, output_type=Output.DICT, config='--dpi {}'.format(self.MINIMUM_OCR_RESOLUTION))
            d = pytesseract.image_to_data(threshold_image, output_type=Output.DICT, config='--psm 12')
            self.text_content.append(d)

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
        for frame in self.opencv_images:
            text = east_detect(frame)
            self.text_content.append(text)

    def identify_placeholders(self):
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
            try:
                placeholders = json.loads(self.mask)
            except:
                print('The mask {} could not be read as JSON'.format(self.mask))
        if isinstance(placeholders, list) is not True:
            placeholders = [placeholders]
        if (placeholders is not None):
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
                                if 'conf' not in keys or int(d['conf'][j]) > self.PYTESSERACT_CONFIDENCE:
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
                        x, y, h, w = (placeholder['x'], placeholder['y'], placeholder['height'], placeholder['width'])                    
                    elif unit == 'mm':
                        constant = self.DPI / 25.4
                        x, y, h, w = (int(placeholder['x']*constant), int(placeholder['y']*constant), int(placeholder['height']*constant), int(placeholder['width']*constant))
                    elif unit == 'cm':
                        constant = self.DPI / 2.54
                        x, y, h, w = (int(placeholder['x']*constant), int(placeholder['y']*constant), int(placeholder['height']*constant), int(placeholder['width']*constant))
                    placeholder_coordinates = {"page":page, "x":x, "y":y, "height":h, "width":w}
                    self.placeholders.append(placeholder_coordinates)

                elif (placeholder_type == 'area'):
                    page = placeholder.get('page', 'all')
                    location = placeholder.get('location', None)
                    percent = placeholder.get('percent', 10)
                    if page == 'all':
                        image_height = self.opencv_images[0].shape[0]
                        image_width = self.opencv_images[0].shape[1]
                    else:
                        image_height = self.opencv_images[page-1].shape[0]
                        image_width = self.opencv_images[page-1].shape[1]
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




    def identify_barcodes(self):
        for i in range(len(self.opencv_images)):
            print("Identify barcodes")
            image_height = self.opencv_images[i].shape[0]
            try:
                barcodes = pylibdmtx.decode(self.opencv_images[i], timeout=5000)
            except:
                logging.debug("pylibdmtx could not be loaded",exc_info=True)
                return
            self.barcodes.extend(barcodes)
            #Add barcode as placehoder
            for barcode in barcodes:
                print(barcode)
                x = barcode.rect.left
                y = image_height - barcode.rect.top - barcode.rect.height
                h = barcode.rect.height
                w = barcode.rect.width
                barcode_mask = {"page":i+1, "x":x, "y":y, "height":h, "width":w}
                self.placeholders.append(barcode_mask)
        pass
    
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


def east_detect(image):
    padding = 0.1
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    text_content={}
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    #Saving a original image and shape
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # set the new height and width to default 320 by using args #dictionary.  
    (newW, newH) = (960, 960)

    #Calculate the ratio between original and new image for both height and weight. 
    #This ratio will be used to translate bounding box location on the original image. 
    rW = origW / float(newW)
    rH = origH / float(newH)

    # resize the original image to new dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # construct a blob from the image to forward pass it to EAST model
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)

    # load the pre-trained EAST model for text detection
    script_dir = os.path.dirname(__file__)
    rel_east_model_path = "frozen_east_text_detection.pb"
    abs_east_model_path = os.path.join(script_dir, rel_east_model_path)

    # check if file abs_east_model_path exists
    if os.path.isfile(abs_east_model_path) is False:
        # Download from url https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb
        print("Downloading frozen_east_text_detection.pb from url")
        url = "https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb"
        urllib.request.urlretrieve(url, abs_east_model_path)

    net = cv2.dnn.readNet(abs_east_model_path)

    # The following two layer need to pulled from EAST model for achieving this. 
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    
    #Forward pass the blob from the image to get the desired output layers
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # Find predictions and  apply non-maxima suppression
    (boxes, confidence_val) = predictions(scores, geometry)
    boxes = non_max_suppression(np.array(boxes), probs=confidence_val)

    ##Text Detection and Recognition 

    # initialize the list of results
    results = {}
    results["text"] = []
    results["left"] = []
    results["top"] = []
    results["width"] = []
    results["height"] = []


    # loop over the bounding boxes to find the coordinate of bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
        startX = int(startX * rW)-10
        startY = int(startY * rH)-10
        endX = int(endX * rW)+10
        endY = int(endY * rH)+10

        if (startX < 0):
            startX = 0
        if (startY < 0):
            startY = 0
        if (endX > origW):
            endX = origW
        if (endY > origH):
            endY = origH
        

        #extract the region of interest
        r = orig[startY:endY, startX:endX]

        #configuration setting to convert image to string.  
        configuration = ("-l eng --oem 1 --psm 8")
        ##This will recognize the text from the image of bounding box
        text = pytesseract.image_to_string(r, config=configuration)

        # append bbox coordinate and associated text to the list of results 
        results["left"].append(abs(startX))
        results["top"].append(abs(startY))
        results["width"].append(abs(endX-startX))
        results["height"].append(abs(endY-startY))
        results["text"].append(text)

    # #Display the image with bounding box and recognized text
    # orig_image = orig.copy()

    # # Moving over the results and display on the image
    # for i in range(len(results["text"])):
    #     cv2.rectangle(orig_image, (results["left"][i], results["top"][i]), (results["left"][i]+results["width"][i], results["top"][i]+results["height"][i]), (0, 255, 0), 2)
    #     cv2.putText(orig_image, results["text"][i], (results["left"][i], results["top"][i]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # cv2.imwrite("east_text_recognized.jpg", orig_image)
    return results
    
# Returns a bounding box and probability score if it is more than minimum confidence
def predictions(prob_score, geo):
    (numR, numC) = prob_score.shape[2:4]
    boxes = []
    confidence_val = []

    # loop over rows
    for y in range(0, numR):
        scoresData = prob_score[0, 0, y]
        x0 = geo[0, 0, y]
        x1 = geo[0, 1, y]
        x2 = geo[0, 2, y]
        x3 = geo[0, 3, y]
        anglesData = geo[0, 4, y]

        # loop over the number of columns
        for i in range(0, numC):
            if scoresData[i] < EAST_CONFIDENCE:
                continue

            (offX, offY) = (i * 4.0, y * 4.0)

            # extracting the rotation angle for the prediction and computing the sine and cosine
            angle = anglesData[i]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # using the geo volume to get the dimensions of the bounding box
            h = x0[i] + x2[i]
            w = x1[i] + x3[i]

            # compute start and end for the text pred bbox
            endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
            endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
            startX = int(endX - w)
            startY = int(endY - h)

            boxes.append((startX, startY, endX, endY))
            confidence_val.append(scoresData[i])

    # return bounding boxes and associated confidence_val
    return (boxes, confidence_val)

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
