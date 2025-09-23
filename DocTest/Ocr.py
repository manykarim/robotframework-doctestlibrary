import time
import os
import cv2
import numpy as np
import pytesseract
from typing import Dict, List
from pytesseract import Output
import urllib
import re
import unicodedata

PYTESSERACT_CONFIDENCE=20

def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

class EastTextExtractor:
    layer_names = ('feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3',)

    def __init__(self, east=None):
        pkg_path = os.path.dirname(__file__)
        pkg_east_model = os.path.join(pkg_path, 'data/frozen_east_text_detection.pb')
        self.east = east or pkg_east_model
        self._load_assets()

    def get_image_text(self,
                       image,
                       width=480,
                       height=480,
                       numbers=True,
                       confThreshold=0.2,
                       nmsThreshold=0.1,
                       percentage=10.0,
                       min_boxes=1,
                       max_iterations=20,
                       **kwargs):
        loaded_image = image
        orig_h, orig_w = image.shape[:2]

        image, width, height, ratio_width, ratio_height = self._resize_image(
            loaded_image, width, height
        )
        scores, geometry = self._compute_scores_geometry(image, width, height)

        # decoding results from the model
        rectangles, confidences = box_extractor(scores, geometry, confThreshold)

        # find countur of all rectangles

        mask = np.zeros((height, width), dtype=np.uint8)
        for i in range(len(rectangles)):
            x1, y1, x2, y2 = rectangles[i]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        results = {'text':[], 'left':[], 'top':[], 'width':[], 'height':[], 'conf':[]}
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            start_x = int(x * ratio_width)
            start_y = int(y * ratio_height)
            end_x = int((x + w) * ratio_width)
            end_y = int((y + h) * ratio_height)
            # ROI to be recognized
            pad_x = max(5, int((end_x - start_x) * 0.6))
            pad_y = max(2, int((end_y - start_y) * 0.2))
            roi_start_x = max(0, start_x - pad_x)
            roi_start_y = max(0, start_y - pad_y)
            roi_end_x = min(orig_w, end_x + pad_x)
            roi_end_y = min(orig_h, end_y + pad_y)
            roi = loaded_image[roi_start_y:roi_end_y, roi_start_x:roi_end_x]
            if roi.size == 0:
                continue

            text = self._recognize_roi(roi)
            
            # recalculate coordinates in tesseract text dict and add start_x and start_y
            for i in range(len(text['text'])):
                if text['conf'][i] == '-1':
                    continue
                text['text'][i] = remove_control_characters(text['text'][i])
                text['left'][i] = text['left'][i] + roi_start_x
                text['top'][i] = text['top'][i] + roi_start_y
                text['width'][i] = text['width'][i]
                text['height'][i] = text['height'][i]
                text['text'][i] = text['text'][i].strip()
                text['conf'][i] = 100



                # collating results
                results['text'].append(text['text'][i])
                results['left'].append(text['left'][i])
                results['top'].append(text['top'][i])
                results['width'].append(text['width'][i])
                results['height'].append(text['height'][i])
                results['conf'].append(text['conf'][i])

        return results

    def _recognize_roi(self, roi: np.ndarray) -> Dict[str, List[str]]:
        config = '--oem 1 --psm 7 -l eng'
        scale_factor = 3 if min(roi.shape[0], roi.shape[1]) < 80 else 1
        roi_for_ocr = roi
        if scale_factor > 1:
            roi_for_ocr = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        data = pytesseract.image_to_data(roi_for_ocr, config=config, output_type=Output.DICT)

        # rescale coordinates back to ROI space
        if scale_factor > 1:
            for key in ('left', 'top', 'width', 'height'):
                data[key] = [int(value / scale_factor) for value in data.get(key, [])]

        return data



    def _load_image(self, image):
        return cv2.imread(image)

    def _resize_image(self, image, width, height):
        (H, W) = image.shape[:2]

        (newW, newH) = (width, height)
        ratio_width = W / float(newW)
        ratio_height = H / float(newH)


        # resize the image and grab the new image dimensions
        resized_image = cv2.resize(image, (newW, newH))
        (H, W) = resized_image.shape[:2]
        return (resized_image, height, width, ratio_width, ratio_height)

    def _compute_scores_geometry(self, image, width, height):
        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (width, height), (123.68, 116.78, 103.94), swapRB=True, crop=False
        )
        start = time.time()
        self.east_net.setInput(blob)
        (scores, geometry) = self.east_net.forward(self.layer_names)
        end = time.time()

        # show timing information on text prediction
        print('[INFO] text detection took {:.6f} seconds'.format(end - start))
        return (scores, geometry)

    def _load_assets(self):
        self._get_east()
        start = time.time()
        self.east_net = cv2.dnn.readNet(self.east)
        end = time.time()
        print('[INFO] Loaded EAST text detector {:.6f} seconds ...'.format(end - start))

    def _get_east(self):
        if os.path.exists(self.east):
            return

            # load the pre-trained EAST model for text detection
        pkg_path = os.path.dirname(__file__)
        data_file = os.path.join(pkg_path, self.east)
        os.makedirs(os.path.dirname(data_file), exist_ok=True)

        # check if file abs_east_model_path exists
        if os.path.isfile(data_file) is False:
            # Download from url https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb
            print("Downloading frozen_east_text_detection.pb from url")
            url = "https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb"
            urllib.request.urlretrieve(url, data_file)


    

    def _extract_text(self, image, boxes, percent, numbers, ratio_width, ratio_height):
        extracted_text = []
        for (start_X, start_Y, end_X, end_Y) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            percent = (percent / 100 + 1) if percent >= 0 else ((100 - percent) / 100)
            start_X = int(start_X * ratio_width * percent)
            start_Y = int(start_Y * ratio_height * percent)
            end_X = int(end_X * ratio_width * percent)
            end_Y = int(end_Y * ratio_height * percent)

            ROIImage = image.copy()[start_Y:end_Y, start_X:end_X]
            config = '--psm 6' if numbers else ''
            extracted_text.append(pytesseract.image_to_string(
                ROIImage, config=config)
            )

        return extracted_text

def box_extractor(scores, geometry, min_confidence):
    """
    Converts results from the forward pass to rectangles depicting text regions & their respective confidences
    :param scores: scores array from the model
    :param geometry: geometry array from the model
    :param min_confidence: minimum confidence required to pass the results forward
    :return: decoded rectangles & their respective confidences
    """
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    return rects, confidences
