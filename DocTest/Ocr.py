import logging
import tempfile
import time
import os
import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Optional, Sequence, Tuple
from pytesseract import Output
import urllib.request
import unicodedata

logger = logging.getLogger(__name__)

PYTESSERACT_CONFIDENCE=20
EAST_MODEL_URL = "https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb"
# A detected region taller than this many median EAST box heights holds
# several merged text lines and is recognized as a block, not a single line.
MULTILINE_RATIO = 1.6

def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")


def _is_word(text: Dict[str, List], index: int) -> bool:
    """True for a recognized word; False for Tesseract's structural rows
    (page/block/paragraph/line, confidence -1) and empty entries."""
    try:
        if float(text['conf'][index]) < 0:
            return False
    except (TypeError, ValueError):
        return False
    return bool(str(text['text'][index]).strip())


def _has_alnum_word(text: Dict[str, List]) -> bool:
    return any(
        _is_word(text, i) and any(ch.isalnum() for ch in str(text['text'][i]))
        for i in range(len(text.get('text', [])))
    )


def _crop(image: np.ndarray, box: Tuple[int, int, int, int], pad_x: int, pad_y: int) -> Tuple[np.ndarray, int, int]:
    """Crop *box* (x0, y0, x1, y1) plus padding, clamped to the image."""
    height, width = image.shape[:2]
    x0, y0 = max(0, box[0] - pad_x), max(0, box[1] - pad_y)
    x1, y1 = min(width, box[2] + pad_x), min(height, box[3] + pad_y)
    return image[y0:y1, x0:x1], x0, y0


def _median_box_height(rectangles: Sequence[Tuple[int, int, int, int]], x: int, y: int, w: int, h: int) -> Optional[float]:
    """Median height of the raw EAST boxes lying inside a region."""
    heights = [
        y2 - y1 for (x1, y1, x2, y2) in rectangles
        if x1 >= x - 1 and y1 >= y - 1 and x2 <= x + w + 1 and y2 <= y + h + 1
    ]
    return float(np.median(heights)) if heights else None

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
            # Touching text lines merge into one contour; a single-line read
            # (psm 7) of such a region silently drops all but one line.
            psm = 7
            line_height = end_y - start_y
            median_height = _median_box_height(rectangles, x, y, w, h)
            if median_height and h > MULTILINE_RATIO * median_height:
                psm = 6
                line_height = max(1, int(median_height * ratio_height))
            # ROI to be recognized; vertical padding follows one line's
            # height so it does not reach into neighbouring lines
            box = (start_x, start_y, end_x, end_y)
            roi, roi_start_x, roi_start_y = _crop(
                loaded_image, box, max(5, int((end_x - start_x) * 0.6)), max(2, int(line_height * 0.2))
            )
            if roi.size == 0:
                continue

            text = self._recognize_roi(roi, psm)
            if not _has_alnum_word(text):
                # Small labels in busy images drown in the wide context;
                # retry once with a tight crop around the detected text.
                tight, tight_x, tight_y = _crop(
                    loaded_image, box, max(4, int(line_height * 0.6)), max(2, int(line_height * 0.25))
                )
                if tight.size > 0:
                    retry = self._recognize_roi(tight, psm)
                    if _has_alnum_word(retry):
                        text, roi_start_x, roi_start_y = retry, tight_x, tight_y

            # recalculate coordinates in tesseract text dict and add start_x and start_y
            for i in range(len(text['text'])):
                if not _is_word(text, i):
                    continue
                text['text'][i] = remove_control_characters(text['text'][i])
                text['left'][i] = text['left'][i] + roi_start_x
                text['top'][i] = text['top'][i] + roi_start_y
                text['width'][i] = text['width'][i]
                text['height'][i] = text['height'][i]
                text['text'][i] = text['text'][i].strip()
                if not text['text'][i]:
                    continue
                text['conf'][i] = 100



                # collating results
                results['text'].append(text['text'][i])
                results['left'].append(text['left'][i])
                results['top'].append(text['top'][i])
                results['width'].append(text['width'][i])
                results['height'].append(text['height'][i])
                results['conf'].append(text['conf'][i])

        return results

    def _recognize_roi(self, roi: np.ndarray, psm: int = 7) -> Dict[str, List[str]]:
        config = f'--oem 1 --psm {psm} -l eng'
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
        return (resized_image, W, H, ratio_width, ratio_height)

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
        logger.info('text detection took %.6f seconds', end - start)
        return (scores, geometry)

    def _load_assets(self):
        self._get_east()
        start = time.time()
        self.east_net = cv2.dnn.readNet(self.east)
        end = time.time()
        logger.info('Loaded EAST text detector %.6f seconds ...', end - start)

    def _get_east(self):
        if os.path.exists(self.east):
            return

        # load the pre-trained EAST model for text detection
        pkg_path = os.path.dirname(__file__)
        data_file = os.path.join(pkg_path, self.east)
        target_dir = os.path.dirname(data_file)
        os.makedirs(target_dir, exist_ok=True)
        if os.path.isfile(data_file):
            return

        # Download into a private temp file and atomically move it into place:
        # concurrent processes (xdist workers, pabot) must never see a
        # half-written model at the final path.
        logger.info("Downloading frozen_east_text_detection.pb from url")
        fd, tmp_path = tempfile.mkstemp(prefix=".east-", suffix=".part", dir=target_dir)
        os.close(fd)
        try:
            urllib.request.urlretrieve(EAST_MODEL_URL, tmp_path)
            try:
                os.replace(tmp_path, data_file)
            except PermissionError:
                # Windows: another process finished first and holds the
                # model open; its complete file is just as good.
                if not os.path.isfile(data_file):
                    raise
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

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
