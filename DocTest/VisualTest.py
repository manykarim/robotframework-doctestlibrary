import json
import logging
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import imutils
import numpy as np
from assertionengine import AssertionOperator, verify_assertion
from robot.api.deco import keyword, library
from robot.libraries.BuiltIn import BuiltIn

from DocTest.DocumentRepresentation import DocumentRepresentation
from DocTest.Downloader import download_file_from_url, is_url
from DocTest.config import DEFAULT_CONFIDENCE
from DocTest.llm import LLMDependencyError, load_llm_settings
from DocTest.TextNormalization import apply_character_replacements

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from DocTest.llm.types import LLMDecision, LLMDecisionLabel
else:  # pragma: no cover - runtime fallback
    LLMDecision = Any  # type: ignore[misc, assignment]
    LLMDecisionLabel = Any  # type: ignore[misc, assignment]

LOG = logging.getLogger(__name__)


_VISUAL_LLM_RUNTIME: Optional[Tuple[Any, Any, Any]] = None


def _coerce_label_value(label: Any) -> str:
    value = getattr(label, "value", label)
    return str(value)


def _decision_equals_flag(label: Any, enum_cls: Any) -> bool:
    candidate = _coerce_label_value(label).lower()
    if enum_cls is None:
        return candidate == "flag"
    flag_member = getattr(enum_cls, "FLAG", None)
    if flag_member is None:
        return candidate == "flag"
    flag_value = _coerce_label_value(flag_member).lower()
    return candidate == flag_value


def _load_visual_llm_runtime() -> Tuple[Any, Any, Any]:
    global _VISUAL_LLM_RUNTIME
    if _VISUAL_LLM_RUNTIME is not None:
        return _VISUAL_LLM_RUNTIME
    try:
        from DocTest.llm.client import assess_visual_diff, create_binary_content
        from DocTest.llm.types import LLMDecisionLabel as _LLMDecisionLabel
    except Exception as exc:  # pragma: no cover - optional dependency missing
        raise LLMDependencyError() from exc
    _VISUAL_LLM_RUNTIME = (assess_visual_diff, create_binary_content, _LLMDecisionLabel)
    return _VISUAL_LLM_RUNTIME


@library
class VisualTest:
    ROBOT_LIBRARY_VERSION = 1.0
    DPI_DEFAULT = 200
    OCR_ENGINE_DEFAULT = "tesseract"
    MOVEMENT_DETECTION_DEFAULT = "template"
    MOVEMENT_DETECTION_METHODS = {"template", "orb", "sift", "text"}
    MOVEMENT_DETECTION_ALIASES = {"classic": "template"}
    PARTIAL_IMAGE_THRESHOLD_DEFAULT = 0.1
    SIFT_RATIO_THRESHOLD_DEFAULT = 0.75
    SIFT_MIN_MATCHES_DEFAULT = 2
    ORB_MAX_MATCHES_DEFAULT = 10
    ORB_MIN_MATCHES_DEFAULT = 2
    RANSAC_THRESHOLD_DEFAULT = 5.0
    WATERMARK_WIDTH = 31
    WATERMARK_HEIGHT = 36
    WATERMARK_CENTER_OFFSET = 3 / 100
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    BOTTOM_LEFT_CORNER_OF_TEXT = (20, 60)
    FONT_SCALE = 0.7
    FONT_COLOR = (255, 0, 0)
    LINE_TYPE = 2
    REFERENCE_LABEL = "Expected Result (Reference)"
    CANDIDATE_LABEL = "Actual Result (Candidate)"

    def __init__(
        self,
        threshold: float = 0.0,
        dpi: int = DPI_DEFAULT,
        take_screenshots: bool = False,
        show_diff: bool = False,
        ocr_engine: Literal["tesseract", "east"] = OCR_ENGINE_DEFAULT,
        screenshot_format: str = "jpg",
        embed_screenshots: bool = False,
        force_ocr: bool = False,
        watermark_file: str = None,
        movement_detection: Literal[
            "template", "orb", "sift", "text"
        ] = MOVEMENT_DETECTION_DEFAULT,
        partial_image_threshold: float = PARTIAL_IMAGE_THRESHOLD_DEFAULT,
        sift_ratio_threshold: float = SIFT_RATIO_THRESHOLD_DEFAULT,
        sift_min_matches: int = SIFT_MIN_MATCHES_DEFAULT,
        orb_max_matches: int = ORB_MAX_MATCHES_DEFAULT,
        orb_min_matches: int = ORB_MIN_MATCHES_DEFAULT,
        ransac_threshold: float = RANSAC_THRESHOLD_DEFAULT,
        stream_documents: bool = True,
        document_page_cache_size: int = 2,
        verbose_movement_logging: bool = False,
        character_replacements: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Initialize the VisualTest library.

        | =Arguments= | =Description= |
        | ``threshold`` | Threshold for image comparison. Value is between 0.0 and 1.0. A higher value will tolerate more differences. Default is 0.0. |
        | ``dpi`` | DPI to be used for image processing. Default is 200. |
        | ``take_screenshots`` | Whether to take screenshots during image comparison. Default is False. |
        | ``show_diff`` | Whether to show diff screenshot in the images during comparison. Default is False. |
        | ``ocr_engine`` | OCR engine to be used for text extraction. Options are ``tesseract`` and ``east``. Default is ``tesseract``. |
        | ``screenshot_format`` | Format of the screenshots to be saved. Options are ``jpg`` and ``png``. Default is ``jpg``. |
        | ``embed_screenshots`` | Whether to embed screenshots as base64 in the log. Default is False. |
        | ``force_ocr`` | Whether to force OCR during image comparison. Default is False. |
        | ``watermark_file`` | Path to an image/document or a folder containing multiple images. They shall only contain a ```solid black`` area of the parts that shall be ignored for visual comparisons |
        | ``movement_detection`` | Method to be used for movement detection. Options are ``template``, ``orb``, ``sift`` or ``text`` (text-based). Default is ``template``. |
        | ``partial_image_threshold`` | The threshold used to identify partial images, e.g. for movement detection. Value is between 0.0 and 1.0. A higher value will tolerate more differences. Default is ``0.1``. |
        | ``sift_ratio_threshold`` | Lowe's ratio test threshold for SIFT feature matching. Lower values are more restrictive. Default is ``0.75``. |
        | ``sift_min_matches`` | Minimum number of good matches required for SIFT homography computation. Default is ``4``. |
        | ``orb_max_matches`` | Maximum number of matches to use for ORB feature matching. Default is ``10``. |
        | ``orb_min_matches`` | Minimum number of matches required for ORB homography computation. Default is ``4``. |
        | ``ransac_threshold`` | RANSAC threshold for robust homography estimation. Higher values are more tolerant of outliers. Default is ``5.0``. |
        | ``stream_documents`` | When ``True``, VisualTest renders PDF pages lazily instead of loading entire documents into memory. Default is ``True``. |
        | ``document_page_cache_size`` | Maximum number of rendered pages to keep in memory per document when streaming. Default is ``2``. |
        | ``verbose_movement_logging`` | When ``True``, emit detailed warnings from movement detection helpers (template, ORB, SIFT). Disabled by default to reduce noise. |
        | ``character_replacements`` | Dict mapping characters to replacements, applied to text extraction results. Example: ``{'\u00A0': ' '}`` to normalize non-breaking spaces. |
        | ``**kwargs`` | Everything else. |


        """
        self.threshold = threshold
        self.dpi = dpi
        self.take_screenshots = take_screenshots
        self.show_diff = show_diff
        self.ocr_engine = ocr_engine
        self.screenshot_format = (
            screenshot_format if screenshot_format in ["jpg", "png"] else "jpg"
        )
        self.embed_screenshots = embed_screenshots
        self.screenshot_dir = Path("screenshots")
        self.watermark_file = watermark_file
        movement_method = movement_detection.lower()
        movement_method = self.MOVEMENT_DETECTION_ALIASES.get(movement_method, movement_method)
        if movement_method not in self.MOVEMENT_DETECTION_METHODS:
            raise ValueError(
                f"Unsupported movement detection method '{movement_detection}'. "
                f"Supported values are: {', '.join(sorted(self.MOVEMENT_DETECTION_METHODS))}."
            )
        self.movement_detection = movement_method
        self.partial_image_threshold = partial_image_threshold
        self.sift_ratio_threshold = sift_ratio_threshold
        self.sift_min_matches = sift_min_matches
        self.orb_max_matches = orb_max_matches
        self.orb_min_matches = orb_min_matches
        self.ransac_threshold = ransac_threshold
        self.force_ocr = force_ocr
        self.stream_documents = bool(stream_documents)
        self.document_page_cache_size = max(1, int(document_page_cache_size))
        self.verbose_movement_logging = bool(verbose_movement_logging)
        self.character_replacements = self._parse_character_replacements(character_replacements)

        output_dir, reference_run, pabot_index = self._read_robot_variables()
        self.output_directory = output_dir
        self.reference_run = reference_run
        self.PABOTQUEUEINDEX = pabot_index

        self.screenshot_path = self.output_directory / self.screenshot_dir

    def _log_verbose_warning(self, message: str) -> None:
        """Emit movement-detection warnings only when verbose logging is enabled."""
        if not self.verbose_movement_logging:
            return
        LOG.warning(message)

    def _read_robot_variables(self) -> Tuple[Path, bool, Optional[str]]:
        """Return Robot Framework runtime variables if Robot is active."""
        try:
            built_in = BuiltIn()
            output_dir = built_in.get_variable_value("${OUTPUT DIR}")
            reference_run = built_in.get_variable_value("${REFERENCE_RUN}", False)
            pabot_index = built_in.get_variable_value("${PABOTQUEUEINDEX}")
        except Exception:
            print("Robot Framework is not running")
            return Path.cwd(), False, None

        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path.cwd()

        return output_path, bool(reference_run), pabot_index

    def _parse_character_replacements(
        self, value: Optional[Any]
    ) -> Optional[Dict[str, str]]:
        """Parse character_replacements from various input formats."""
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                LOG.warning(f"Invalid character_replacements JSON: {value}")
        return None

    @keyword
    def compare_images(
        self,
        reference_image: str,
        candidate_image: str,
        placeholder_file: Union[str, dict, list] = None,
        check_text_content: bool = False,
        move_tolerance: int = None,
        movement_detection: Optional[str] = None,
        contains_barcodes: bool = False,
        watermark_file: Optional[Union[str, list]] = None,
        ignore_watermarks: bool = None,
        force_ocr: bool = False,
        ocr_engine: Literal["tesseract", "east"] = None,
        DPI: int = None,
        resize_candidate: bool = False,
        blur: bool = False,
        threshold: float = None,
        mask: Union[str, dict, list] = None,
        get_pdf_content: bool = False,
        block_based_ssim: bool = False,
        block_size: int = 32,
        **kwargs,
    ):
        """Compares the documents/images ``reference_image`` and ``test_image``.

        Result is passed if no visual differences are detected.

        | =Arguments= | =Description= |
        | ``reference_image`` | Path or URL of the Reference Image/Document, your expected result. May be .pdf, .ps, .pcl or image files |
        | ``candidate_image`` | Path or URL of the Candidate Image/Document, that's the one you want to test. May be .pdf, .ps, .pcl or image files |
        | ``placeholder_file`` | Path to a ``.json`` which defines areas that shall be ignored for comparison. Those parts will be replaced with solid placeholders  |
        | ``mask`` | Same purpose as ``placeholder_file`` but instead of a file path, this is either ``json`` , a ``dict`` , a ``list`` or a ``string`` which defines the areas to be ignored  |
        | ``check_text_content`` | In case of visual differences: Is it acceptable, if only the text content in the different areas is equal |
        | ``move_tolerance`` | In case of visual differences: Is is acceptable, if only parts in the different areas are moved by ``move_tolerance`` pixels  |
        | ``movement_detection`` | Override movement detection method for this comparison. Options: ``template``, ``orb``, ``sift``, ``text``. Defaults to the library-level setting. |
        | ``contains_barcodes`` | Shall the image be scanned for barcodes and shall their content be checked (currently only data matrices are supported) |
        | ``get_pdf_content`` | Only relevant in case of using ``move_tolerance`` and ``check_text_content``: Shall the PDF Content like Texts and Boxes be used for calculations |
        | ``force_ocr`` | Always use OCR to find Texts in Images, even for PDF Documents |
        | ``DPI`` | Resolution in which documents are rendered before comparison |
        | ``watermark_file`` | Path to an image/document, a folder containing multiple images, or a list of paths. They shall only contain a ```solid black`` area of the parts that shall be ignored for visual comparisons |
        | ``ignore_watermarks`` | Ignores a very special watermark in the middle of the document |
        | ``ocr_engine`` | Use ``tesseract`` or ``east`` for Text Detection and OCR |
        | ``resize_candidate`` | Allow visual comparison, even of documents have different sizes |
        | ``blur`` | Blur the image before comparison to reduce visual difference caused by noise |
        | ``threshold`` | Threshold for visual comparison between 0.0000 and 1.0000 . Default is 0.0000. Higher values mean more tolerance for visual differences. |
        | ``block_based_ssim`` | Uses additional block based block-based comparison, to catch differences in smaller areas. Makes only sense, for ``threshold`` > 0 . Default is `False` |
        | ``block_size`` | Size of the blocks for block-based comparison. Default is 32. Only relevant for ``block_based_ssim`` |
        | ``llm`` / ``llm_enabled`` | When ``${True}``, summarise differences and forward them to the configured LLM. Default ``${False}``. |
        | ``llm_override`` | Allow an approving LLM verdict to override SSIM/structural failures. Default ``${False}``. |
        | ``llm_prompt`` / ``llm_visual_prompt`` | Custom per-call LLM prompt. Default ``None`` uses the library default. |
        | ``**kwargs`` | Everything else |


        Examples:
        | `Compare Images`   reference.pdf   candidate.pdf                                  #Performs a pixel comparison of both files
        | `Compare Images`   reference.pdf (not existing)    candidate.pdf                  #Will always return passed and save the candidate.pdf as reference.pdf
        | `Compare Images`   reference.pdf   candidate.pdf   placeholder_file=mask.json     #Performs a pixel comparison of both files and excludes some areas defined in mask.json
        | `Compare Images`   reference.pdf   candidate.pdf   contains_barcodes=${true}      #Identified barcodes in documents and excludes those areas from visual comparison. The barcode data will be checked instead
        | `Compare Images`   reference.pdf   candidate.pdf   check_text_content${true}      #In case of visual differences, the text content in the affected areas will be identified using OCR. If text content it equal, the test is considered passed
        | `Compare Images`   reference.pdf   candidate.pdf   move_tolerance=10              #In case of visual differences, it is checked if difference is caused only by moved areas. If the move distance is within 10 pixels the test is considered as passed. Else it is failed
        | `Compare Images`   reference.pdf   candidate.pdf   move_tolerance=20   movement_detection=text   #Use text-based movement detection (PDF/OCR) instead of image feature matching
        | `Compare Images`   reference.pdf   candidate.pdf   check_text_content=${true}   get_pdf_content=${true}   #In case of visual differences, the text content in the affected areas will be read directly from  PDF (not OCR). If text content it equal, the test is considered passed
        | `Compare Images`   reference.pdf   candidate.pdf   watermark_file=watermark.pdf     #Provides a watermark file as an argument. In case of visual differences, watermark content will be subtracted
        | `Compare Images`   reference.pdf   candidate.pdf   watermark_file=${CURDIR}${/}watermarks     #Provides a watermark folder as an argument. In case of visual differences, all watermarks in folder will be subtracted
        | @{watermarks}    Create List    watermark1.pdf    watermark2.pdf
        | `Compare Images`   reference.pdf   candidate.pdf   watermark_file=${watermarks}     #Provides a list of watermark files. Individual files will be checked first, then combined if individual files don't cover all differences
        | `Compare Images`   reference.pdf   candidate.pdf   move_tolerance=10   get_pdf_content=${true}   #In case of visual differences, it is checked if difference is caused only by moved areas. Move distance is identified directly from PDF data. If the move distance is within 10 pixels the test is considered as passed. Else it is failed
        | `Compare Images`   reference.pdf   candidate.pdf   llm=${True}   llm_prompt=Please output JSON decision | Summarise differences and ask the configured LLM to decide whether the mismatch may pass

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
        llm_requested = bool(
            kwargs.pop("llm", False)
            or kwargs.pop("llm_enabled", False)
            or kwargs.pop("use_llm", False)
        )
        llm_override_result = bool(kwargs.pop("llm_override", False))
        llm_overrides: Dict[str, Optional[str]] = {}
        llm_runtime_notes: List[str] = []
        custom_llm_prompt = kwargs.pop("llm_prompt", None)
        visual_prompt_override = kwargs.pop("llm_visual_prompt", None)
        if visual_prompt_override:
            custom_llm_prompt = visual_prompt_override
        llm_keys = [
            "llm_provider",
            "llm_models",
            "llm_vision_models",
            "llm_api_key",
            "llm_base_url",
            "llm_temperature",
            "llm_max_output_tokens",
            "llm_request_timeout",
            "azure_openai_endpoint",
            "azure_openai_deployment",
            "azure_openai_api_version",
            "azure_openai_api_key",
        ]
        for key in llm_keys:
            if key in kwargs:
                value = kwargs.pop(key)
                if value is not None:
                    llm_overrides[key] = value

        stream_documents_override = kwargs.pop("stream_documents", None)
        page_cache_size_override = kwargs.pop("page_cache_size", None)
        if page_cache_size_override is None:
            page_cache_size_value = self.document_page_cache_size
        else:
            page_cache_size_value = max(1, int(page_cache_size_override))
        stream_documents_flag = (
            self.stream_documents
            if stream_documents_override is None
            else bool(stream_documents_override)
        )

        # Download files if URLs are provided
        if is_url(reference_image):
            reference_image = download_file_from_url(reference_image)
        if is_url(candidate_image):
            candidate_image = download_file_from_url(candidate_image)

        # Set DPI and threshold if provided
        dpi = DPI if DPI else self.dpi
        threshold = threshold if threshold is not None else self.threshold

        # Set OCR engine if provided
        ocr_engine = ocr_engine if ocr_engine else self.ocr_engine

        if watermark_file is None:
            watermark_file = self.watermark_file
        if ignore_watermarks is None:
            ignore_watermarks = os.getenv("IGNORE_WATERMARKS", False)

        force_ocr = force_ocr or self.force_ocr
        detection_method = (
            movement_detection or self.movement_detection or self.MOVEMENT_DETECTION_DEFAULT
        ).lower()
        detection_method = self.MOVEMENT_DETECTION_ALIASES.get(detection_method, detection_method)
        if detection_method not in self.MOVEMENT_DETECTION_METHODS:
            raise ValueError(
                f"Unsupported movement detection method '{detection_method}'. "
                f"Supported values are: {', '.join(sorted(self.MOVEMENT_DETECTION_METHODS))}."
            )

        # Load reference and candidate documents
        force_kwargs = {"force_ocr": force_ocr} if force_ocr else {}
        reference_doc = DocumentRepresentation(
            reference_image,
            dpi=dpi,
            ocr_engine=ocr_engine,
            ignore_area_file=placeholder_file,
            ignore_area=mask,
            stream_pages=stream_documents_flag,
            page_cache_size=page_cache_size_value,
            **force_kwargs,
            **kwargs,
        )
        candidate_doc = DocumentRepresentation(
            candidate_image,
            dpi=dpi,
            ocr_engine=ocr_engine,
            stream_pages=stream_documents_flag,
            page_cache_size=page_cache_size_value,
            **force_kwargs,
            **kwargs,
        )

        watermarks = []

        # Apply ignore areas if provided
        abstract_ignore_areas = None

        detected_differences: List[Dict[str, Any]] = []
        try:
            # Compare visual content through the Page class
            for ref_page, cand_page in reference_doc.iter_page_pairs(candidate_doc):
                page_notes: List[str] = []
                diff_rectangles_cache: Optional[List[Dict[str, int]]] = None
                combined_image_with_differences = None
                # Resize the candidate page if needed
                if resize_candidate and ref_page.image.shape != cand_page.image.shape:
                    cand_page.image = cv2.resize(
                        cand_page.image, (ref_page.image.shape[1], ref_page.image.shape[0])
                    )

                # Check if dimensions are different
                if ref_page.image.shape != cand_page.image.shape:
                    detected_differences.append(
                        {
                            "ref_page": ref_page,
                            "cand_page": cand_page,
                            "message": "Image dimensions are different.",
                            "rectangles": [],
                            "score": None,
                            "threshold": threshold,
                            "absolute_diff": None,
                            "combined_diff": None,
                            "notes": page_notes[:],
                        }
                    )
                    combined_image = self.concatenate_images_safely(
                        ref_page.image, cand_page.image, axis=1, fill_color=(240, 240, 240)
                    )
                    self.add_screenshot_to_log(
                        combined_image, suffix="_combined", original_size=False
                    )
                    continue

                similar, diff, thresh, absolute_diff, score = ref_page.compare_with(
                    cand_page,
                    threshold=threshold,
                    blur=blur,
                    block_based_ssim=block_based_ssim,
                    block_size=block_size,
                )

                if self.take_screenshots:
                    # Save original images to the screenshot directory and add them to the Robot Framework log
                    # But add them next to each other in the log
                    # Use safe concatenation to handle different image dimensions
                    combined_image = self.concatenate_images_safely(
                        ref_page.image, cand_page.image, axis=1, fill_color=(240, 240, 240)
                    )
                    self.add_screenshot_to_log(
                        combined_image, suffix="_combined", original_size=False
                    )

                if not similar and ignore_watermarks:
                    # Get bounding rect of differences
                    diff_rectangles = self.get_diff_rectangles(absolute_diff)
                    diff_rectangles_cache = diff_rectangles
                    # Check if the differences are only in the watermark area
                    if len(diff_rectangles) == 1:
                        diff_rect = diff_rectangles[0]
                        x, y, w, h = (
                            diff_rect["x"],
                            diff_rect["y"],
                            diff_rect["width"],
                            diff_rect["height"],
                        )
                        diff_center_x = abs((x + w / 2) - ref_page.image.shape[1] / 2)
                        diff_center_y = abs((y + h / 2) - ref_page.image.shape[0] / 2)
                        if (
                            diff_center_x
                            < ref_page.image.shape[1] * self.WATERMARK_CENTER_OFFSET
                            and (w * 25.4 / dpi < self.WATERMARK_WIDTH)
                            and (h * 25.4 / dpi < self.WATERMARK_HEIGHT)
                        ):
                            similar = True
                            print("Visual differences are only in the watermark area.")

                if not similar and watermark_file:
                    if watermarks == []:
                        watermarks = self.load_watermarks(watermark_file)

                    # First, try each watermark mask individually
                    for mask in watermarks:
                        if (
                            mask.shape[0] != ref_page.image.shape[0]
                            or mask.shape[1] != ref_page.image.shape[1]
                        ):
                            # Resize mask to match thresh
                            mask = cv2.resize(
                                mask, (ref_page.image.shape[1], ref_page.image.shape[0])
                            )

                        mask_inv = cv2.bitwise_not(mask)
                        # dilate the mask to account for slight misalignments
                        mask_inv = cv2.dilate(mask_inv, None, iterations=2)
                        result = cv2.subtract(absolute_diff, mask_inv)
                        if (
                            cv2.countNonZero(cv2.subtract(absolute_diff, mask_inv)) == 0
                            or cv2.countNonZero(cv2.subtract(thresh, mask_inv)) == 0
                        ):
                            similar = True
                            print(
                                "A watermark file was provided. After removing watermark area, both images are equal"
                            )
                            break

                    # If individual watermarks don't work and there are multiple watermarks,
                    # try combining all watermark files by merging their areas
                    if not similar and len(watermarks) > 1:
                        try:
                            print(
                                f"Individual watermarks failed. Attempting to merge {len(watermarks)} watermark masks..."
                            )

                            # Create combined mask by merging all watermark areas
                            # Initialize with all white (all areas to be ignored initially)
                            # We'll AND with each mask to get union of black comparison areas
                            combined_mask = np.full(
                                (ref_page.image.shape[0], ref_page.image.shape[1]),
                                255,
                                dtype=np.uint8,
                            )

                            total_individual_pixels = 0
                            for i, mask in enumerate(watermarks):
                                # Ensure all masks have the same dimensions
                                if (
                                    mask.shape[0] != ref_page.image.shape[0]
                                    or mask.shape[1] != ref_page.image.shape[1]
                                ):
                                    mask = cv2.resize(
                                        mask,
                                        (ref_page.image.shape[1], ref_page.image.shape[0]),
                                    )

                                # Count pixels for debugging
                                mask_pixels = np.sum(mask > 0)
                                total_individual_pixels += mask_pixels
                                print(f"  Watermark {i + 1}: {mask_pixels} white pixels")

                                # Add debugging screenshot for individual watermarks
                                if self.take_screenshots:
                                    self.add_screenshot_to_log(
                                        mask,
                                        suffix=f"_individual_watermark_{i + 1}",
                                        original_size=False,
                                    )

                                # Merge watermark areas: any black pixel in any mask becomes black in combined mask
                                # This creates a union of all black comparison areas
                                # Note: Using AND operation because we want union of black (0) pixels
                                # OR would give intersection of black areas, AND gives union of black areas
                                combined_mask = cv2.bitwise_and(combined_mask, mask)

                            if combined_mask is not None:
                                combined_black_pixels = np.sum(combined_mask == 0)
                                print(
                                    f"  Combined mask: {combined_black_pixels} black pixels (union of all comparison areas)"
                                )

                                # Apply morphological operations to clean up the combined mask
                                kernel = cv2.getStructuringElement(
                                    cv2.MORPH_ELLIPSE, (3, 3)
                                )
                                combined_mask = cv2.morphologyEx(
                                    combined_mask, cv2.MORPH_CLOSE, kernel
                                )

                                combined_mask_inv = cv2.bitwise_not(combined_mask)
                                # dilate the combined mask to account for slight misalignments
                                combined_mask_inv = cv2.dilate(
                                    combined_mask_inv, None, iterations=2
                                )

                                # Take screenshots if enabled
                                if self.take_screenshots:
                                    # Screenshot of the original combined watermark mask (before inversion)
                                    self.add_screenshot_to_log(
                                        combined_mask,
                                        suffix="_combined_watermark_mask_original",
                                        original_size=False,
                                    )

                                    # Screenshot of the inverted dilated mask
                                    self.add_screenshot_to_log(
                                        combined_mask_inv,
                                        suffix="_combined_watermark_mask_inverted",
                                        original_size=False,
                                    )

                                    # Screenshot showing differences before watermark removal
                                    self.add_screenshot_to_log(
                                        absolute_diff,
                                        suffix="_differences_before_watermark_removal",
                                        original_size=False,
                                    )

                                    # Screenshot showing differences after watermark removal
                                    diff_after_watermark = cv2.subtract(
                                        absolute_diff, combined_mask_inv
                                    )
                                    self.add_screenshot_to_log(
                                        diff_after_watermark,
                                        suffix="_differences_after_watermark_removal",
                                        original_size=False,
                                    )

                                if (
                                    cv2.countNonZero(
                                        cv2.subtract(absolute_diff, combined_mask_inv)
                                    )
                                    == 0
                                    or cv2.countNonZero(
                                        cv2.subtract(thresh, combined_mask_inv)
                                    )
                                    == 0
                                ):
                                    similar = True
                                    print(
                                        "Multiple watermark files were provided. After removing combined watermark areas, both images are equal"
                                    )

                                # Print diff image with combined watermark
                                if self.take_screenshots:
                                    # Ensure both images have the same number of channels
                                    if len(absolute_diff.shape) == 2:
                                        absolute_diff_rgb = cv2.cvtColor(
                                            absolute_diff, cv2.COLOR_GRAY2BGR
                                        )
                                    else:
                                        absolute_diff_rgb = absolute_diff

                                    if len(combined_mask_inv.shape) == 2:
                                        combined_mask_inv_rgb = cv2.cvtColor(
                                            combined_mask_inv, cv2.COLOR_GRAY2BGR
                                        )
                                    else:
                                        combined_mask_inv_rgb = combined_mask_inv

                                    combined_diff = self.blend_two_images(
                                        absolute_diff_rgb, combined_mask_inv_rgb
                                    )
                                    self.add_screenshot_to_log(
                                        combined_diff,
                                        suffix="_combined_watermark_diff_blend",
                                    )
                        except Exception as e:
                            LOG.warning(f"Failed to combine watermark masks: {str(e)}")

                if check_text_content and not similar:
                    similar = True
                    diff_rectangles = (
                        diff_rectangles_cache
                        if diff_rectangles_cache is not None
                        else self.get_diff_rectangles(absolute_diff)
                    )
                    diff_rectangles_cache = diff_rectangles
                    # Compare text content only in the areas that have differences
                    for rect in diff_rectangles:
                        same_text, ref_area_text, cand_area_text = (
                            ref_page._compare_text_content_in_area_with(
                                cand_page, rect, force_ocr
                            )
                        )
                        # Save the reference and candidate areas as images and add them to the log
                        reference_area = ref_page.get_area(rect)
                        candidate_area = cand_page.get_area(rect)
                        self.add_screenshot_to_log(
                            reference_area, suffix="_reference_area", original_size=False
                        )
                        self.add_screenshot_to_log(
                            candidate_area, suffix="_candidate_area", original_size=False
                        )
                        if not same_text:
                            similar = False
                            page_notes.append(
                                f"Rect {rect} has different text. "
                                f"Reference: {ref_area_text!r} | Candidate: {cand_area_text!r}"
                            )
                            # Add log message with the text content differences
                            # Add screenshots to the log of the reference and candidate areas

                            print(
                                f"Text content in the area {rect} differs:\n\nReference Text:\n{ref_area_text}\n\nCandidate Text:\n{cand_area_text}"
                            )
                        else:
                            page_notes.append(
                                f"Rect {rect} visual change but identical text."
                            )
                            print(
                                f"Visual differences in the area {rect} but text content is the same."
                            )
                            print(
                                f"Reference Text:\n{ref_area_text}\n\nCandidate Text:\n{cand_area_text}"
                            )

                if move_tolerance and int(move_tolerance) > 0 and not similar:
                    diff_rectangles = (
                        diff_rectangles_cache
                        if diff_rectangles_cache is not None
                        else self.get_diff_rectangles(absolute_diff)
                    )
                    diff_rectangles_cache = diff_rectangles
                    has_pdf_text = bool(ref_page.pdf_text_words) and bool(cand_page.pdf_text_words)
                    if get_pdf_content:
                        import fitz
                        fitz.TOOLS.set_aa_level(0)
                        similar = True
                        ref_words = ref_page.pdf_text_words
                        cand_words = cand_page.pdf_text_words

                        # If no words are fount, proceed with nornmal tolerance check and set check_pdf_content to False
                        if len(ref_words) == 0 or len(cand_words) == 0:
                            check_pdf_content = False
                            print(
                                "No pdf layout elements found. Proceeding with normal tolerance check."
                            )

                    auto_use_pdf_content = get_pdf_content
                    auto_detection_method = detection_method

                    if has_pdf_text and detection_method != "text":
                        auto_detection_method = "text"
                        auto_use_pdf_content = True

                    text_based_detection = (
                        auto_detection_method == "text" or auto_use_pdf_content
                    )

                    if text_based_detection:
                        similar = self._check_movement_with_text(
                            ref_page=ref_page,
                            cand_page=cand_page,
                            diff_rectangles=diff_rectangles,
                            move_tolerance=move_tolerance,
                            detection_method=auto_detection_method,
                            use_pdf_content=auto_use_pdf_content,
                            force_ocr=force_ocr,
                        )
                    else:
                        similar = self._check_movement_with_images(
                            ref_page=ref_page,
                            cand_page=cand_page,
                            diff_rectangles=diff_rectangles,
                            move_tolerance=move_tolerance,
                            detection_method=auto_detection_method,
                        )

                if not similar:
                    # Save original images to the screenshot directory and add them to the Robot Framework log
                    # But add them next to each other in the log
                    # Use safe concatenation to handle different image dimensions
                    cv2.putText(
                        ref_page.image,
                        self.REFERENCE_LABEL,
                        self.BOTTOM_LEFT_CORNER_OF_TEXT,
                        self.FONT,
                        self.FONT_SCALE,
                        self.FONT_COLOR,
                        self.LINE_TYPE,
                    )
                    cv2.putText(
                        cand_page.image,
                        self.CANDIDATE_LABEL,
                        self.BOTTOM_LEFT_CORNER_OF_TEXT,
                        self.FONT,
                        self.FONT_SCALE,
                        self.FONT_COLOR,
                        self.LINE_TYPE,
                    )

                    combined_image = self.concatenate_images_safely(
                        ref_page.image, cand_page.image, axis=1, fill_color=(240, 240, 240)
                    )
                    self.add_screenshot_to_log(
                        combined_image, suffix="_combined", original_size=False
                    )

                    # Generate side-by-side image highlighting differences using the SSIM diff image
                    reference_img, candidate_img, _ = (
                        self.get_images_with_highlighted_differences(
                            thresh, ref_page.image, cand_page.image
                        )
                    )

                    combined_image_with_differences = self.concatenate_images_safely(
                        reference_img, candidate_img, axis=1, fill_color=(240, 240, 240)
                    )
                    # Add the side-by-side comparison with differences to the Robot Framework log
                    self.add_screenshot_to_log(
                        combined_image_with_differences,
                        suffix="_combined_with_diff",
                        original_size=False,
                    )

                    # Add absolute difference image to the log
                    self.add_screenshot_to_log(
                        absolute_diff, suffix="_absolute_diff", original_size=False
                    )

                    if diff_rectangles_cache is None:
                        diff_rectangles_cache = self.get_diff_rectangles(absolute_diff)

                    detected_differences.append(
                        {
                            "ref_page": ref_page,
                            "cand_page": cand_page,
                            "message": f"Visual differences detected. SSIM score: {score:.20f}",
                            "rectangles": diff_rectangles_cache,
                            "score": score,
                            "threshold": threshold,
                            "absolute_diff": absolute_diff.copy() if absolute_diff is not None else None,
                            "combined_diff": combined_image_with_differences.copy()
                            if combined_image_with_differences is not None
                            else None,
                            "notes": page_notes[:],
                        }
                    )

            llm_decision = None
            llm_label_enum = None
            if detected_differences and llm_requested:
                llm_runtime_notes.append(f"Threshold: {threshold}")
                if check_text_content:
                    llm_runtime_notes.append("Text content verification was enabled.")
                if move_tolerance:
                    llm_runtime_notes.append(f"Movement tolerance: {move_tolerance}")

                try:
                    assess_visual_diff_fn, create_binary_content_fn, llm_label_enum = (
                        _load_visual_llm_runtime()
                    )
                except LLMDependencyError as exc:
                    print(str(exc))
                else:
                    llm_decision = self._handle_llm_for_visual_differences(
                        reference_image=reference_image,
                        candidate_image=candidate_image,
                        differences=detected_differences,
                        overrides=llm_overrides,
                        notes=llm_runtime_notes,
                        custom_prompt=custom_llm_prompt,
                        create_binary_content_fn=create_binary_content_fn,
                        assess_visual_diff_fn=assess_visual_diff_fn,
                    )

                if llm_decision:
                    decision_value = _coerce_label_value(llm_decision.decision)
                    print(
                        f"LLM decision: {decision_value} "
                        f"(confidence={llm_decision.confidence!r}) - {llm_decision.reason}"
                    )
                    if llm_decision.notes:
                        print(f"LLM notes: {llm_decision.notes}")
                    if llm_override_result and llm_decision.is_positive:
                        print("LLM approved differences. Overriding baseline failure.")
                        detected_differences = []
                    elif _decision_equals_flag(llm_decision.decision, llm_label_enum):
                        print("LLM returned FLAG - keeping original comparison result.")
                    elif not llm_decision.is_positive and llm_override_result:
                        print("LLM rejected differences. Baseline failure will be raised.")

            for diff in detected_differences:
                print(diff["message"])
                self._raise_comparison_failure()

            print("Images/Document comparison passed.")
        finally:
            reference_doc.close()
            candidate_doc.close()

    @keyword
    def compare_images_with_llm(self, *args, llm_override: bool = False, **kwargs):
        """Compare images while consulting the configured LLM before raising a failure.

        | =Arguments= | =Description= |
        | ``*args`` | Positional arguments passed through to ``Compare Images`` (typically reference and candidate paths). |
        | ``llm_override`` | When ``${True}``, a positive LLM verdict overrides baseline SSIM failures. Default ``${False}``. |
        | ``**kwargs`` | Keyword arguments forwarded to ``Compare Images``. Use ``llm_prompt`` or ``llm_visual_prompt`` to customise the LLM prompt. |

        Example:
        | `Compare Images With LLM`   reference.pdf   candidate.pdf   llm_override=${True}   llm_prompt=Always respond with JSON
        """

        kwargs["llm_enabled"] = kwargs.get("llm_enabled", True)
        if "llm_override" not in kwargs:
            kwargs["llm_override"] = llm_override
        return self.compare_images(*args, **kwargs)

    def _handle_llm_for_visual_differences(
        self,
        reference_image: str,
        candidate_image: str,
        differences: List[Dict[str, Any]],
        overrides: Dict[str, Optional[str]],
        notes: List[str],
        custom_prompt: Optional[str],
        create_binary_content_fn: Optional[Any] = None,
        assess_visual_diff_fn: Optional[Any] = None,
    ) -> Optional[LLMDecision]:
        overrides = {key: (str(value) if value is not None else None) for key, value in overrides.items()}
        overrides.setdefault("llm_enabled", "true")
        overrides.setdefault("llm_visual_enabled", "true")

        try:
            settings = load_llm_settings(overrides)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("Failed to load LLM settings: %s", exc)
            return None

        if not settings.visual_enabled:
            print("LLM visual evaluation requested but disabled via settings.")
            return None

        if create_binary_content_fn is None or assess_visual_diff_fn is None:
            try:
                default_assess, default_create, _ = _load_visual_llm_runtime()
            except LLMDependencyError as exc:
                print(str(exc))
                return None
            if assess_visual_diff_fn is None:
                assess_visual_diff_fn = default_assess
            if create_binary_content_fn is None:
                create_binary_content_fn = default_create

        attachments = []
        summary_lines = [
            f"Reference file: {reference_image}",
            f"Candidate file: {candidate_image}",
        ]

        extra_messages: List[str] = list(notes or [])

        try:
            for index, diff in enumerate(differences, 1):
                ref_page = diff.get("ref_page")
                page_number = getattr(ref_page, "page_number", None)
                score = diff.get("score")
                rects = diff.get("rectangles") or []
                summary_lines.append(
                    f"Difference {index}: page={page_number}, score={score}, rectangles={rects}"
                )
                for note in diff.get("notes") or []:
                    extra_messages.append(str(note))

                combined = diff.get("combined_diff")
                if combined is not None:
                    attachments.append(
                        create_binary_content_fn(
                            self._encode_image_to_png(combined), "image/png"
                        )
                    )
                absolute = diff.get("absolute_diff")
                if absolute is not None:
                    attachments.append(
                        create_binary_content_fn(
                            self._encode_image_to_png(absolute), "image/png"
                        )
                    )
        except LLMDependencyError as exc:
            print(str(exc))
            return None
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("Failed to prepare LLM payload: %s", exc)
            return None

        try:
            decision = assess_visual_diff_fn(
                settings=settings,
                textual_summary="\n".join(summary_lines),
                attachments=tuple(attachments),
                extra_messages=extra_messages,
                system_prompt=custom_prompt,
            )
        except LLMDependencyError as exc:
            print(str(exc))
            return None
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("LLM evaluation failed: %s", exc)
            return None
        return decision

    def _encode_image_to_png(self, image) -> bytes:
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Failed to encode image to PNG format.")
        return encoded.tobytes()

    def _check_movement_with_text(
        self,
        ref_page,
        cand_page,
        diff_rectangles,
        move_tolerance: int,
        detection_method: str,
        use_pdf_content: bool,
        force_ocr: bool,
    ) -> bool:
        if not diff_rectangles:
            print("No difference areas detected for movement tolerance check.")
            return True

        failed_areas = []
        prefer_pdf = use_pdf_content or (
            bool(ref_page.pdf_text_words) and bool(cand_page.pdf_text_words)
        )
        allow_ocr_fallback = detection_method == "text"
        epsilon = 0.5

        for index, rect in enumerate(diff_rectangles, 1):
            diff_area_reference = ref_page.get_area(rect)
            diff_area_candidate = cand_page.get_area(rect)

            self.add_screenshot_to_log(
                diff_area_reference,
                "_page_"
                + str(ref_page.page_number + 1)
                + "_diff_area_reference_"
                + str(index),
            )
            self.add_screenshot_to_log(
                diff_area_candidate,
                "_page_"
                + str(ref_page.page_number + 1)
                + "_diff_area_test_"
                + str(index),
            )

            ref_words, ref_source = self._collect_words_for_rect(
                ref_page,
                rect,
                prefer_pdf=prefer_pdf,
                force_ocr=force_ocr,
            )
            cand_words, cand_source = self._collect_words_for_rect(
                cand_page,
                rect,
                prefer_pdf=prefer_pdf,
                force_ocr=force_ocr,
            )

            if allow_ocr_fallback and (not ref_words or not cand_words):
                ref_words, ref_source = self._collect_words_for_rect(
                    ref_page,
                    rect,
                    prefer_pdf=False,
                    force_ocr=True,
                )
                cand_words, cand_source = self._collect_words_for_rect(
                    cand_page,
                    rect,
                    prefer_pdf=False,
                    force_ocr=True,
                )

            if not ref_words or not cand_words:
                failed_areas.append((rect, "no textual elements found"))
                self.add_screenshot_to_log(
                    self.blend_two_images(diff_area_reference, diff_area_candidate),
                    suffix="_moved_area",
                    original_size=False,
                )
                continue

            measurement = self._measure_text_movement(ref_words, cand_words)
            if measurement["status"] != "ok":
                failed_areas.append((rect, measurement["message"]))
                self.add_screenshot_to_log(
                    self.blend_two_images(diff_area_reference, diff_area_candidate),
                    suffix="_moved_area",
                    original_size=False,
                )
                continue

            distance = measurement["distance"]
            self.add_screenshot_to_log(
                self.blend_two_images(diff_area_reference, diff_area_candidate),
                "_diff_area_blended",
            )

            if distance > move_tolerance + epsilon:
                failed_areas.append((rect, round(distance, 3)))
                print(
                    f"Area {rect} is moved {distance:.2f} pixels which is more than the tolerated {move_tolerance} pixels."
                )
            else:
                source = ref_source or cand_source or "text"
                print(
                    f"Area {rect} movement (max {distance:.2f}px) detected via {source}; within tolerance ({move_tolerance}px)."
                )

        if failed_areas:
            print(
                f"Movement tolerance check failed for {len(failed_areas)} area(s) out of {len(diff_rectangles)} total areas."
            )
            for rect, reason in failed_areas:
                print(f"  - Area {rect}: {reason}")
            raise AssertionError("The compared images are different.")

        print(
            f"All {len(diff_rectangles)} moved area(s) are within the {move_tolerance} pixel tolerance."
        )
        return True

    def _collect_words_for_rect(
        self,
        page,
        rect,
        *,
        prefer_pdf: bool,
        force_ocr: bool,
    ):
        import fitz

        rect_bounds = (
            rect["x"],
            rect["y"],
            rect["x"] + rect["width"],
            rect["y"] + rect["height"],
        )

        if prefer_pdf and page.pdf_text_words:
            rect_pts = fitz.Rect(
                rect_bounds[0] * 72 / page.dpi,
                rect_bounds[1] * 72 / page.dpi,
                rect_bounds[2] * 72 / page.dpi,
                rect_bounds[3] * 72 / page.dpi,
            )
            words = []
            for word in page.pdf_text_words:
                if fitz.Rect(word[:4]).intersects(rect_pts):
                    bbox = (
                        word[0] * page.dpi / 72,
                        word[1] * page.dpi / 72,
                        word[2] * page.dpi / 72,
                        word[3] * page.dpi / 72,
                    )
                    words.append(
                        {
                            "text": word[4].strip(),
                            "bbox": bbox,
                            "sort": (word[5], word[6], word[7], bbox[1], bbox[0]),
                        }
                    )
            if words:
                words.sort(key=lambda w: w["sort"])
                return words, "pdf"

        if force_ocr or not prefer_pdf:
            if not page.ocr_performed:
                page.apply_ocr(self.ocr_engine)
            data = page.ocr_text_data or {}
            texts = data.get("text", [])
            lefts = data.get("left", [])
            tops = data.get("top", [])
            widths = data.get("width", [])
            heights = data.get("height", [])
            confs = data.get("conf", [])
            blocks = data.get("block_num", [])
            lines = data.get("line_num", [])
            words_idx = data.get("word_num", [])

            words = []
            for idx, text in enumerate(texts):
                if not text or not text.strip():
                    continue
                try:
                    conf_val = float(confs[idx]) if confs else DEFAULT_CONFIDENCE
                except (TypeError, ValueError):
                    conf_val = DEFAULT_CONFIDENCE
                if conf_val < DEFAULT_CONFIDENCE:
                    continue

                x = lefts[idx]
                y = tops[idx]
                w = widths[idx]
                h = heights[idx]
                bbox = (x, y, x + w, y + h)
                if not self._rectangles_intersect(rect_bounds, bbox):
                    continue

                block = blocks[idx] if blocks else 0
                line = lines[idx] if lines else 0
                word_number = words_idx[idx] if words_idx else idx

                words.append(
                    {
                        "text": text.strip(),
                        "bbox": bbox,
                        "sort": (block, line, word_number, bbox[1], bbox[0]),
                    }
                )
            if words:
                words.sort(key=lambda w: w["sort"])
                return words, "ocr"

        return [], None

    def _measure_text_movement(self, reference_words, candidate_words):
        if len(reference_words) != len(candidate_words):
            return {
                "status": "mismatch",
                "message": "different number of text elements",
            }

        max_distance = 0.0
        for ref_word, cand_word in zip(reference_words, candidate_words):
            if ref_word["text"] != cand_word["text"]:
                return {
                    "status": "mismatch",
                    "message": f"text differs ('{ref_word['text']}' vs '{cand_word['text']}')",
                }

            ref_bbox = ref_word["bbox"]
            cand_bbox = cand_word["bbox"]
            deltas = [
                abs(ref_bbox[0] - cand_bbox[0]),
                abs(ref_bbox[1] - cand_bbox[1]),
                abs(ref_bbox[2] - cand_bbox[2]),
                abs(ref_bbox[3] - cand_bbox[3]),
            ]
            max_distance = max(max_distance, max(deltas))

        return {"status": "ok", "distance": max_distance}

    @staticmethod
    def _rectangles_intersect(rect_a, rect_b):
        return not (
            rect_a[2] <= rect_b[0]
            or rect_b[2] <= rect_a[0]
            or rect_a[3] <= rect_b[1]
            or rect_b[3] <= rect_a[1]
        )

    def _check_movement_with_images(
        self,
        ref_page,
        cand_page,
        diff_rectangles,
        move_tolerance: int,
        detection_method: str,
    ) -> bool:
        if not diff_rectangles:
            print("No difference areas detected for movement tolerance check.")
            return True

        failed_areas = []
        fallback_detections = 0
        epsilon = 0.75

        for rect in diff_rectangles:
            expanded_rect = self._expand_rect_for_detection(rect, move_tolerance, ref_page)
            reference_area = ref_page.get_area(expanded_rect)
            candidate_area = cand_page.get_area(expanded_rect)

            try:
                result = self.find_partial_image_position(
                    reference_area,
                    candidate_area,
                    threshold=self.partial_image_threshold,
                    detection=detection_method,
                )

                if result:
                    distance = float(result.get("distance", 0.0))
                    if "pt1" in result and "pt2" in result:
                        pt1 = np.array(result["pt1"])
                        pt2 = np.array(result["pt2"])
                        distance = float(np.linalg.norm(pt1 - pt2))

                    is_likely_fallback = (
                        "method" in result
                        and "fallback" in result["method"]
                        and distance <= 5
                    )

                    if is_likely_fallback:
                        fallback_detections += 1
                        print(
                            f"Area {rect}: Detected likely content removal (fallback distance: {distance:.2f}px, method: {result.get('method', 'unknown')})"
                        )

                    if distance > move_tolerance + epsilon:
                        failed_areas.append((rect, round(distance, 3)))
                        print(
                            f"Area {rect} is moved {distance:.2f} pixels which is more than the tolerated {move_tolerance} pixels."
                        )
                        self.add_screenshot_to_log(
                            self.blend_two_images(reference_area, candidate_area),
                            suffix="_moved_area",
                            original_size=False,
                        )
                    else:
                        print(f"Area {rect} is moved {distance:.2f} pixels.")
                else:
                    failed_areas.append((rect, "detection_failed"))
                    print(
                        f"Area {rect} movement detection failed - assuming movement exceeds tolerance."
                    )
                    self.add_screenshot_to_log(
                        self.blend_two_images(reference_area, candidate_area),
                        suffix="_moved_area",
                        original_size=False,
                    )
            except Exception as exc:
                print(f"Could not compare areas: {exc}")
                failed_areas.append((rect, f"error: {str(exc)}"))
                self.add_screenshot_to_log(
                    self.blend_two_images(reference_area, candidate_area),
                    suffix="_moved_area",
                    original_size=False,
                )

        if failed_areas:
            print(
                f"Movement tolerance check failed for {len(failed_areas)} area(s) out of {len(diff_rectangles)} total areas."
            )
            for rect, reason in failed_areas:
                print(f"  - Area {rect}: {reason}")
            raise AssertionError("The compared images are different.")

        if fallback_detections:
            print(
                f"{fallback_detections} out of {len(diff_rectangles)} area(s) required fallback movement detection."
            )

        print(
            f"All {len(diff_rectangles)} moved area(s) are within the {move_tolerance} pixel tolerance."
        )
        return True

    def _expand_rect_for_detection(self, rect, move_tolerance, page):
        pad = max(5, int(move_tolerance) + 2)
        x = max(0, rect["x"] - pad)
        y = max(0, rect["y"] - pad)
        width = rect["width"] + 2 * pad
        height = rect["height"] + 2 * pad
        max_width = page.image.shape[1] - x
        max_height = page.image.shape[0] - y
        width = min(width, max_width)
        height = min(height, max_height)
        return {"x": x, "y": y, "width": width, "height": height}

    @keyword
    def set_movement_detection(self, movement_detection: str):
        """Set the default movement detection method for subsequent comparisons.

        | =Arguments= | =Description= |
        | ``movement_detection`` | One of ``template``, ``orb``, ``sift``, ``text``. |

        Examples:
        | `Set Movement Detection`    text    # Switch to text-based movement detection |
        | `Set Movement Detection`    orb     # Revert to ORB feature matching |

        """
        method = movement_detection.lower()
        method = self.MOVEMENT_DETECTION_ALIASES.get(method, method)
        if method not in self.MOVEMENT_DETECTION_METHODS:
            raise ValueError(
                f"Unsupported movement detection method '{movement_detection}'. "
                f"Supported values are: {', '.join(sorted(self.MOVEMENT_DETECTION_METHODS))}."
            )
        self.movement_detection = method
        print(f"Movement detection method set to '{method}'.")

    @keyword
    def get_text_from_area(
        self,
        document: str,
        area: Union[str, dict, list] = None,
        assertion_operator: Optional[AssertionOperator] = None,
        assertion_expected: Any = None,
        message: str = None,
    ):
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

        Note: Character replacements can be configured via library initialization or
        ``Set Character Replacements`` keyword to normalize special characters.
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

        # Apply character replacements if configured
        if self.character_replacements and text:
            if isinstance(text, list):
                text = [apply_character_replacements(t, self.character_replacements) if t else t for t in text]
            else:
                text = apply_character_replacements(text, self.character_replacements)

        return text

    @keyword
    def get_text_from_document(
        self,
        document: str,
        assertion_operator: Optional[AssertionOperator] = None,
        assertion_expected: Any = None,
        message: str = None,
    ):
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
    def get_text(
        self,
        document: str,
        assertion_operator: Optional[AssertionOperator] = None,
        assertion_expected: Any = None,
        message: str = None,
    ):
        """Get the text content of a document.

        | =Arguments= | =Description= |
        | ``document`` | Path or URL of the document |
        | ``assertion_operator`` | Assertion operator to be used. |
        | ``assertion_expected`` | Expected value for the assertion. |
        | ``message`` | Message to be displayed in the log. |

        Examples:
        | `Get Text`    document.pdf    ==    "Expected text"    # Get the text content of the document
        | `Get Text`    document.pdf    !=    "Unexpected text"    # Get the text content of the document and check if it's not equal to the expected text
        | ${text}    `Get Text`    document.pdf    # Get the text content of the document and store it in a variable
        | `Should Be Equal`    ${text}    "Expected text"    # Check if the text content is equal to the expected text

        Note: Character replacements can be configured via library initialization or
        ``Set Character Replacements`` keyword to normalize special characters like
        non-breaking spaces before comparison.
        """
        if is_url(document):
            document = download_file_from_url(document)

        # Load the document
        doc = DocumentRepresentation(document, dpi=self.dpi, ocr_engine=self.ocr_engine)

        # Get the text content of the document
        text = doc.get_text(force_ocr=self.force_ocr)

        # Apply character replacements if configured
        if self.character_replacements:
            text = apply_character_replacements(text, self.character_replacements)

        return verify_assertion(text, assertion_operator, assertion_expected, message)

    @keyword
    def set_ocr_engine(self, ocr_engine: Literal["tesseract", "east"]):
        """Set the OCR engine to be used for text extraction.

        | =Arguments= | =Description= |
        | ``ocr_engine`` | OCR engine to be used. Options are ``tesseract`` and ``east``. |

        Examples:
        | `Set OCR Engine`    tesseract    # Set the OCR engine to Tesseract
        | `Set OCR Engine`    east    # Set the OCR engine to EAST

        """
        self.ocr_engine = ocr_engine

    @keyword
    def set_dpi(self, dpi: int):
        """Set the DPI to be used for image processing

        | =Arguments= | =Description= |
        | ``dpi`` | DPI to be used for image processing. |

        Examples:
        | `Set DPI`    300    # Set the DPI to 300

        """
        self.dpi = dpi

    @keyword
    def set_threshold(self, threshold: float):
        """Set the threshold for image comparison.

        | =Arguments= | =Description= |
        | ``threshold`` | Threshold for image comparison. |

        Examples:
        | `Set Threshold`    0.1    # Set the threshold to 0.1

        """
        self.threshold = threshold

    @keyword
    def set_screenshot_format(self, screenshot_format: str):
        """Set the format of the screenshots to be saved.

        | =Arguments= | =Description= |
        | ``screenshot_format`` | Format of the screenshots to be saved. Options are ``jpg`` and ``png``. |

        Examples:
        | `Set Screenshot Format`    jpg    # Set the screenshot format to jpg
        | `Set Screenshot Format`    png    # Set the screenshot format to png
        """
        self.screenshot_format = screenshot_format

    @keyword
    def set_embed_screenshots(self, embed_screenshots: bool):
        """Set whether to embed screenshots as base64 in the log.

        | =Arguments= | =Description= |
        | ``embed_screenshots`` | Whether to embed screenshots in the log. |

        Examples:
        | `Set Embed Screenshots`    True    # Set to embed screenshots as base64  in the log
        | `Set Embed Screenshots`    False    # Set not to embed screenshots in the log
        """
        self.embed_screenshots = embed_screenshots

    @keyword
    def set_take_screenshots(self, take_screenshots: bool):
        """Set whether to take screenshots during image comparison.

        | =Arguments= | =Description= |
        | ``take_screenshots`` | Whether to take screenshots during image comparison. |

        Examples:
        | `Set Take Screenshots`    True    # Set to take screenshots during image comparison
        | `Set Take Screenshots`    False    # Set not to take screenshots during image comparison

        """
        self.take_screenshots = take_screenshots

    @keyword
    def set_show_diff(self, show_diff: bool):
        """Set whether to show diff screenshot in the images during comparison.

        | =Arguments= | =Description= |
        | ``show_diff`` | Whether to show diff screenshot in the images during comparison. |

        Examples:
        | `Set Show Diff`    True    # Set to show diff screenshot in the images during comparison
        | `Set Show Diff`    False    # Set not to show diff screenshot in the images during comparison

        """
        self.show_diff = show_diff

    @keyword
    def set_screenshot_dir(self, screenshot_dir: str):
        """Set the directory to save screenshots.

        | =Arguments= | =Description= |
        | ``screenshot_dir`` | Directory to save screenshots. |

        Examples:
        | `Set Screenshot Dir`    screenshots    # Set the directory to save screenshots as 'screenshots'

        """
        self.screenshot_dir = Path(screenshot_dir)

    @keyword
    def set_reference_run(self, reference_run: bool):
        """Set whether the run is a reference run.
        In a Reference Run, the candidate images are saved as reference images and no comparison is done.

        | =Arguments= | =Description= |
        | ``reference_run`` | Whether the run is a reference run. |

        Examples:
        | `Set Reference Run`    True    # Set the run as a reference run
        | `Compare Images`    reference.pdf    candidate.pdf    # Saves `candidate.pdf` as `reference.pdf` and passes the test

        """
        self.reference_run = reference_run

    @keyword
    def set_force_ocr(self, force_ocr: bool):
        """Set whether to force OCR during image comparison.

        | =Arguments= | =Description= |
        | ``force_ocr`` | Whether to force OCR during image comparison. |

        Examples:
        | `Set Force OCR`    True    # Set to force OCR during image comparison
        | `Get Text`    document.pdf    # Get the text content of the document using OCR
        | `Set Force OCR`    False    # Set not to force OCR during image comparison
        | `Get Text`    document.pdf    # Get the text content of the document without using OCR

        """
        self.force_ocr = force_ocr

    @keyword
    def set_character_replacements(
        self, character_replacements: Optional[Union[Dict[str, str], str]] = None
    ):
        """Set character replacements to be applied during text extraction.

        Replacements are applied before returning text from keywords like
        ``Get Text``, ``Get Text From Document``, and ``Get Text From Area``.

        | =Arguments= | =Description= |
        | ``character_replacements`` | Dict mapping source chars to replacements, JSON string, or ``${NONE}`` to clear. |

        Examples:
        | `Set Character Replacements`    {'\u00A0': ' '}    # Replace non-breaking space with regular space
        | `Set Character Replacements`    {'\\u00A0': ' ', '\\u2013': '-'}    # Multiple replacements
        | ${text}=    `Get Text`    document.pdf    # Extracted text will have replacements applied
        | `Set Character Replacements`    ${NONE}    # Clear/disable replacements

        """
        self.character_replacements = self._parse_character_replacements(character_replacements)

    @keyword
    def get_barcodes(
        self,
        document: str,
        assertion_operator: Optional[AssertionOperator] = None,
        assertion_expected: Any = None,
        message: str = None,
    ):
        """Get the barcodes from a document. Returns the barcodes as a list of dictionaries.

        | =Arguments= | =Description= |
        | ``document`` | Path or URL of the document |
        | ``assertion_operator`` | Assertion operator to be used. |
        | ``assertion_expected`` | Expected value for the assertion. |
        | ``message`` | Message to be displayed in the log. |

        Optionally assert the barcodes using the ``assertion_operator`` and ``assertion_expected`` arguments.

        Examples:
        | ${data}    `Get Barcodes`    document.pdf    # Get the barcodes from the document
        | `Length Should Be`    ${data}    2    # Check if the number of barcodes is 2
        | `Should Be True`    ${data[0]} == {'x': 5, 'y': 7, 'height': 95, 'width': 96, 'value': 'Stegosaurus'}    # Check if the first barcode is as expected
        | `Should Be True`    ${data[1]} == {'x': 298, 'y': 7, 'height': 95, 'width': 95, 'value': 'Plesiosaurus'}    # Check if the second barcode is as expected
        | `Should Be True`    ${data[0]['value']} == 'Stegosaurus'    # Check if the value of the first barcode is 'Stegosaurus'
        | `Should Be True`    ${data[1]['value']} == 'Plesiosaurus'    # Check if the value of the second barcode is 'Plesiosaurus'
        | `Get Barcodes`    document.pdf    contains    'Stegosaurus'    # Check if the document contains a barcode with the value 'Stegosaurus'
        """
        if is_url(document):
            document = download_file_from_url(document)

        # Load the document
        doc = DocumentRepresentation(document, dpi=self.dpi, contains_barcodes=True)

        # Get the barcodes from the document
        barcodes = doc.get_barcodes()
        if assertion_operator:
            barcodes = [barcode["value"] for barcode in barcodes]
        return verify_assertion(
            barcodes, assertion_operator, assertion_expected, message
        )

    @keyword
    def get_barcodes_from_document(
        self,
        document: str,
        assertion_operator: Optional[AssertionOperator] = None,
        assertion_expected: Any = None,
        message: str = None,
    ):
        """Get the barcodes from a document. Returns the barcodes as a list of dictionaries.

        | =Arguments= | =Description= |
        | ``document`` | Path or URL of the document |
        | ``assertion_operator`` | Assertion operator to be used. |
        | ``assertion_expected`` | Expected value for the assertion. |
        | ``message`` | Message to be displayed in the log. |

        Optionally assert the barcodes using the ``assertion_operator`` and ``assertion_expected`` arguments.

        Examples:
        | ${data}    `Get Barcodes From Document`    document.pdf    # Get the barcodes from the document
        | `Length Should Be`    ${data}    2    # Check if the number of barcodes is 2
        | `Should Be True`    ${data[0]} == {'x': 5, 'y': 7, 'height': 95, 'width': 96, 'value': 'Stegosaurus'}    # Check if the first barcode is as expected
        | `Should Be True`    ${data[1]} == {'x': 298, 'y': 7, 'height': 95, 'width': 95, 'value': 'Plesiosaurus'}    # Check if the second barcode is as expected
        | `Should Be True`    ${data[0]['value']} == 'Stegosaurus'    # Check if the value of the first barcode is 'Stegosaurus'
        | `Should Be True`    ${data[1]['value']} == 'Plesiosaurus'    # Check if the value of the second barcode is 'Plesiosaurus'
        | `Get Barcodes From Document`    document.pdf    contains    'Stegosaurus'    # Check if the document contains a barcode with the value 'Stegosaurus'

        """
        return self.get_barcodes(
            document, assertion_operator, assertion_expected, message
        )

    @keyword
    def image_should_contain_template(
        self,
        image: str,
        template: str,
        threshold: float = 0.0,
        take_screenshots: bool = False,
        log_template: bool = False,
        detection: str = "template",
        tpl_crop_x1: int = None,
        tpl_crop_y1: int = None,
        tpl_crop_x2: int = None,
        tpl_crop_y2: int = None,
    ):
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
        | `Image Should Contain Template`    reference.jpg    template.jpg    #Checks if template is in image
        | `Image Should Contain Template`    reference.jpg    template.jpg    threshold=0.9    #Checks if template is in image with a higher threshold
        | `Image Should Contain Template`    reference.jpg    template.jpg    take_screenshots=True    #Checks if template is in image and adds screenshots to log
        | `Image Should Contain Template`    reference.jpg    template.jpg    tpl_crop_x1=50  tpl_crop_y1=50  tpl_crop_x2=100  tpl_crop_y2=100    #Before image comparison, the template image gets cropped to that selection.
        | `${coordinates}`    `Image Should Contain Template`    reference.jpg    template.jpg    #Checks if template is in image and returns coordinates of template
        | `Should Be Equal As Numbers`    ${coordinates['pt1'][0]}    100    #Checks if x coordinate of found template is 100
        """
        # Validate crop arguments and load images
        all_crop_args = all((tpl_crop_x1, tpl_crop_y1, tpl_crop_x2, tpl_crop_y2))
        any_crop_args = any((tpl_crop_x1, tpl_crop_y1, tpl_crop_x2, tpl_crop_y2))
        if not all_crop_args and any_crop_args:
            raise ValueError("Either provide all crop arguments or none of them.")

        img = DocumentRepresentation(image).pages[0].image
        original_template = DocumentRepresentation(template).pages[0].image

        # Log original dimensions for debugging
        print(f"Original template dimensions: {original_template.shape}")
        print(f"Original image dimensions: {img.shape}")

        # Crop the template image if crop boundaries are provided
        if all_crop_args:
            # Validate crop coordinates
            tpl_height, tpl_width = original_template.shape[:2]
            if (
                tpl_crop_x1 < 0
                or tpl_crop_y1 < 0
                or tpl_crop_x2 > tpl_width
                or tpl_crop_y2 > tpl_height
                or tpl_crop_x1 >= tpl_crop_x2
                or tpl_crop_y1 >= tpl_crop_y2
            ):
                raise ValueError(
                    f"Invalid crop coordinates. Template size: {tpl_width}x{tpl_height}, "
                    f"Crop: ({tpl_crop_x1},{tpl_crop_y1}) to ({tpl_crop_x2},{tpl_crop_y2})"
                )

            template = original_template[
                tpl_crop_y1:tpl_crop_y2, tpl_crop_x1:tpl_crop_x2
            ].copy()
            print(f"Cropped template dimensions: {template.shape}")
        else:
            template = original_template

        # Validate template size
        if template.size == 0:
            raise ValueError("Template image is empty after cropping")

        if template.shape[0] < 3 or template.shape[1] < 3:
            raise ValueError(f"Template too small after cropping: {template.shape}")

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        h, w = template.shape[0:2]

        # Validate that template is smaller than image
        img_h, img_w = img_gray.shape
        if h > img_h or w > img_w:
            raise ValueError(
                f"Template ({w}x{h}) is larger than image ({img_w}x{img_h})"
            )

        if detection == "template":
            print(f"Using template matching with threshold: {threshold}")

            # Perform template matching
            res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Log matching results for debugging
            print(
                f"Template matching results - min_val: {min_val:.6f}, max_val: {max_val:.6f}"
            )
            print(f"Best match location: {min_loc}")

            top_left = min_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # For TM_SQDIFF_NORMED, lower values indicate better matches
            # A threshold of 0.0 means exact match, higher values allow more differences
            match = min_val <= threshold

            if match:
                print(
                    f"Template found at location: {top_left} with confidence: {1.0 - min_val:.6f}"
                )
                # Draw rectangle on a copy to avoid modifying original
                img_result = img.copy()
                cv2.rectangle(img_result, top_left, bottom_right, (0, 255, 0), 2)

                if take_screenshots:
                    self.add_screenshot_to_log(img_result, "image_with_template")
            else:
                print(
                    f"Template not found. Best match confidence: {1.0 - min_val:.6f}, required: {1.0 - threshold:.6f}"
                )
                if take_screenshots:
                    # Show best match attempt for debugging
                    img_debug = img.copy()
                    cv2.rectangle(img_debug, top_left, bottom_right, (0, 0, 255), 2)
                    self.add_screenshot_to_log(img_debug, "image_with_failed_match")

            if log_template:
                self.add_screenshot_to_log(
                    template, "template_used", original_size=True
                )

            if match:
                return {
                    "pt1": top_left,
                    "pt2": bottom_right,
                    "confidence": 1.0 - min_val,
                }
            else:
                raise AssertionError(
                    f"The Template was not found in the Image. Best match confidence: {1.0 - min_val:.6f}, threshold: {1.0 - threshold:.6f}"
                )

        elif detection == "sift" or detection == "orb":
            img_kp, img_des, template_kp, template_des = (
                self.get_sift_keypoints_and_descriptors(img_gray, template_gray)
            )

            if img_kp is None or template_kp is None:
                raise AssertionError("The Template was not found in the Image.")

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
                img_matches = cv2.drawMatches(
                    template,
                    template_kp,
                    img,
                    img_kp,
                    matches_for_drawing,
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                self.add_screenshot_to_log(img_matches, "good_sift_matches")
                src_pts = np.float32(
                    [template_kp[m.queryIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [img_kp[m.trainIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)

                # Use homography to find the template in the larger image
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
                if M is not None:
                    # Define corners of the template and transform them into the full image
                    template_corners = np.float32(
                        [[0, 0], [w, 0], [w, h], [0, h]]
                    ).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(template_corners, M)

                    # Draw bounding box on the full image
                    cv2.polylines(
                        img,
                        [np.int32(transformed_corners)],
                        True,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA,
                    )

                    # Add a screenshot with the detected area to the log
                    self.add_screenshot_to_log(img, "image_with_template")

                    # Return the coordinates of the detected area
                    top_left = (
                        int(transformed_corners[0][0][0]),
                        int(transformed_corners[0][0][1]),
                    )
                    bottom_right = (
                        int(transformed_corners[2][0][0]),
                        int(transformed_corners[2][0][1]),
                    )

                    return {"pt1": top_left, "pt2": bottom_right}
                else:
                    raise AssertionError("The Template was not found in the Image.")
            else:
                raise AssertionError("The Template was not found in the Image.")

        else:
            raise ValueError("Detection method must be 'template', 'orb' or 'sift'.")

    def _get_diff_rectangles(self, absolute_diff):
        """Get rectangles around differences in the page."""
        # Increase the size of the conturs to make sure the differences are covered and small differences are grouped
        # Every contour is a rectangle, overlapping rectangles are grouped together

        absolute_diff = cv2.dilate(absolute_diff, None, iterations=10)
        contours, _ = cv2.findContours(
            absolute_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rectangles = [cv2.boundingRect(contour) for contour in contours]
        return rectangles

    def _raise_comparison_failure(
        self, message: str = "The compared images are different."
    ):
        """Handle failures in image comparison."""
        raise AssertionError(message)

    def _compare_text_content(
        self,
        reference_doc: DocumentRepresentation,
        candidate_doc: DocumentRepresentation,
    ):
        """Compare the text content of the two documents after OCR."""
        reference_text = reference_doc.get_text_content()
        candidate_text = candidate_doc.get_text_content()
        if reference_text != candidate_text:
            raise AssertionError(
                f"Text content differs:\n\nReference Text:\n{reference_text}\n\nCandidate Text:\n{candidate_text}"
            )

    def _load_placeholders(self, placeholder_file: Union[str, dict, list]):
        """Load and return placeholders from file or dictionary."""
        if isinstance(placeholder_file, str) and os.path.exists(placeholder_file):
            with open(placeholder_file, "r") as f:
                return json.load(f)
        return placeholder_file  # If it's already in dict/list format

    def get_images_with_highlighted_differences(
        self, thresh, reference, candidate, extension=10
    ):
        # thresh = cv2.dilate(thresh, None, iterations=extension)
        thresh = cv2.dilate(thresh, None, iterations=extension)
        thresh = cv2.erode(thresh, None, iterations=extension)
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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

        # Use moderate morphological operations for balanced detection
        # iterations=5: Compromise between preserving separate objects and grouping characters
        # - Enough to group characters within words/text strings
        # - Not so much as to merge completely separate moved objects
        absolute_diff = cv2.dilate(absolute_diff, None, iterations=5)
        absolute_diff = cv2.erode(absolute_diff, None, iterations=5)
        contours, _ = cv2.findContours(
            absolute_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter out very small contours that are likely noise
        min_contour_area = 100  # Minimum area to consider as a meaningful difference
        filtered_contours = [
            c for c in contours if cv2.contourArea(c) >= min_contour_area
        ]

        rectangles = [cv2.boundingRect(contour) for contour in filtered_contours]
        # Convert the rectangles to a list of dictionaries
        rectangles = [
            {"x": rect[0], "y": rect[1], "width": rect[2], "height": rect[3]}
            for rect in rectangles
        ]
        return rectangles

    def add_screenshot_to_log(self, image, suffix, original_size=False):
        if original_size:
            img_style = "width: auto; height: auto;"
        else:
            img_style = "width:50%; height: auto;"

        if self.embed_screenshots:
            import base64

            if self.screenshot_format == "jpg":
                _, encoded_img = cv2.imencode(
                    ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                )  # im_arr: image in Numpy one-dim array format.
                im_b64 = base64.b64encode(encoded_img).decode()
                print(
                    "*HTML* "
                    + f'{suffix}:<br><img alt="screenshot" src="data:image/jpeg;base64,{im_b64}" style="{img_style}">'
                )
            else:
                _, encoded_img = cv2.imencode(".png", image)
                im_b64 = base64.b64encode(encoded_img).decode()
                print(
                    "*HTML* "
                    + f'{suffix}:<br><img alt="screenshot" src="data:image/png;base64,{im_b64}" style="{img_style}">'
                )
        else:
            screenshot_name = str(
                str(uuid.uuid1()) + suffix + ".{}".format(self.screenshot_format)
            )
            if self.PABOTQUEUEINDEX is not None:
                rel_screenshot_path = str(
                    self.screenshot_dir
                    / "{}-{}".format(self.PABOTQUEUEINDEX, screenshot_name)
                )
            else:
                rel_screenshot_path = str(self.screenshot_dir / screenshot_name)
            abs_screenshot_path = str(
                self.output_directory / self.screenshot_dir / screenshot_name
            )
            os.makedirs(os.path.dirname(abs_screenshot_path), exist_ok=True)
            if self.screenshot_format == "jpg":
                cv2.imwrite(
                    abs_screenshot_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                )
            else:
                cv2.imwrite(abs_screenshot_path, image)
            print(
                "*HTML* "
                + f'{suffix}:<br><a href="{rel_screenshot_path}" target="_blank"><img src="{rel_screenshot_path}" style="{img_style}"></a>'
            )

    def find_partial_image_position(
        self, img, template, threshold=0.1, detection="classic"
    ):
        """
        Find the position and movement distance of a template image within a larger image.

        Args:
            img: The source image (numpy array)
            template: The template image to find (numpy array)
            threshold: Threshold for matching sensitivity (float)
            detection: Detection method ('template', 'classic', 'orb', 'sift')

        Returns:
            Dictionary with movement information or None if not found
        """
        # Enhanced input validation
        if img is None or template is None:
            self._log_verbose_warning("Input images are None")
            return None

        if not isinstance(img, np.ndarray) or not isinstance(template, np.ndarray):
            self._log_verbose_warning("Input images must be numpy arrays")
            return None

        if img.size == 0 or template.size == 0:
            self._log_verbose_warning("Input images are empty")
            return None

        # Check for minimum image dimensions
        if img.shape[0] < 5 or img.shape[1] < 5:
            self._log_verbose_warning("Source image too small for movement detection")
            return None

        if template.shape[0] < 3 or template.shape[1] < 3:
            self._log_verbose_warning("Template image too small for movement detection")
            return None

        # Ensure images have the same number of channels
        if len(img.shape) != len(template.shape):
            self._log_verbose_warning("Images have different number of channels")
            return None

        # Check if template is larger than source image
        if template.shape[0] > img.shape[0] or template.shape[1] > img.shape[1]:
            self._log_verbose_warning("Template image is larger than source image")
            return None

        # Validate threshold parameter
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            self._log_verbose_warning("Invalid threshold value, using default")
            threshold = self.partial_image_threshold

        # Fallback chain for robust detection
        fallback_methods = []
        if detection == "template" or detection == "classic":
            fallback_methods = ["template", "sift", "orb"]
        elif detection == "sift":
            fallback_methods = ["sift", "template", "orb"]
        elif detection == "orb":
            fallback_methods = ["orb", "template", "sift"]
        else:
            self._log_verbose_warning(
                f"Unknown detection method: {detection}, using template"
            )
            fallback_methods = ["template", "sift", "orb"]

        # Try detection methods with fallback
        for method in fallback_methods:
            try:
                result = None
                if method == "template":
                    result = self.find_partial_image_distance_with_matchtemplate(
                        img, template, threshold
                    )
                elif method == "orb":
                    result = self.find_partial_image_distance_with_orb(img, template)
                elif method == "sift":
                    result = self.find_partial_image_distance_with_sift(
                        img, template, threshold
                    )

                if result is not None:
                    if method != detection:
                        LOG.info(
                            f"Primary method '{detection}' failed, succeeded with '{method}'"
                        )
                    return result

            except Exception as e:
                self._log_verbose_warning(
                    f"Movement detection failed with {method} method: {str(e)}"
                )
                continue

        self._log_verbose_warning("All movement detection methods failed")
        return None

    def find_partial_image_distance_with_sift(self, img, template, threshold=0.1):
        """
        Find movement distance using SIFT feature matching.

        Args:
            img: Source image (numpy array)
            template: Template image (numpy array)
            threshold: Matching threshold (float)

        Returns:
            Dictionary with displacement and distance information or None
        """
        try:
            # Enhanced input validation
            if img is None or template is None:
                self._log_verbose_warning("SIFT: Input images are None")
                return None

            if not isinstance(img, np.ndarray) or not isinstance(template, np.ndarray):
                self._log_verbose_warning(
                    "SIFT: Input images must be numpy arrays"
                )
                return None

            if img.size == 0 or template.size == 0:
                self._log_verbose_warning("SIFT: Input images are empty")
                return None

            # Check image dimensions
            if len(img.shape) < 2 or len(template.shape) < 2:
                self._log_verbose_warning("SIFT: Invalid image dimensions")
                return None

            # Adaptive grayscale conversion
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img.copy()

            if len(template.shape) == 3:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template.copy()

            # Find non-white area bounding boxes with improved robustness
            def get_content_bbox(image):
                """Get bounding box of non-background content with adaptive thresholding"""
                # Try multiple threshold values for robustness
                for thresh_val in [254, 250, 240, 200]:
                    _, binary = cv2.threshold(
                        image, thresh_val, 255, cv2.THRESH_BINARY_INV
                    )
                    nonzero = cv2.findNonZero(binary)
                    if nonzero is not None and len(nonzero) > 10:  # Minimum points
                        return cv2.boundingRect(nonzero)
                return None

            template_bbox = get_content_bbox(template_gray)
            img_bbox = get_content_bbox(img_gray)

            if template_bbox is None or img_bbox is None:
                self._log_verbose_warning(
                    "SIFT: Could not find content areas in images"
                )
                return None

            # Initialize SIFT detector with adaptive parameters
            def create_sift_detector(contrast_threshold=0.04, edge_threshold=10):
                """Create SIFT detector with given parameters"""
                try:
                    return cv2.SIFT_create(
                        contrastThreshold=contrast_threshold,
                        edgeThreshold=edge_threshold,
                        nfeatures=1000,
                    )
                except Exception:
                    # Fallback to basic SIFT if parameterized version fails
                    return cv2.SIFT_create()

            # Try SIFT detection with different sensitivity settings
            sift_params = [
                (0.04, 10),  # Default parameters
                (0.02, 15),  # More sensitive
                (0.01, 20),  # Very sensitive
            ]

            keypoints_img = None
            descriptors_img = None
            keypoints_template = None
            descriptors_template = None

            for contrast_thresh, edge_thresh in sift_params:
                try:
                    sift = create_sift_detector(contrast_thresh, edge_thresh)

                    # Detect SIFT keypoints and descriptors
                    keypoints_img, descriptors_img = sift.detectAndCompute(
                        img_gray, None
                    )
                    keypoints_template, descriptors_template = sift.detectAndCompute(
                        template_gray, None
                    )

                    # Check if we have sufficient keypoints
                    if (
                        keypoints_img is not None
                        and keypoints_template is not None
                        and len(keypoints_img) >= self.sift_min_matches
                        and len(keypoints_template) >= self.sift_min_matches
                        and descriptors_img is not None
                        and descriptors_template is not None
                    ):
                        break

                except Exception as e:
                    self._log_verbose_warning(
                        f"SIFT: Failed with parameters ({contrast_thresh}, {edge_thresh}): {str(e)}"
                    )
                    continue

            # Final validation of keypoints and descriptors
            if (
                descriptors_img is None
                or descriptors_template is None
                or len(keypoints_img) < self.sift_min_matches
                or len(keypoints_template) < self.sift_min_matches
            ):
                self._log_verbose_warning("SIFT: Insufficient keypoints detected")
                return None

            # Robust feature matching with adaptive parameters
            def perform_matching():
                """Perform feature matching with error handling"""
                try:
                    # Use BFMatcher with L2 norm for SIFT
                    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                    matches = bf.knnMatch(descriptors_template, descriptors_img, k=2)

                    # Enhanced Lowe's ratio test with adaptive threshold
                    good_matches = []
                    ratio_thresholds = [
                        self.sift_ratio_threshold,
                        0.8,
                        0.9,
                    ]  # Try different ratios

                    for ratio_thresh in ratio_thresholds:
                        good_matches = []
                        for match_pair in matches:
                            if (
                                len(match_pair) >= 2
                            ):  # Ensure we have 2 matches for ratio test
                                m, n = match_pair[0], match_pair[1]
                                if m.distance < ratio_thresh * n.distance:
                                    good_matches.append(m)

                        if len(good_matches) >= self.sift_min_matches:
                            break

                    return (
                        good_matches
                        if len(good_matches) >= self.sift_min_matches
                        else []
                    )

                except Exception as e:
                    self._log_verbose_warning(
                        f"SIFT: Matching failed: {str(e)}"
                    )
                    return []

            good_matches = perform_matching()

            # Use adaptive minimum matches for small text areas
            effective_min_matches = min(
                self.sift_min_matches,
                len(good_matches) if len(good_matches) >= 2 else 0,
            )

            if len(good_matches) < effective_min_matches:
                self._log_verbose_warning(
                    f"SIFT: Insufficient good matches: {len(good_matches)} < {effective_min_matches}"
                )
                return None

            # Extract matched points with error handling
            try:
                src_pts = np.float32(
                    [keypoints_template[m.queryIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [keypoints_img[m.trainIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)
            except (IndexError, AttributeError) as e:
                self._log_verbose_warning(
                    f"SIFT: Error extracting match points: {str(e)}"
                )
                return None

            # Robust homography computation with outlier rejection
            try:
                # Try different RANSAC thresholds for robustness
                ransac_thresholds = [
                    self.ransac_threshold,
                    self.ransac_threshold * 2,
                    self.ransac_threshold / 2,
                ]

                M = None
                inlier_ratio = 0
                for ransac_thresh in ransac_thresholds:
                    try:
                        M, mask = cv2.findHomography(
                            src_pts,
                            dst_pts,
                            cv2.RANSAC,
                            ransac_thresh,
                            maxIters=2000,
                            confidence=0.995,
                        )

                        if M is not None:
                            # Validate homography matrix
                            if self._validate_homography_matrix(M):
                                # Count inliers
                                inlier_count = np.sum(mask) if mask is not None else 0
                                inlier_ratio = inlier_count / len(good_matches)

                                # Require reasonable inlier ratio
                                if (
                                    inlier_ratio >= 0.15
                                ):  # At least 15% inliers (reduced for text)
                                    break
                        M = None  # Reset if validation failed
                    except Exception as e:
                        self._log_verbose_warning(
                            f"SIFT: Homography computation failed with config {ransac_thresholds.index(ransac_thresh)}: {str(e)}"
                        )
                        continue

                if M is None:
                    self._log_verbose_warning("SIFT: Could not compute valid homography")
                    return None

                # Extract translation with bounds checking
                dx = float(M[0, 2])
                dy = float(M[1, 2])

                # Sanity check: movement should be reasonable relative to image size
                max_reasonable_movement = (
                    max(img_gray.shape) * 2
                )  # Allow up to 2x image dimension
                if (
                    abs(dx) > max_reasonable_movement
                    or abs(dy) > max_reasonable_movement
                ):
                    self._log_verbose_warning(
                        f"SIFT: Unreasonable movement detected: dx={dx}, dy={dy}"
                    )
                    return None

                # Calculate moved distance
                moved_distance = float(np.sqrt(dx**2 + dy**2))

                return {
                    "displacement_x": dx,
                    "displacement_y": dy,
                    "distance": moved_distance,
                    "inlier_ratio": inlier_ratio,
                    "num_matches": len(good_matches),
                    "method": "sift",
                }

            except Exception as e:
                self._log_verbose_warning(
                    f"SIFT: Homography computation failed: {str(e)}"
                )
                return None

        except Exception as e:
            LOG.error(f"SIFT: Unexpected error in movement detection: {str(e)}")
            return None

    def _validate_homography_matrix(self, H):
        """
        Validate homography matrix for reasonableness.

        Args:
            H: 3x3 homography matrix

        Returns:
            bool: True if matrix appears valid
        """
        try:
            if H is None or H.shape != (3, 3):
                return False

            # Check for NaN or infinite values
            if not np.all(np.isfinite(H)):
                return False

            # Check matrix determinant (should not be zero or too close to zero)
            det = np.linalg.det(H[:2, :2])  # Upper 2x2 submatrix
            if abs(det) < 1e-6:
                return False

            # Check for reasonable scale (should be close to 1 for movement detection)
            scale_x = np.sqrt(H[0, 0] ** 2 + H[0, 1] ** 2)
            scale_y = np.sqrt(H[1, 0] ** 2 + H[1, 1] ** 2)

            # Allow scale variation between 0.5 and 2.0
            if scale_x < 0.5 or scale_x > 2.0 or scale_y < 0.5 or scale_y > 2.0:
                return False

            # Check for reasonable shear (should be minimal for pure translation)
            shear = abs(H[0, 1] * H[1, 0]) / (scale_x * scale_y)
            if shear > 0.5:  # Allow some shear but not too much
                return False

            return True

        except Exception:
            return False

    def find_partial_image_distance_with_matchtemplate(
        self, img, template, threshold=0.1
    ):
        """
        Find movement distance using template matching with enhanced robustness.

        Args:
            img: Source image (numpy array)
            template: Template image (numpy array)
            threshold: Matching threshold (float)

        Returns:
            Dictionary with movement information or None
        """
        try:
            # Enhanced input validation
            if img is None or template is None:
                self._log_verbose_warning("Template matching: Input images are None")
                return None

            if not isinstance(img, np.ndarray) or not isinstance(template, np.ndarray):
                self._log_verbose_warning(
                    "Template matching: Input images must be numpy arrays"
                )
                return None

            if img.size == 0 or template.size == 0:
                self._log_verbose_warning("Template matching: Input images are empty")
                return None

            # Validate threshold
            if (
                not isinstance(threshold, (int, float))
                or threshold < 0
                or threshold > 1
            ):
                self._log_verbose_warning(
                    "Template matching: Invalid threshold, using default"
                )
                threshold = self.partial_image_threshold

            # Adaptive grayscale conversion
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img.copy()

            if len(template.shape) == 3:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template.copy()

            h, w = template_gray.shape

            # Enhanced size validation
            if h > img_gray.shape[0] or w > img_gray.shape[1]:
                self._log_verbose_warning(
                    "Template matching: Template larger than source image"
                )
                return None

            if h < 3 or w < 3:
                self._log_verbose_warning("Template matching: Template too small")
                return None

            # Enhanced difference calculation with noise reduction
            def calculate_robust_mask(img1, img2):
                """Calculate difference mask with noise reduction"""
                # Apply slight Gaussian blur to reduce noise
                img1_blur = cv2.GaussianBlur(img1, (3, 3), 0.5)
                img2_blur = cv2.GaussianBlur(img2, (3, 3), 0.5)

                # Calculate absolute difference
                diff = cv2.absdiff(img1_blur, img2_blur)

                # Use adaptive threshold to handle varying illumination
                threshold_value = max(
                    3, np.mean(diff) + 1 * np.std(diff)
                )  # More sensitive for text
                mask = np.where(diff > threshold_value, 255, 0).astype(np.uint8)

                # Morphological operations to clean up the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                return mask

            mask = calculate_robust_mask(img_gray, template_gray)

            # Enhanced contour finding with error handling
            try:
                cnts, hierarchy = cv2.findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
            except Exception as e:
                self._log_verbose_warning(
                    f"Template matching: Contour detection failed: {str(e)}"
                )
                return None

            if len(cnts) == 0:
                self._log_verbose_warning("Template matching: No contours found")
                return None

            # Enhanced contour processing
            def process_contours(contours, image_shape):
                """Process contours with area filtering and merging"""
                # Filter contours by area - more lenient for text detection
                min_area = 5  # Very small minimum contour area for text
                valid_contours = [
                    cnt for cnt in contours if cv2.contourArea(cnt) >= min_area
                ]

                if not valid_contours:
                    return None

                # Create merged contour mask
                merged_mask = np.zeros(image_shape, np.uint8)
                for cnt in valid_contours:
                    cv2.drawContours(merged_mask, [cnt], -1, 255, -1)

                # Additional morphological processing to connect nearby regions
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel)

                return merged_mask

            merged_contour = process_contours(cnts, mask.shape)
            if merged_contour is None:
                self._log_verbose_warning(
                    "Template matching: No valid contours after processing"
                )
                return None

            # Enhanced masked image creation with improved masking
            def create_masked_images(img, template, mask):
                """Create masked images with improved background handling"""
                # Invert the mask to work with non-difference areas
                inv_mask = cv2.bitwise_not(mask)

                # Create masked images using the inverted mask
                masked_img = cv2.bitwise_or(img, inv_mask)
                masked_template = cv2.bitwise_or(template, inv_mask)

                return masked_img, masked_template

            masked_img, masked_template = create_masked_images(
                img_gray, template_gray, merged_contour
            )

            # Enhanced template region extraction with adaptive processing
            def extract_template_roi(template, blur_kernel=(3, 3)):
                """Extract ROI from template with adaptive thresholding"""
                # Apply Gaussian blur for noise reduction
                template_blur = cv2.GaussianBlur(template, blur_kernel, 0)

                # Use Otsu's thresholding for adaptive binarization
                _, template_thresh = cv2.threshold(
                    template_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )

                # Find bounding rectangle of non-zero regions
                coords = cv2.findNonZero(template_thresh)
                if coords is None:
                    return None, None, None, None, None

                x, y, w, h = cv2.boundingRect(coords)

                # Add small padding if possible
                padding = 2
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(template.shape[1] - x, w + 2 * padding)
                h = min(template.shape[0] - y, h + 2 * padding)

                return x, y, w, h, template_thresh

            temp_x, temp_y, temp_w, temp_h, template_thresh = extract_template_roi(
                masked_template
            )

            if temp_x is None or temp_w < 5 or temp_h < 5:
                self._log_verbose_warning(
                    "Template matching: Template ROI too small or invalid"
                )
                # Return a small distance as fallback for very small movements
                return {"distance": 3, "method": "template_fallback"}

            # Boundary validation with safety margins
            if (
                temp_y + temp_h > masked_template.shape[0]
                or temp_x + temp_w > masked_template.shape[1]
                or temp_x < 0
                or temp_y < 0
            ):
                self._log_verbose_warning(
                    "Template matching: ROI bounds exceed image dimensions"
                )
                return None

            # Extract template ROI
            template_roi = masked_template[
                temp_y : temp_y + temp_h, temp_x : temp_x + temp_w
            ]

            # Enhanced template matching with multiple methods
            def perform_template_matching(image, template_roi):
                """Perform template matching with multiple methods for robustness"""
                methods = [
                    cv2.TM_SQDIFF_NORMED,
                    cv2.TM_CCORR_NORMED,
                    cv2.TM_CCOEFF_NORMED,
                ]

                results = []
                for method in methods:
                    try:
                        res = cv2.matchTemplate(image, template_roi, method)
                        if method == cv2.TM_SQDIFF_NORMED:
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                            results.append((min_val, min_loc, method))
                        else:
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                            results.append(
                                (1 - max_val, max_loc, method)
                            )  # Convert to distance-like metric
                    except Exception as e:
                        self._log_verbose_warning(
                            f"Template matching method {method} failed: {str(e)}"
                        )
                        continue

                return results

            # Perform matching on both images for comparison
            img_results = perform_template_matching(masked_img, template_roi)
            template_results = perform_template_matching(masked_template, template_roi)

            if not img_results or not template_results:
                self._log_verbose_warning(
                    "Template matching: All matching methods failed"
                )
                return None

            # Find best results (lowest distance-like metric)
            best_img_result = min(img_results, key=lambda x: x[0])
            best_template_result = min(template_results, key=lambda x: x[0])

            min_val, min_loc, method_used = best_img_result
            min_val_temp, min_loc_temp, method_temp = best_template_result

            # Enhanced threshold validation - very permissive for large movements
            effective_threshold = max(
                threshold, 0.5
            )  # Very high threshold to handle large movements
            if min_val > effective_threshold:
                self._log_verbose_warning(
                    f"Template matching: Match quality too low: {min_val} > {effective_threshold}"
                )
                return None

            # Calculate movement with error bounds checking
            try:
                dx = float(min_loc[0] - min_loc_temp[0])
                dy = float(min_loc[1] - min_loc_temp[1])

                # Sanity check for reasonable movement
                max_movement = max(img_gray.shape) * 1.5
                if abs(dx) > max_movement or abs(dy) > max_movement:
                    self._log_verbose_warning(
                        f"Template matching: Unreasonable movement: dx={dx}, dy={dy}"
                    )
                    return None

                distance = float(np.sqrt(dx**2 + dy**2))

                return {
                    "pt1": min_loc,
                    "pt2": min_loc_temp,
                    "distance": distance,
                    "displacement_x": dx,
                    "displacement_y": dy,
                    "method": f"template_{method_used}",
                }

            except (ValueError, TypeError) as e:
                self._log_verbose_warning(
                    f"Template matching: Error calculating movement: {str(e)}"
                )
                return None

        except Exception as e:
            LOG.error(f"Template matching: Unexpected error: {str(e)}")
            return None

    def get_orb_keypoints_and_descriptors(
        self, img1, img2, edgeThreshold=5, patchSize=10
    ):
        """
        Get ORB keypoints and descriptors with enhanced robustness.

        Args:
            img1: First image (numpy array)
            img2: Second image (numpy array)
            edgeThreshold: Edge threshold for ORB detection
            patchSize: Patch size for ORB detection

        Returns:
            Tuple of (img1_kp, img1_des, img2_kp, img2_des) or (None, None, None, None)
        """
        try:
            # Input validation
            if img1 is None or img2 is None:
                self._log_verbose_warning("ORB keypoints: Input images are None")
                return None, None, None, None

            if img1.size == 0 or img2.size == 0:
                self._log_verbose_warning("ORB keypoints: Input images are empty")
                return None, None, None, None

            # Enhanced ORB detector with error handling
            def create_orb_detector(edge_thresh, patch_sz):
                """Create ORB detector with validation"""
                try:
                    return cv2.ORB_create(
                        nfeatures=1000,
                        edgeThreshold=edge_thresh,
                        patchSize=patch_sz,
                        fastThreshold=20,
                        scaleFactor=1.2,
                        nlevels=8,
                    )
                except Exception as e:
                    self._log_verbose_warning(
                        f"ORB keypoints: Failed to create detector: {str(e)}"
                    )
                    return None

            orb = create_orb_detector(edgeThreshold, patchSize)
            if orb is None:
                return None, None, None, None

            # Detect keypoints and descriptors with error handling
            try:
                img1_kp, img1_des = orb.detectAndCompute(img1, None)
                img2_kp, img2_des = orb.detectAndCompute(img2, None)
            except Exception as e:
                self._log_verbose_warning(
                    f"ORB keypoints: Detection failed: {str(e)}"
                )
                return None, None, None, None

            # Validate results
            if (
                img1_kp is None
                or img2_kp is None
                or len(img1_kp) == 0
                or len(img2_kp) == 0
                or img1_des is None
                or img2_des is None
            ):
                # Try with more relaxed parameters if initial attempt failed
                if patchSize > 4:
                    new_patch_size = patchSize - 4
                    new_edge_threshold = max(1, int(new_patch_size / 2))
                    LOG.info(
                        f"ORB keypoints: Retrying with relaxed parameters: edge={new_edge_threshold}, patch={new_patch_size}"
                    )
                    return self.get_orb_keypoints_and_descriptors(
                        img1, img2, new_edge_threshold, new_patch_size
                    )
                else:
                    self._log_verbose_warning(
                        "ORB keypoints: Could not detect sufficient keypoints"
                    )
                    return None, None, None, None

            return img1_kp, img1_des, img2_kp, img2_des

        except Exception as e:
            LOG.error(f"ORB keypoints: Unexpected error: {str(e)}")
            return None, None, None, None

    def get_sift_keypoints_and_descriptors(self, img1, img2):
        """
        Get SIFT keypoints and descriptors with enhanced robustness.

        Args:
            img1: First image (numpy array)
            img2: Second image (numpy array)

        Returns:
            Tuple of (img1_kp, img1_des, img2_kp, img2_des) or (None, None, None, None)
        """
        try:
            # Input validation
            if img1 is None or img2 is None:
                self._log_verbose_warning("SIFT keypoints: Input images are None")
                return None, None, None, None

            if img1.size == 0 or img2.size == 0:
                self._log_verbose_warning("SIFT keypoints: Input images are empty")
                return None, None, None, None

            # Enhanced SIFT detector with multiple parameter sets
            sift_configs = [
                {"contrastThreshold": 0.04, "edgeThreshold": 10},  # Default
                {"contrastThreshold": 0.02, "edgeThreshold": 15},  # More sensitive
                {"contrastThreshold": 0.06, "edgeThreshold": 8},  # Less sensitive
            ]

            for config in sift_configs:
                try:
                    sift = cv2.SIFT_create(**config)
                    img1_kp, img1_des = sift.detectAndCompute(img1, None)
                    img2_kp, img2_des = sift.detectAndCompute(img2, None)

                    # Check if we have sufficient keypoints
                    if (
                        img1_kp is not None
                        and img2_kp is not None
                        and len(img1_kp) >= 2  # Reduced to match new default
                        and len(img2_kp) >= 2  # Reduced to match new default
                        and img1_des is not None
                        and img2_des is not None
                    ):
                        return img1_kp, img1_des, img2_kp, img2_des

                except Exception as e:
                    self._log_verbose_warning(
                        f"SIFT keypoints: Config {config} failed: {str(e)}"
                    )
                    continue

            self._log_verbose_warning("SIFT keypoints: All configurations failed")
            return None, None, None, None

        except Exception as e:
            LOG.error(f"SIFT keypoints: Unexpected error: {str(e)}")
            return None, None, None, None

    def find_partial_image_distance_with_orb(self, img, template):
        """
        Find movement distance using ORB feature matching with enhanced robustness.

        Args:
            img: Source image (numpy array)
            template: Template image (numpy array)

        Returns:
            Dictionary with movement information or None
        """
        try:
            # Enhanced input validation
            if img is None or template is None:
                self._log_verbose_warning("ORB: Input images are None")
                return None

            if not isinstance(img, np.ndarray) or not isinstance(template, np.ndarray):
                self._log_verbose_warning("ORB: Input images must be numpy arrays")
                return None

            if img.size == 0 or template.size == 0:
                self._log_verbose_warning("ORB: Input images are empty")
                return None

            # Check minimum dimensions
            if (
                img.shape[0] < 10
                or img.shape[1] < 10
                or template.shape[0] < 10
                or template.shape[1] < 10
            ):
                self._log_verbose_warning("ORB: Images too small for ORB detection")
                return None

            # Adaptive grayscale conversion
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img.copy()

            if len(template.shape) == 3:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template.copy()

            h, w = template_gray.shape

            # Enhanced difference mask calculation
            def calculate_robust_difference_mask(img1, img2):
                """Calculate difference mask with noise reduction and morphological operations"""
                # Apply slight blur to reduce noise
                img1_blur = cv2.GaussianBlur(img1, (3, 3), 0.5)
                img2_blur = cv2.GaussianBlur(img2, (3, 3), 0.5)

                # Calculate absolute difference
                mask = cv2.absdiff(img1_blur, img2_blur)

                # Use adaptive threshold
                threshold_val = max(10, np.mean(mask) + np.std(mask))
                mask[mask > threshold_val] = 255
                mask[mask <= threshold_val] = 0

                # Morphological operations to clean up the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                return mask

            mask = calculate_robust_difference_mask(img_gray, template_gray)

            # Enhanced contour detection with validation
            try:
                cnts, hierarchy = cv2.findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
            except Exception as e:
                self._log_verbose_warning(
                    f"ORB: Contour detection failed: {str(e)}"
                )
                return None

            if len(cnts) == 0:
                self._log_verbose_warning("ORB: No contours found in difference mask")
                return None

            # Enhanced contour processing
            def create_enhanced_contour_mask(contours, image_shape):
                """Create contour mask with area filtering and merging"""
                # Filter contours by area - more lenient for text
                min_area = 10  # Reduced for text detection
                valid_contours = [
                    cnt for cnt in contours if cv2.contourArea(cnt) >= min_area
                ]

                if not valid_contours:
                    self._log_verbose_warning(
                        "ORB: No valid contours after area filtering"
                    )
                    return None

                # Create mask from valid contours
                contour_mask = np.zeros(image_shape, np.uint8)
                for cnt in valid_contours:
                    cv2.drawContours(contour_mask, [cnt], -1, 255, -1)

                # Additional morphological operations for better connectivity
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, kernel)
                contour_mask = cv2.dilate(contour_mask, kernel, iterations=1)

                return contour_mask

            contour_mask = create_enhanced_contour_mask(cnts, mask.shape)
            if contour_mask is None:
                return None

            # Enhanced masked image creation
            def create_masked_images_for_orb(img, template, mask):
                """Create masked images optimized for ORB detection"""
                # Create inverse mask
                inv_mask = cv2.bitwise_not(mask)

                # Apply mask to preserve difference areas
                masked_img = cv2.bitwise_or(img, inv_mask)
                masked_template = cv2.bitwise_or(template, inv_mask)

                return masked_img, masked_template

            masked_img, masked_template = create_masked_images_for_orb(
                img_gray, template_gray, contour_mask
            )

            # Enhanced edge detection with adaptive parameters
            def calculate_adaptive_edges(image):
                """Calculate edges with adaptive parameters based on image content"""
                # Calculate image statistics for adaptive thresholding
                mean_intensity = np.mean(image)
                std_intensity = np.std(image)

                # Adaptive Canny thresholds
                lower_thresh = max(50, mean_intensity - std_intensity)
                upper_thresh = min(200, mean_intensity + 2 * std_intensity)

                # Apply Gaussian blur before edge detection
                blurred = cv2.GaussianBlur(image, (3, 3), 0)
                edges = cv2.Canny(blurred, int(lower_thresh), int(upper_thresh))

                return edges

            edges_img = calculate_adaptive_edges(masked_img)
            edges_template = calculate_adaptive_edges(masked_template)

            # Validate edge content
            if np.sum(edges_img) < 100 or np.sum(edges_template) < 100:
                self._log_verbose_warning(
                    "ORB: Insufficient edge content after masking"
                )
                return None

            # Enhanced ORB keypoint detection with adaptive parameters
            def get_robust_orb_keypoints(template_edges, img_edges):
                """Get ORB keypoints with multiple parameter sets for robustness"""
                orb_configs = [
                    # (nfeatures, edgeThreshold, patchSize)
                    (1000, 5, 10),  # Default sensitive
                    (1500, 3, 8),  # More sensitive
                    (2000, 7, 12),  # Less sensitive, more features
                    (800, 10, 15),  # Least sensitive
                ]

                for nfeatures, edge_thresh, patch_size in orb_configs:
                    try:
                        orb = cv2.ORB_create(
                            nfeatures=nfeatures,
                            edgeThreshold=edge_thresh,
                            patchSize=patch_size,
                            fastThreshold=20,
                            scaleFactor=1.2,
                            nlevels=8,
                        )

                        template_kp, template_des = orb.detectAndCompute(
                            template_edges, None
                        )
                        img_kp, img_des = orb.detectAndCompute(img_edges, None)

                        # Check for sufficient keypoints
                        if (
                            template_kp is not None
                            and img_kp is not None
                            and len(template_kp) >= self.orb_min_matches
                            and len(img_kp) >= self.orb_min_matches
                            and template_des is not None
                            and img_des is not None
                        ):
                            return template_kp, template_des, img_kp, img_des

                    except Exception as e:
                        self._log_verbose_warning(
                            f"ORB: Configuration {orb_configs.index((nfeatures, edge_thresh, patch_size))} failed: {str(e)}"
                        )
                        continue

                return None, None, None, None

            (
                template_keypoints,
                template_descriptors,
                target_keypoints,
                target_descriptors,
            ) = get_robust_orb_keypoints(edges_template, edges_img)

            if (
                template_keypoints is None
                or target_keypoints is None
                or len(template_keypoints) < self.orb_min_matches
                or len(target_keypoints) < self.orb_min_matches
            ):
                self._log_verbose_warning("ORB: Insufficient keypoints detected")
                return None

            # Enhanced feature matching with robust filtering
            def perform_robust_matching(desc1, desc2):
                """Perform feature matching with multiple validation steps"""
                try:
                    # Use BruteForce matcher with Hamming distance for ORB
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(desc1, desc2)

                    if len(matches) == 0:
                        return []

                    # Sort matches by distance
                    matches = sorted(matches, key=lambda x: x.distance)

                    # Filter matches by distance threshold
                    good_matches = []
                    if len(matches) > 0:
                        # Adaptive distance threshold based on match quality distribution
                        distances = [m.distance for m in matches]
                        mean_dist = np.mean(distances)
                        std_dist = np.std(distances)

                        # Use mean - std as threshold, but with bounds
                        distance_thresh = max(30, min(80, mean_dist - std_dist))

                        good_matches = [
                            m for m in matches if m.distance < distance_thresh
                        ]

                    # Additional filtering: remove matches that are too clustered
                    if len(good_matches) > 10:
                        # Keep only the best matches
                        good_matches = good_matches[
                            : min(self.orb_max_matches, len(good_matches))
                        ]

                    return good_matches

                except Exception as e:
                    self._log_verbose_warning(f"ORB: Matching failed: {str(e)}")
                    return []

            best_matches = perform_robust_matching(
                template_descriptors, target_descriptors
            )

            # Use adaptive minimum matches for small text areas
            effective_min_matches = min(
                self.orb_min_matches, len(best_matches) if len(best_matches) >= 2 else 0
            )

            if len(best_matches) < effective_min_matches:
                self._log_verbose_warning(
                    f"ORB: Insufficient good matches: {len(best_matches)} < {effective_min_matches}"
                )
                return None

            # Enhanced homography computation with validation
            try:
                # Extract point correspondences
                src_pts = np.float32(
                    [template_keypoints[m.queryIdx].pt for m in best_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [target_keypoints[m.trainIdx].pt for m in best_matches]
                ).reshape(-1, 1, 2)

                # Robust homography estimation with multiple RANSAC attempts
                ransac_configs = [
                    (self.ransac_threshold, 1000, 0.99),
                    (self.ransac_threshold * 1.5, 2000, 0.95),
                    (self.ransac_threshold * 0.5, 3000, 0.999),
                ]

                best_homography = None
                best_inlier_ratio = 0

                for ransac_thresh, max_iters, confidence in ransac_configs:
                    try:
                        M, mask = cv2.findHomography(
                            src_pts,
                            dst_pts,
                            cv2.RANSAC,
                            ransac_thresh,
                            maxIters=max_iters,
                            confidence=confidence,
                        )

                        if M is not None and self._validate_homography_matrix(M):
                            # Calculate inlier ratio
                            inlier_count = np.sum(mask) if mask is not None else 0
                            inlier_ratio = inlier_count / len(best_matches)

                            if (
                                inlier_ratio > best_inlier_ratio
                                and inlier_ratio
                                >= 0.15  # Reduced threshold for text detection
                            ):
                                best_homography = M
                                best_inlier_ratio = inlier_ratio

                    except Exception as e:
                        self._log_verbose_warning(
                            f"ORB: Homography computation failed with config {ransac_configs.index((ransac_thresh, max_iters, confidence))}: {str(e)}"
                        )
                        continue

                if best_homography is None:
                    self._log_verbose_warning("ORB: Could not compute valid homography")
                    return None

                # Extract movement from homography
                dx = float(best_homography[0, 2])
                dy = float(best_homography[1, 2])

                # Validate movement magnitude
                max_reasonable_movement = max(img_gray.shape) * 2
                if (
                    abs(dx) > max_reasonable_movement
                    or abs(dy) > max_reasonable_movement
                ):
                    self._log_verbose_warning(
                        f"ORB: Unreasonable movement detected: dx={dx}, dy={dy}"
                    )
                    return None

                movement = float(np.sqrt(dx**2 + dy**2))

                # Add visualization for debugging
                if self.take_screenshots and len(best_matches) > 0:
                    try:
                        match_img = cv2.drawMatches(
                            masked_template,
                            template_keypoints,
                            masked_img,
                            target_keypoints,
                            best_matches[
                                : min(20, len(best_matches))
                            ],  # Limit to 20 matches for clarity
                            None,
                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                        )
                        self.add_screenshot_to_log(match_img, "ORB_matches")
                    except Exception as e:
                        self._log_verbose_warning(
                            f"ORB: Could not create match visualization: {str(e)}"
                        )

                return {
                    "distance": movement,
                    "displacement_x": dx,
                    "displacement_y": dy,
                    "inlier_ratio": best_inlier_ratio,
                    "num_matches": len(best_matches),
                    "method": "orb",
                }

            except Exception as e:
                self._log_verbose_warning(
                    f"ORB: Error in homography computation: {str(e)}"
                )
                return None

        except Exception as e:
            LOG.error(f"ORB: Unexpected error in movement detection: {str(e)}")
            return None

    def blend_two_images(self, image, overlay, ignore_color=[255, 255, 255]):
        ignore_color = np.asarray(ignore_color)
        mask = ~(overlay == ignore_color).all(-1)
        # Or mask = (overlay!=ignore_color).any(-1)
        out = image.copy()
        out[mask] = image[mask] * 0.5 + overlay[mask] * 0.5
        return out

    def is_bounding_box_reasonable(self, corners):
        """Check if the bounding box is spatially consistent (rectangular and not too skewed)."""
        try:
            # Reshape corners to access individual points
            corners = corners.reshape(4, 2)
            tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]

            # Calculate side lengths
            width_top = np.linalg.norm(tr - tl)
            width_bottom = np.linalg.norm(br - bl)
            height_left = np.linalg.norm(bl - tl)
            height_right = np.linalg.norm(br - tr)

            # Check if the bounding box is reasonable:
            # 1. All sides should have reasonable length (not zero or negative)
            if min(width_top, width_bottom, height_left, height_right) <= 0:
                return False

            # 2. Opposite sides should be similar in length (within 50% tolerance)
            width_ratio = min(width_top, width_bottom) / max(width_top, width_bottom)
            height_ratio = min(height_left, height_right) / max(
                height_left, height_right
            )

            if width_ratio < 0.5 or height_ratio < 0.5:
                return False

            # 3. Check for reasonable aspect ratio (not too skewed)
            avg_width = (width_top + width_bottom) / 2
            avg_height = (height_left + height_right) / 2
            aspect_ratio = max(avg_width, avg_height) / min(avg_width, avg_height)

            # Allow aspect ratios up to 10:1
            if aspect_ratio > 10:
                return False

            # 4. Check if points form a reasonable quadrilateral (no self-intersections)
            # Calculate area using shoelace formula
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]
            area = 0.5 * abs(
                sum(
                    x_coords[i] * y_coords[(i + 1) % 4]
                    - x_coords[(i + 1) % 4] * y_coords[i]
                    for i in range(4)
                )
            )

            # Area should be positive and reasonable
            min_expected_area = (
                avg_width * avg_height * 0.1
            )  # At least 10% of expected rectangular area
            if area < min_expected_area:
                return False

            return True

        except Exception as e:
            print(f"Error validating bounding box: {e}")
            return False

        # Ensure the width and height are approximately consistent (not too skewed)
        width_diff = abs(width_top - width_bottom)
        height_diff = abs(height_left - height_right)

        return width_diff < 20 and height_diff < 20  # Thresholds can be fine-tuned

    def concatenate_images_safely(self, img1, img2, axis=1, fill_color=(255, 255, 255)):
        """
        Safely concatenate two images with potentially different dimensions.

        Args:
            img1: First image (numpy array)
            img2: Second image (numpy array)
            axis: Axis along which to concatenate (0 for vertical, 1 for horizontal)
            fill_color: Color to fill empty areas (BGR format, default is white)

        Returns:
            Concatenated image with both images padded to the same dimensions
        """
        # Get dimensions of both images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if axis == 1:  # Horizontal concatenation
            # Make both images the same height
            max_height = max(h1, h2)

            # Pad images to same height if needed
            if h1 < max_height:
                pad_height = max_height - h1
                if len(img1.shape) == 3:  # Color image
                    padding = np.full(
                        (pad_height, w1, img1.shape[2]), fill_color, dtype=img1.dtype
                    )
                else:  # Grayscale image
                    padding = np.full((pad_height, w1), fill_color[0], dtype=img1.dtype)
                img1 = np.vstack([img1, padding])

            if h2 < max_height:
                pad_height = max_height - h2
                if len(img2.shape) == 3:  # Color image
                    padding = np.full(
                        (pad_height, w2, img2.shape[2]), fill_color, dtype=img2.dtype
                    )
                else:  # Grayscale image
                    padding = np.full((pad_height, w2), fill_color[0], dtype=img2.dtype)
                img2 = np.vstack([img2, padding])

            # Now concatenate horizontally
            return np.hstack([img1, img2])

        else:  # Vertical concatenation (axis == 0)
            # Make both images the same width
            max_width = max(w1, w2)

            # Pad images to same width if needed
            if w1 < max_width:
                pad_width = max_width - w1
                if len(img1.shape) == 3:  # Color image
                    padding = np.full(
                        (h1, pad_width, img1.shape[2]), fill_color, dtype=img1.dtype
                    )
                else:  # Grayscale image
                    padding = np.full((h1, pad_width), fill_color[0], dtype=img1.dtype)
                img1 = np.hstack([img1, padding])

            if w2 < max_width:
                pad_width = max_width - w2
                if len(img2.shape) == 3:  # Color image
                    padding = np.full(
                        (h2, pad_width, img2.shape[2]), fill_color, dtype=img2.dtype
                    )
                else:  # Grayscale image
                    padding = np.full((h2, pad_width), fill_color[0], dtype=img2.dtype)
                img2 = np.hstack([img2, padding])

            # Now concatenate vertically
            return np.vstack([img1, img2])

    def _init_movement_detection_stats(self):
        """Initialize movement detection statistics tracking."""
        if not hasattr(self, "_movement_stats"):
            self._movement_stats = {
                "total_attempts": 0,
                "successful_detections": 0,
                "method_success": {"template": 0, "sift": 0, "orb": 0},
                "method_attempts": {"template": 0, "sift": 0, "orb": 0},
                "fallback_usage": 0,
                "error_count": 0,
                "average_processing_time": 0.0,
            }

    def _record_movement_detection_attempt(
        self, method, success, processing_time=0.0, used_fallback=False
    ):
        """Record statistics for movement detection attempts."""
        self._init_movement_detection_stats()

        self._movement_stats["total_attempts"] += 1
        self._movement_stats["method_attempts"][method] += 1

        if success:
            self._movement_stats["successful_detections"] += 1
            self._movement_stats["method_success"][method] += 1

        if used_fallback:
            self._movement_stats["fallback_usage"] += 1

        # Update average processing time
        current_avg = self._movement_stats["average_processing_time"]
        total_attempts = self._movement_stats["total_attempts"]
        self._movement_stats["average_processing_time"] = (
            current_avg * (total_attempts - 1) + processing_time
        ) / total_attempts

    def _record_movement_detection_error(self):
        """Record a movement detection error."""
        self._init_movement_detection_stats()
        self._movement_stats["error_count"] += 1

    def get_movement_detection_statistics(self):
        """
        Get movement detection performance statistics.

        Returns:
            Dictionary with detection statistics and performance metrics
        """
        self._init_movement_detection_stats()
        stats = self._movement_stats.copy()

        # Calculate success rates
        if stats["total_attempts"] > 0:
            stats["overall_success_rate"] = (
                stats["successful_detections"] / stats["total_attempts"]
            )
            stats["error_rate"] = stats["error_count"] / stats["total_attempts"]
            stats["fallback_rate"] = stats["fallback_usage"] / stats["total_attempts"]
        else:
            stats["overall_success_rate"] = 0.0
            stats["error_rate"] = 0.0
            stats["fallback_rate"] = 0.0

        # Calculate method-specific success rates
        for method in ["template", "sift", "orb"]:
            attempts = stats["method_attempts"][method]
            if attempts > 0:
                stats[f"{method}_success_rate"] = (
                    stats["method_success"][method] / attempts
                )
            else:
                stats[f"{method}_success_rate"] = 0.0

        return stats

    def reset_movement_detection_statistics(self):
        """Reset movement detection statistics."""
        self._movement_stats = {
            "total_attempts": 0,
            "successful_detections": 0,
            "method_success": {"template": 0, "sift": 0, "orb": 0},
            "method_attempts": {"template": 0, "sift": 0, "orb": 0},
            "fallback_usage": 0,
            "error_count": 0,
            "average_processing_time": 0.0,
        }

    def _validate_movement_result(self, result, method, img_shape):
        """
        Validate movement detection result for reasonableness.

        Args:
            result: Movement detection result dictionary
            method: Detection method used
            img_shape: Shape of the image being processed

        Returns:
            bool: True if result appears valid
        """
        if result is None:
            return False

        try:
            # Check required fields
            if "distance" not in result:
                return False

            distance = result["distance"]

            # Distance should be non-negative and reasonable
            if distance < 0 or not np.isfinite(distance):
                return False

            # Distance should not exceed reasonable bounds (2x image diagonal)
            max_distance = np.sqrt(img_shape[0] ** 2 + img_shape[1] ** 2) * 2
            if distance > max_distance:
                self._log_verbose_warning(
                    f"Movement distance {distance} exceeds reasonable bound {max_distance}"
                )
                return False

            # Check displacement values if present
            if "displacement_x" in result and "displacement_y" in result:
                dx, dy = result["displacement_x"], result["displacement_y"]
                if not (np.isfinite(dx) and np.isfinite(dy)):
                    return False

                # Displacement should be consistent with distance
                calculated_distance = np.sqrt(dx**2 + dy**2)
                if abs(calculated_distance - distance) > 1e-6:
                    self._log_verbose_warning(
                        f"Distance inconsistency: reported={distance}, calculated={calculated_distance}"
                    )
                    return False

            # Method-specific validations
            if method == "sift" or method == "orb":
                # Feature-based methods should have additional metrics
                if "num_matches" in result:
                    num_matches = result["num_matches"]
                    min_matches = (
                        self.sift_min_matches
                        if method == "sift"
                        else self.orb_min_matches
                    )
                    if num_matches < min_matches:
                        return False

                if "inlier_ratio" in result:
                    inlier_ratio = result["inlier_ratio"]
                    if (
                        inlier_ratio < 0.15
                    ):  # More lenient: require at least 15% inliers
                        return False

            return True

        except Exception as e:
            self._log_verbose_warning(
                f"Error validating movement result: {str(e)}"
            )
            return False

    def _create_fallback_result(self, method, reason="unknown"):
        """
        Create a minimal fallback result when detection fails.

        Args:
            method: The detection method that failed
            reason: Reason for fallback

        Returns:
            Dictionary with minimal movement information
        """
        return {
            "distance": 1.0,  # Small movement to indicate potential change
            "displacement_x": 0.0,
            "displacement_y": 0.0,
            "method": f"{method}_fallback",
            "fallback_reason": reason,
            "confidence": "low",
        }

    def load_watermarks(self, watermark_file):
        """
        Load watermark files with enhanced error handling and validation.

        Args:
            watermark_file: Path to watermark file or directory, or list of paths

        Returns:
            List of processed watermark masks
        """
        if isinstance(watermark_file, str):
            watermarks = []
            if os.path.isdir(watermark_file):
                try:
                    # Get all image files from directory
                    image_extensions = {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".bmp",
                        ".tiff",
                        ".tif",
                        ".pdf",
                    }
                    watermark_files = [
                        str(os.path.join(watermark_file, f))
                        for f in os.listdir(watermark_file)
                        if (
                            os.path.isfile(os.path.join(watermark_file, f))
                            and os.path.splitext(f.lower())[1] in image_extensions
                        )
                    ]
                    if not watermark_files:
                        LOG.warning(
                            f"No valid image files found in watermark directory: {watermark_file}"
                        )
                        return []
                except Exception as e:
                    LOG.error(
                        f"Error reading watermark directory {watermark_file}: {str(e)}"
                    )
                    return []
            else:
                watermark_files = [watermark_file]

            if isinstance(watermark_files, list):
                for single_watermark in watermark_files:
                    try:
                        # Validate file exists
                        if not os.path.exists(single_watermark):
                            LOG.warning(
                                f"Watermark file does not exist: {single_watermark}"
                            )
                            continue

                        # Load watermark document
                        watermark = (
                            DocumentRepresentation(single_watermark).pages[0].image
                        )

                        # Handle RGBA images
                        if len(watermark.shape) == 3 and watermark.shape[2] == 4:
                            watermark = watermark[:, :, :3]

                        # Convert to grayscale with validation
                        if len(watermark.shape) == 3:
                            watermark_gray = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
                        else:
                            watermark_gray = watermark.copy()

                        # Create binary mask using OTSU thresholding
                        # This method automatically determines the optimal threshold
                        ret, mask = cv2.threshold(
                            watermark_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                        )

                        if mask is not None:
                            # Log mask statistics for debugging
                            mask_pixels = np.sum(mask > 0)
                            content_ratio = mask_pixels / watermark_gray.size
                            print(
                                f"  Watermark mask: {mask_pixels} white pixels ({content_ratio:.1%})"
                            )

                            watermarks.append(mask)
                            LOG.info(
                                f"Successfully loaded watermark: {single_watermark}"
                            )
                        else:
                            LOG.warning(
                                f"Could not create valid mask for watermark: {single_watermark}"
                            )

                    except Exception as e:
                        LOG.warning(
                            f"Watermark file {single_watermark} could not be loaded: {str(e)}"
                        )
                        continue

            return watermarks

        elif isinstance(watermark_file, list):
            # Handle list of watermark files
            all_watermarks = []
            for single_file in watermark_file:
                watermarks = self.load_watermarks(single_file)  # Recursive call
                all_watermarks.extend(watermarks)
            return all_watermarks

        else:
            LOG.warning(f"Invalid watermark_file type: {type(watermark_file)}")
            return []
