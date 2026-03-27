"""Integration tests for GitHub issues #132 and #134.

Validates that overview screenshots (_combined, _combined_with_diff,
_absolute_diff) are generated even when movement tolerance check fails.

Before the fix, _check_movement_with_text() and _check_movement_with_images()
raised AssertionError directly, which bypassed the screenshot generation code
in compare_images (lines 754-823).  The fix changed them to return False so
that the screenshot logic executes before the final AssertionError is raised.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from DocTest.VisualTest import VisualTest


# ---------------------------------------------------------------------------
# Helpers (reused from test_movetolerance.py)
# ---------------------------------------------------------------------------

def create_image(image_size=(500, 300), color=(255, 255, 255)):
    """Create a blank image of the specified size and color."""
    img = np.ones((image_size[1], image_size[0], 3), np.uint8)
    img[:] = color
    return img


def add_text_to_image(img, text, text_position=(10, 50), font_scale=1, thickness=2):
    """Add text to the image at the given position, clamping to bounds."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    image_height, image_width = img.shape[:2]

    x, y = text_position
    if x + text_size[0] > image_width:
        x = image_width - text_size[0] - 10
    if y - text_size[1] < 0:
        y = text_size[1] + 10
    if y > image_height:
        y = image_height - 10

    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img


def _save_pair(tmp_dir, text, move_distance, image_size=(500, 300)):
    """Create a reference/candidate image pair with a vertical text shift.

    Returns (ref_path, cand_path).
    """
    ref_image = create_image(image_size)
    ref_image = add_text_to_image(ref_image, text, (10, 50))
    ref_path = tmp_dir / "ref.png"
    cv2.imwrite(str(ref_path), ref_image)

    cand_image = create_image(image_size)
    cand_image = add_text_to_image(cand_image, text, (10, 50 + move_distance))
    cand_path = tmp_dir / "cand.png"
    cv2.imwrite(str(cand_path), cand_image)

    return ref_path, cand_path


# ---------------------------------------------------------------------------
# Suffix extraction helper
# ---------------------------------------------------------------------------

def _extract_suffixes(mock_obj):
    """Return the set of *suffix* arguments passed to add_screenshot_to_log."""
    suffixes = set()
    for call in mock_obj.call_args_list:
        # suffix is the second positional arg, or a keyword arg
        if len(call.args) >= 2:
            suffixes.add(call.args[1])
        elif "suffix" in call.kwargs:
            suffixes.add(call.kwargs["suffix"])
    return suffixes


def _extract_suffix_list(mock_obj):
    """Return the ordered list of *suffix* arguments passed to add_screenshot_to_log."""
    suffixes = []
    for call in mock_obj.call_args_list:
        if len(call.args) >= 2:
            suffixes.append(call.args[1])
        elif "suffix" in call.kwargs:
            suffixes.append(call.kwargs["suffix"])
    return suffixes


# ---------------------------------------------------------------------------
# Tests: movement exceeds tolerance  ->  screenshots MUST be generated
# ---------------------------------------------------------------------------

OVERVIEW_SUFFIXES = {"_combined", "_combined_with_diff", "_absolute_diff"}


@pytest.mark.parametrize("detection_method", ["template", "sift"])
def test_overview_screenshots_generated_on_failed_movement(detection_method):
    """When movement exceeds tolerance, the overview screenshots must still be
    generated (the bug fixed in #132 / #134)."""
    move_distance = 20
    # Tolerance is intentionally smaller than the move distance so the check
    # fails.
    tolerance = move_distance - 5

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        ref_path, cand_path = _save_pair(
            tmp_dir, "Issue132", move_distance, image_size=(500, 300)
        )

        lib = VisualTest(
            take_screenshots=True,
            show_diff=True,
            screenshot_format="png",
            movement_detection=detection_method,
        )

        with patch.object(
            lib, "add_screenshot_to_log", wraps=lib.add_screenshot_to_log
        ) as spy:
            with pytest.raises(AssertionError):
                lib.compare_images(
                    str(ref_path),
                    str(cand_path),
                    move_tolerance=tolerance,
                )

            suffixes = _extract_suffixes(spy)

            for expected in OVERVIEW_SUFFIXES:
                assert expected in suffixes, (
                    f"Expected suffix '{expected}' not found in screenshot calls. "
                    f"Recorded suffixes: {sorted(suffixes)}"
                )


def test_overview_screenshots_generated_on_failed_movement_with_pdf():
    """Same check as above but using the sample PDF test data that ships with
    the project.  sample_1_page.pdf vs sample_1_page_moved.pdf have known
    visual movement."""
    testdata = Path(__file__).resolve().parent / "testdata"
    ref_pdf = testdata / "sample_1_page.pdf"
    cand_pdf = testdata / "sample_1_page_moved.pdf"

    if not ref_pdf.exists() or not cand_pdf.exists():
        pytest.skip("PDF test data not available")

    lib = VisualTest(
        take_screenshots=True,
        show_diff=True,
        screenshot_format="png",
        movement_detection="template",
    )

    with patch.object(
        lib, "add_screenshot_to_log", wraps=lib.add_screenshot_to_log
    ) as spy:
        # Use a very small tolerance so the movement check fails.
        with pytest.raises(AssertionError):
            lib.compare_images(
                str(ref_pdf),
                str(cand_pdf),
                move_tolerance=1,
            )

        suffixes = _extract_suffixes(spy)

        for expected in OVERVIEW_SUFFIXES:
            assert expected in suffixes, (
                f"Expected suffix '{expected}' not found in screenshot calls. "
                f"Recorded suffixes: {sorted(suffixes)}"
            )


# ---------------------------------------------------------------------------
# Tests: movement within tolerance  ->  comparison passes, no overview shots
# ---------------------------------------------------------------------------

# Suffixes that are ONLY generated when the comparison fails (lines 784-804).
# Note: _combined is also generated unconditionally at line 456 when
# take_screenshots=True, so we cannot use it to distinguish pass/fail.
FAILURE_ONLY_SUFFIXES = {"_combined_with_diff", "_absolute_diff"}


@pytest.mark.parametrize("detection_method", ["template", "sift"])
def test_no_failure_screenshots_when_movement_within_tolerance(detection_method):
    """When movement is within tolerance the comparison should pass.  The
    failure-specific screenshots (_combined_with_diff, _absolute_diff) should
    NOT be generated because the images are considered similar."""
    move_distance = 10
    # Tolerance is larger than the move distance so the check passes.
    tolerance = move_distance + 5

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        ref_path, cand_path = _save_pair(
            tmp_dir, "Issue134", move_distance, image_size=(500, 300)
        )

        lib = VisualTest(
            take_screenshots=True,
            show_diff=True,
            screenshot_format="png",
            movement_detection=detection_method,
        )

        with patch.object(
            lib, "add_screenshot_to_log", wraps=lib.add_screenshot_to_log
        ) as spy:
            # Should NOT raise -- movement is within tolerance.
            lib.compare_images(
                str(ref_path),
                str(cand_path),
                move_tolerance=tolerance,
            )

            suffixes = _extract_suffixes(spy)

            for unexpected in FAILURE_ONLY_SUFFIXES:
                assert unexpected not in suffixes, (
                    f"Unexpected suffix '{unexpected}' found when comparison should "
                    f"have passed.  Recorded suffixes: {sorted(suffixes)}"
                )


# ---------------------------------------------------------------------------
# Test: returned value is False (not raised), so screenshots run
# ---------------------------------------------------------------------------

def test_check_movement_with_images_returns_false_not_raises():
    """Directly verify that _check_movement_with_images returns False rather
    than raising AssertionError when movement exceeds tolerance."""
    move_distance = 25

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        ref_path, cand_path = _save_pair(
            tmp_dir, "ReturnCheck", move_distance, image_size=(500, 300)
        )

        lib = VisualTest(
            take_screenshots=True,
            show_diff=True,
            screenshot_format="png",
            movement_detection="template",
        )

        # Build lightweight page-like objects using DocumentRepresentation.
        from DocTest.DocumentRepresentation import DocumentRepresentation

        ref_doc = DocumentRepresentation(str(ref_path), lib.dpi)
        cand_doc = DocumentRepresentation(str(cand_path), lib.dpi)

        ref_page = ref_doc.get_page(0)
        cand_page = cand_doc.get_page(0)

        # Manufacture a diff rectangle covering the full image.
        h, w = ref_page.image.shape[:2]
        diff_rect = {"x": 0, "y": 0, "width": w, "height": h}

        # The method must return False, NOT raise.
        result = lib._check_movement_with_images(
            ref_page=ref_page,
            cand_page=cand_page,
            diff_rectangles=[diff_rect],
            move_tolerance=1,  # intentionally tiny
            detection_method="template",
        )

        assert result is False, (
            "_check_movement_with_images should return False when movement "
            "exceeds tolerance, not raise AssertionError"
        )

        ref_doc.close()
        cand_doc.close()


# ---------------------------------------------------------------------------
# Test: screenshot ordering — overview before per-area movement details
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("detection_method", ["template", "sift"])
def test_overview_screenshots_appear_before_movement_area_screenshots(detection_method):
    """Overview screenshots (_combined, _combined_with_diff, _absolute_diff)
    must appear in the log BEFORE the per-area movement screenshots
    (_moved_area, _diff_area_blended, etc.)."""
    move_distance = 20
    tolerance = move_distance - 5  # fail on purpose

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        ref_path, cand_path = _save_pair(
            tmp_dir, "Order", move_distance, image_size=(500, 300)
        )

        lib = VisualTest(
            take_screenshots=True,
            show_diff=True,
            screenshot_format="png",
            movement_detection=detection_method,
        )

        with patch.object(
            lib, "add_screenshot_to_log", wraps=lib.add_screenshot_to_log
        ) as spy:
            with pytest.raises(AssertionError):
                lib.compare_images(
                    str(ref_path),
                    str(cand_path),
                    move_tolerance=tolerance,
                )

            suffix_list = _extract_suffix_list(spy)

            # Find the index of the last overview screenshot
            overview = {"_combined_with_diff", "_absolute_diff"}
            last_overview_idx = -1
            for i, s in enumerate(suffix_list):
                if s in overview:
                    last_overview_idx = i

            # Find the index of the first per-area movement screenshot
            movement_area = {"_moved_area", "_diff_area_blended"}
            first_movement_idx = len(suffix_list)
            for i, s in enumerate(suffix_list):
                if s in movement_area:
                    first_movement_idx = i
                    break

            assert last_overview_idx != -1, (
                f"No overview screenshots found. Suffixes: {suffix_list}"
            )
            assert first_movement_idx < len(suffix_list), (
                f"No movement area screenshots found. Suffixes: {suffix_list}"
            )
            assert last_overview_idx < first_movement_idx, (
                f"Overview screenshots must appear before movement area screenshots. "
                f"Last overview at index {last_overview_idx}, "
                f"first movement area at index {first_movement_idx}. "
                f"Full order: {suffix_list}"
            )
