from pathlib import Path

import pytest

from DocTest.VisualTest import VisualTest


def _prepare_visual_test(tmp_path) -> VisualTest:
    vt = VisualTest(show_diff=False, take_screenshots=False, movement_detection="orb")
    vt.output_directory = Path(tmp_path)
    vt.screenshot_path = vt.output_directory / vt.screenshot_dir
    vt.screenshot_path.mkdir(parents=True, exist_ok=True)
    return vt


def test_text_based_movement_detection_passes(tmp_path):
    vt = _prepare_visual_test(tmp_path)
    vt.compare_images(
        "testdata/invoice.pdf",
        "testdata/invoice_moved.pdf",
        move_tolerance=20,
        movement_detection="text",
    )


def test_text_based_movement_detection_detects_text_changes(tmp_path):
    vt = _prepare_visual_test(tmp_path)
    with pytest.raises(AssertionError):
        vt.compare_images(
            "testdata/invoice.pdf",
            "testdata/invoice_moved_and_different.pdf",
            move_tolerance=25,
            movement_detection="text",
        )


def test_set_movement_detection_keyword(tmp_path):
    vt = _prepare_visual_test(tmp_path)
    vt.set_movement_detection("text")
    vt.compare_images(
        "testdata/invoice.pdf",
        "testdata/invoice_moved.pdf",
        move_tolerance=20,
    )
    vt.set_movement_detection("orb")
    with pytest.raises(AssertionError):
        vt.compare_images(
            "testdata/invoice.pdf",
            "testdata/invoice_moved.pdf",
            move_tolerance=5,
        )
