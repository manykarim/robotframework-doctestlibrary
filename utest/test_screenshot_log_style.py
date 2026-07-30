"""Screenshots logged to the Robot log must never be upscaled (issue #140).

`width:50%` forced every image to exactly half the log column, so small
crops (the moved-area / diff-area images from a move_tolerance failure)
were blown up. `max-width:50%` bounds large images exactly as before while
letting small ones render at their natural size.

Note: `robot_logger` is `robot.api.logger`, which outside a Robot run routes
to the Python logger named "RobotFramework" — so `caplog` on the module
logger captures nothing. These tests patch the imported symbol instead.
"""

from unittest.mock import patch

import numpy as np
import pytest

from DocTest.VisualTest import VisualTest

BOUNDED_STYLE = "max-width:50%; height: auto;"
NATURAL_STYLE = "width: auto; height: auto;"
# The regressed form. Anchored to the start of the style attribute so it is
# not matched by the "max-width:50%" substring.
FIXED_WIDTH_STYLE = 'style="width:50%'


@pytest.fixture
def small_image():
    """A crop far narrower than half a log column — the issue #140 case."""
    return np.full((20, 40, 3), 200, dtype=np.uint8)


def _emit(tester, image, **kwargs):
    with patch("DocTest.VisualTest.robot_logger") as logger:
        tester.add_screenshot_to_log(image, "_diff_area", **kwargs)
    assert logger.info.called, "no screenshot was logged"
    return logger.info.call_args[0][0]


def test_default_file_link_path_bounds_without_upscaling(small_image, tmp_path, monkeypatch):
    """The shipped default: screenshot_format='jpg', embed_screenshots=False."""
    monkeypatch.chdir(tmp_path)
    tester = VisualTest()
    assert tester.screenshot_format == "jpg"
    assert tester.embed_screenshots is False

    html = _emit(tester, small_image)

    assert f'style="{BOUNDED_STYLE}"' in html
    assert FIXED_WIDTH_STYLE not in html
    # full-size inspection must still be reachable
    assert 'target="_blank"' in html


@pytest.mark.parametrize("screenshot_format", ["jpg", "png"])
def test_embedded_paths_bound_without_upscaling(
    small_image, tmp_path, monkeypatch, screenshot_format
):
    monkeypatch.chdir(tmp_path)
    tester = VisualTest(
        embed_screenshots=True, screenshot_format=screenshot_format
    )

    html = _emit(tester, small_image)

    assert f'style="{BOUNDED_STYLE}"' in html
    assert FIXED_WIDTH_STYLE not in html
    assert "base64," in html


def test_original_size_mode_stays_unbounded(small_image, tmp_path, monkeypatch):
    """`original_size=True` exists so template screenshots keep natural size."""
    monkeypatch.chdir(tmp_path)
    tester = VisualTest(embed_screenshots=True)

    html = _emit(tester, small_image, original_size=True)

    assert f'style="{NATURAL_STYLE}"' in html
    assert "max-width" not in html


def test_large_image_is_still_bounded(tmp_path, monkeypatch):
    """Large screenshots keep the same half-column bound as before."""
    monkeypatch.chdir(tmp_path)
    tester = VisualTest(embed_screenshots=True)
    large = np.full((1200, 1600, 3), 128, dtype=np.uint8)

    html = _emit(tester, large)

    assert f'style="{BOUNDED_STYLE}"' in html
