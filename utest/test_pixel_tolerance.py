"""Pixel-difference tolerance (max_diff_pixels / max_diff_ratio) on Compare Images."""

import cv2
import numpy as np
import pytest

from DocTest.VisualTest import VisualTest


def _pair(tmp_path, pixels=0, delta=255, size=(100, 200)):
    """Reference plus candidate differing in exactly ``pixels`` pixels by ``delta``."""
    reference = np.full((*size, 3), 255, dtype=np.uint8)
    candidate = reference.copy()
    changed = 0
    for row in range(size[0]):
        for col in range(size[1]):
            if changed >= pixels:
                break
            candidate[row, col] = 255 - delta
            changed += 1
        if changed >= pixels:
            break
    ref_path, cand_path = tmp_path / "ref.png", tmp_path / "cand.png"
    cv2.imwrite(str(ref_path), reference)
    cv2.imwrite(str(cand_path), candidate)
    return str(ref_path), str(cand_path)


@pytest.fixture
def visual(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return VisualTest()


def test_diff_within_pixel_budget_passes(visual, tmp_path):
    ref, cand = _pair(tmp_path, pixels=50)
    with pytest.raises(AssertionError):
        visual.compare_images(ref, cand)
    visual.compare_images(ref, cand, max_diff_pixels=100)


def test_diff_exceeding_pixel_budget_fails(visual, tmp_path):
    ref, cand = _pair(tmp_path, pixels=50)
    with pytest.raises(AssertionError):
        visual.compare_images(ref, cand, max_diff_pixels=10)


def test_ratio_budget_boundaries(visual, tmp_path):
    ref, cand = _pair(tmp_path, pixels=50)  # 50 / 20000 = 0.25%
    visual.compare_images(ref, cand, max_diff_ratio=0.01)
    with pytest.raises(AssertionError):
        visual.compare_images(ref, cand, max_diff_ratio=0.001)


def test_both_budgets_must_hold(visual, tmp_path):
    ref, cand = _pair(tmp_path, pixels=50)
    with pytest.raises(AssertionError):
        visual.compare_images(ref, cand, max_diff_pixels=100, max_diff_ratio=0.001)


def test_faint_noise_below_intensity_threshold_is_free(visual, tmp_path):
    # deltas of 5 (< default 20) — a strict zero-pixel budget must still pass
    ref, cand = _pair(tmp_path, pixels=500, delta=5)
    visual.compare_images(ref, cand, max_diff_pixels=0)


def test_intensity_threshold_is_tunable(visual, tmp_path):
    ref, cand = _pair(tmp_path, pixels=500, delta=5)
    with pytest.raises(AssertionError):
        visual.compare_images(ref, cand, max_diff_pixels=0, pixel_intensity_threshold=2)


def test_dimension_mismatch_is_never_rescued(visual, tmp_path):
    ref, _ = _pair(tmp_path, pixels=0)
    small = np.full((50, 50, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "small.png"), small)
    with pytest.raises(AssertionError):
        visual.compare_images(ref, str(tmp_path / "small.png"), max_diff_pixels=10_000_000)


def test_robot_string_kwargs_are_converted(visual, tmp_path):
    # RF passes **kwargs values as strings
    ref, cand = _pair(tmp_path, pixels=50)
    visual.compare_images(ref, cand, max_diff_pixels="100")
    visual.compare_images(ref, cand, max_diff_ratio="0.01")
    with pytest.raises(AssertionError):
        visual.compare_images(ref, cand, max_diff_pixels="10")


def test_web_keywords_pass_budget_through(tmp_path):
    import shutil

    from DocTest.WebVisualTest import WebVisualTest

    ref, cand = _pair(tmp_path, pixels=1500)  # 1500 changed pixels
    sources = {0: ref, 1: cand, 2: cand}  # per stable attempt: create → diff → diff

    class BudgetAdapter:
        library_name = "Fake"
        captures = 0

        def capture_page(self, path, full_page=True):
            shutil.copyfile(sources[min(self.captures // 2, 2)], path)
            self.captures += 1
            return path

        def describe(self):
            return {"library": "Fake"}

    lib = WebVisualTest(baseline_directory=str(tmp_path / "b"), retry_timeout="0")
    lib._adapter = BudgetAdapter()
    lib._robot_variable = lambda variable, default: str(tmp_path)
    lib.compare_page_to_baseline("home")
    with pytest.raises(AssertionError):
        lib.compare_page_to_baseline("home")
    # RF-style string kwarg travels through the web keyword into the engine
    lib.compare_page_to_baseline("home", max_diff_pixels="2000")


# -- anti-aliasing tolerance --------------------------------------------------

def _text_pair(tmp_path, text_b="Hello World", blur_b=True):
    """Same text rendered twice; candidate optionally blurred (AA-style noise)."""
    def render(text, blur):
        image = np.full((300, 600, 3), 255, dtype=np.uint8)
        cv2.putText(image, text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
        if blur:
            image = cv2.GaussianBlur(image, (3, 3), 0.8)
        return image
    ref_path, cand_path = tmp_path / "aa_ref.png", tmp_path / "aa_cand.png"
    cv2.imwrite(str(ref_path), render("Hello World", False))
    cv2.imwrite(str(cand_path), render(text_b, blur_b))
    return str(ref_path), str(cand_path)


def test_antialiasing_noise_passes_with_ignore_antialiasing(visual, tmp_path):
    ref, cand = _text_pair(tmp_path)  # same text, different edge rendering
    with pytest.raises(AssertionError):
        visual.compare_images(ref, cand)
    visual.compare_images(ref, cand, ignore_antialiasing=True, max_diff_pixels=30)


def test_solid_change_fails_despite_ignore_antialiasing(visual, tmp_path):
    ref, cand = _pair(tmp_path, pixels=1500)  # solid block, no edges in reference
    with pytest.raises(AssertionError):
        visual.compare_images(ref, cand, ignore_antialiasing=True)


def test_text_change_fails_despite_ignore_antialiasing(visual, tmp_path):
    ref, cand = _text_pair(tmp_path, text_b="Hollo Warld", blur_b=False)
    with pytest.raises(AssertionError):
        visual.compare_images(ref, cand, ignore_antialiasing=True)


def test_ignore_antialiasing_composes_with_ratio_budget(visual, tmp_path):
    ref, cand = _text_pair(tmp_path)
    visual.compare_images(ref, cand, ignore_antialiasing=True, max_diff_ratio=0.001)


def test_ignore_antialiasing_accepts_rf_string(visual, tmp_path):
    ref, cand = _text_pair(tmp_path)
    visual.compare_images(ref, cand, ignore_antialiasing="True", max_diff_pixels="30")


def test_classifier_counts_directly():
    from DocTest.VisualTest import count_real_difference_pixels

    base = np.full((300, 600), 255, np.uint8)
    cv2.putText(base, "Hello World", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
    changed = base.copy()
    cv2.rectangle(changed, (30, 150), (200, 220), 100, -1)
    differing, antialiased, real = count_real_difference_pixels(base, changed, 20)
    assert real == differing and antialiased == 0  # solid block is never AA
