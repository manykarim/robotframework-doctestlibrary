"""EAST text detection: model provisioning and region recognition contract."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from DocTest import Ocr
from DocTest.DocumentRepresentation import DocumentRepresentation

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bare_extractor(model_path):
    """An extractor that skips loading the network in __init__."""
    extractor = Ocr.EastTextExtractor.__new__(Ocr.EastTextExtractor)
    extractor.east = str(model_path)
    return extractor


# ---------------------------------------------------------------------------
# Model provisioning
# ---------------------------------------------------------------------------


def test_interrupted_download_leaves_no_model(tmp_path, monkeypatch):
    model = tmp_path / "model.pb"

    def broken_download(url, filename):
        Path(filename).write_bytes(b"partial")
        raise ConnectionError("connection reset")

    monkeypatch.setattr(Ocr.urllib.request, "urlretrieve", broken_download)
    with pytest.raises(ConnectionError):
        _bare_extractor(model)._get_east()
    assert not model.exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_model_is_not_downloaded_again(tmp_path, monkeypatch):
    model = tmp_path / "model.pb"
    model.write_bytes(b"complete")

    def unexpected_download(url, filename):
        raise AssertionError("must not download when the model exists")

    monkeypatch.setattr(Ocr.urllib.request, "urlretrieve", unexpected_download)
    _bare_extractor(model)._get_east()
    assert model.read_bytes() == b"complete"


def test_concurrent_provisioning_never_exposes_partial_model(tmp_path, monkeypatch):
    model = tmp_path / "model.pb"
    payload = b"x" * 200_000
    observed = []

    def slow_download(url, filename):
        with open(filename, "wb") as handle:
            for start in range(0, len(payload), 20_000):
                handle.write(payload[start:start + 20_000])
                handle.flush()
                if model.exists():
                    observed.append(model.stat().st_size)
                time.sleep(0.005)

    monkeypatch.setattr(Ocr.urllib.request, "urlretrieve", slow_download)
    barrier = threading.Barrier(2)
    errors = []

    def provision():
        try:
            barrier.wait()
            _bare_extractor(model)._get_east()
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=provision) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert model.read_bytes() == payload
    # whenever the final path existed, it already held the complete model
    assert all(size == len(payload) for size in observed)
    assert [p.name for p in tmp_path.iterdir()] == ["model.pb"]


def test_download_works_in_a_fresh_interpreter(tmp_path):
    """The download must not rely on another module importing urllib.request."""
    model = tmp_path / "model.pb"
    script = (
        "import sys\n"
        "from DocTest import Ocr\n"
        "Ocr.urllib.request.urlretrieve = lambda url, filename: open(filename, 'wb').write(b'ok')\n"
        "extractor = Ocr.EastTextExtractor.__new__(Ocr.EastTextExtractor)\n"
        f"extractor.east = {str(model)!r}\n"
        "extractor._get_east()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, errors="replace"
    )
    assert result.returncode == 0, result.stderr
    assert model.read_bytes() == b"ok"


# ---------------------------------------------------------------------------
# Region recognition (EAST network and Tesseract stubbed)
# ---------------------------------------------------------------------------


def _ocr_result(rows):
    """Build a pytesseract DICT from (text, conf) rows with dummy geometry."""
    return {
        "text": [text for text, _ in rows],
        "conf": [conf for _, conf in rows],
        "left": [1] * len(rows),
        "top": [1] * len(rows),
        "width": [5] * len(rows),
        "height": [5] * len(rows),
    }


@pytest.fixture
def stubbed_extractor(monkeypatch):
    """Extractor whose EAST output is the rectangles set on ``.rects``.

    The image and the EAST input are both 480x480, so rectangle coordinates
    map 1:1 to image pixels.
    """
    extractor = Ocr.EastTextExtractor.__new__(Ocr.EastTextExtractor)
    extractor.rects = []
    monkeypatch.setattr(extractor, "_compute_scores_geometry", lambda image, w, h: (None, None), raising=False)
    monkeypatch.setattr(Ocr, "box_extractor", lambda scores, geometry, conf: (extractor.rects, [1.0] * len(extractor.rects)))
    return extractor


def test_structural_rows_are_not_returned(stubbed_extractor, monkeypatch):
    stubbed_extractor.rects = [(100, 100, 200, 120)]
    rows = [("", -1), ("", -1), ("", -1), ("", -1), ("Invoice", 91), ("  ", 50), ("42", 88)]
    monkeypatch.setattr(stubbed_extractor, "_recognize_roi", lambda roi, psm=7: _ocr_result(rows), raising=False)

    result = stubbed_extractor.get_image_text(np.zeros((480, 480, 3), dtype=np.uint8))

    assert result["text"] == ["Invoice", "42"]


def test_multi_line_region_is_read_as_block(stubbed_extractor, monkeypatch):
    # three stacked, touching 12px boxes merge into one 36px region
    stubbed_extractor.rects = [(100, 100, 300, 112), (100, 112, 300, 124), (100, 124, 300, 136)]
    modes = []

    def recognize(roi, psm=7):
        modes.append(psm)
        return _ocr_result([("line", 90)])

    monkeypatch.setattr(stubbed_extractor, "_recognize_roi", recognize, raising=False)
    stubbed_extractor.get_image_text(np.zeros((480, 480, 3), dtype=np.uint8))

    assert modes == [6]


def test_single_line_region_is_read_as_line(stubbed_extractor, monkeypatch):
    stubbed_extractor.rects = [(100, 100, 300, 112), (150, 101, 320, 113)]
    modes = []

    def recognize(roi, psm=7):
        modes.append(psm)
        return _ocr_result([("line", 90)])

    monkeypatch.setattr(stubbed_extractor, "_recognize_roi", recognize, raising=False)
    stubbed_extractor.get_image_text(np.zeros((480, 480, 3), dtype=np.uint8))

    assert modes == [7]


def test_empty_region_is_retried_once_with_tight_crop(stubbed_extractor, monkeypatch):
    stubbed_extractor.rects = [(200, 200, 260, 212)]
    crops = []

    def recognize(roi, psm=7):
        crops.append(roi.shape[:2])
        if len(crops) == 1:
            return _ocr_result([("", -1), ("~", 30)])
        return _ocr_result([("", -1), ("SOUVENIR", 70)])

    monkeypatch.setattr(stubbed_extractor, "_recognize_roi", recognize, raising=False)
    result = stubbed_extractor.get_image_text(np.zeros((480, 480, 3), dtype=np.uint8))

    assert result["text"] == ["SOUVENIR"]
    assert len(crops) == 2
    wide, tight = crops
    assert tight[1] < wide[1]


def test_readable_region_is_not_retried(stubbed_extractor, monkeypatch):
    stubbed_extractor.rects = [(200, 200, 260, 212)]
    calls = []

    def recognize(roi, psm=7):
        calls.append(psm)
        return _ocr_result([("01-Jan-2021", 92)])

    monkeypatch.setattr(stubbed_extractor, "_recognize_roi", recognize, raising=False)
    stubbed_extractor.get_image_text(np.zeros((480, 480, 3), dtype=np.uint8))

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Real images (EAST model + Tesseract)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("image", ["text_small.png", "text_big.png"])
def test_stacked_lines_are_all_recognized(testdata_dir, image):
    text = DocumentRepresentation(testdata_dir / image, ocr_engine="east").get_text()
    assert "ABCDEFGHI" in text
    assert "abcdefghi" in text
    assert "1234567890" in text


def test_small_label_in_busy_image_is_recognized(testdata_dir):
    text = DocumentRepresentation(testdata_dir / "Beach_date.png", ocr_engine="east").get_text()
    assert "01-Jan-2021" in text
    assert "123456789" in text
    assert "SOUVENIR" in text.upper()
