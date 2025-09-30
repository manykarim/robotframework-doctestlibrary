"""Tests for the OCRS Python adapter."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from DocTest import ocrs_adapter


def test_extract_allowed_chars_parses_whitelist():
    config = "--psm 6 tessedit_char_whitelist=ABC123"
    assert ocrs_adapter.extract_allowed_chars(config) == "ABC123"


def test_extract_allowed_chars_without_whitelist():
    assert ocrs_adapter.extract_allowed_chars("--psm 6") is None


def test_models_available_prefers_environment(tmp_path, monkeypatch):
    detection = tmp_path / "text-detection.rten"
    recognition = tmp_path / "text-recognition.rten"
    detection.write_bytes(b"det")
    recognition.write_bytes(b"rec")
    monkeypatch.setenv("OCRS_MODEL_DIR", str(tmp_path))
    assert ocrs_adapter.models_available() is True


def test_run_ocr_invokes_extension(monkeypatch):
    captured = {}

    def fake_run_ocr(image: np.ndarray, allowed_chars, beam_search):
        captured["dtype"] = image.dtype
        captured["shape"] = image.shape
        captured["allowed_chars"] = allowed_chars
        captured["beam_search"] = beam_search
        return {
            "text": ["sample"],
            "left": [0],
            "top": [0],
            "width": [image.shape[1]],
            "height": [image.shape[0]],
            "conf": ["100"],
        }

    monkeypatch.setattr(
        ocrs_adapter,
        "_ocrs_extension",
        SimpleNamespace(run_ocr=fake_run_ocr),
    )

    image = np.zeros((4, 4, 4), dtype=np.uint16)
    result = ocrs_adapter.run_ocr(image, allowed_chars="ABC", beam_search=True)

    assert captured["dtype"] == np.uint8
    assert captured["shape"] == (4, 4, 3)
    assert captured["allowed_chars"] == "ABC"
    assert captured["beam_search"] is True
    assert result["text"] == ["sample"]


def test_run_ocr_raises_without_extension(monkeypatch):
    monkeypatch.setattr(ocrs_adapter, "_ocrs_extension", None)
    with pytest.raises(ocrs_adapter.OcrsError):
        ocrs_adapter.run_ocr(np.zeros((2, 2, 3), dtype=np.uint8))


def test_ensure_ready_false_without_extension(monkeypatch):
    monkeypatch.setattr(ocrs_adapter, "_ocrs_extension", None)
    assert ocrs_adapter.ensure_ready() is False
