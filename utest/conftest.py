"""Shared pytest fixtures for unit tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pytest
import cv2

from DocTest import ocrs_adapter


@dataclass
class _StubResult:
    text: List[str]
    width: int
    height: int

    def to_dict(self) -> Dict[str, List[str]]:
        count = len(self.text)
        return {
            "text": self.text,
            "left": [0] * count,
            "top": [0] * count,
            "width": [self.width] * count,
            "height": [self.height] * count,
            "conf": ["100"] * count,
        }


@pytest.fixture
def fake_ocrs(monkeypatch: pytest.MonkeyPatch) -> Callable[[Optional[str]], Dict[str, List[str]]]:
    """Provide a deterministic OCRS stub for tests that need OCR output.

    The fixture installs a lightweight stub that mimics the Rust extension so tests
    can focus on higher-level integration without requiring the actual OCR models.
    It returns a callable allowing tests to adjust the whitelist filtering logic.
    """

    base_tokens = [
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "abcdefghi",
        "ABCDEFGHI",
        "1234567890",
        "01-Jan-2021",
        "SOUVENIR",
    ]

    def _filter_tokens(allowed_chars: Optional[str]) -> Dict[str, List[str]]:
        if not allowed_chars:
            tokens = base_tokens
        else:
            whitelist = set(allowed_chars)
            tokens = ["".join(ch for ch in token if ch in whitelist) for token in base_tokens]
        height = width = 100
        return _StubResult(tokens, width, height).to_dict()

    class _StubExtension:
        @staticmethod
        def run_ocr(image: np.ndarray, allowed_chars: Optional[str], beam_search: bool):
            _ = image, beam_search
            return _filter_tokens(allowed_chars)

    monkeypatch.setattr(ocrs_adapter, "_ocrs_extension", _StubExtension())

    return _filter_tokens


@pytest.fixture
def testdata_dir() -> Path:
    """Return the absolute path to the shared testdata directory."""

    return Path(__file__).resolve().parent.parent / "testdata"


SAMPLE_IMAGE = (Path(__file__).resolve().parent.parent / "testdata" / "text_big.png").resolve()
HAS_IMAGE_SUPPORT = SAMPLE_IMAGE.exists() and cv2.imread(str(SAMPLE_IMAGE)) is not None


@pytest.fixture(scope="session")
def require_image_samples() -> None:
    """Skip tests that rely on packaged images when codecs are unavailable."""

    if not HAS_IMAGE_SUPPORT:
        pytest.skip("Image codecs for test assets are unavailable in this environment")
