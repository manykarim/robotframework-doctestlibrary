"""Shared pytest fixtures for unit tests."""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest


@pytest.fixture
def testdata_dir() -> Path:
    """Return the absolute path to the shared testdata directory."""

    return Path(__file__).resolve().parent / "testdata"

SAMPLE_IMAGE = (Path(__file__).resolve().parent / "testdata" / "text_big.png").resolve()
HAS_IMAGE_SUPPORT = SAMPLE_IMAGE.exists() and cv2.imread(str(SAMPLE_IMAGE)) is not None


@pytest.fixture(scope="session")
def require_image_samples() -> None:
    """Skip tests that rely on packaged images when codecs are unavailable."""

    if not HAS_IMAGE_SUPPORT:
        pytest.skip("Image codecs for test assets are unavailable in this environment")
