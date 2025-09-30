"""Adapter for invoking the OCRS Rust extension."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from DocTest import _ocrs as _ocrs_extension  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - exercised in tests via monkeypatching
    _ocrs_extension = None

MISSING_EXTENSION_MESSAGE = (
    "OCRS extension is not available. Run `maturin develop` during development "
    "or install the packaged wheel to enable OCR features."
)

LOGGER = logging.getLogger(__name__)

MODEL_FILENAMES = {
    "detection": "text-detection.rten",
    "recognition": "text-recognition.rten",
}


class OcrsError(RuntimeError):
    """Raised when OCRS cannot process an image."""


@dataclass
class OcrResult:
    text: List[str]
    left: List[int]
    top: List[int]
    width: List[int]
    height: List[int]
    conf: List[str]

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "text": self.text,
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "conf": self.conf,
        }


def _coerce_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return np.ascontiguousarray(image)


def extract_allowed_chars(tesseract_config: Optional[str]) -> Optional[str]:
    if not tesseract_config:
        return None
    match = re.search(r"tessedit_char_whitelist\s*=\s*([\w]+)", tesseract_config)
    if match:
        return match.group(1)
    return None


def _require_extension() -> None:
    if _ocrs_extension is None:
        raise OcrsError(MISSING_EXTENSION_MESSAGE)


def run_ocr(
    image: np.ndarray,
    *,
    allowed_chars: Optional[str] = None,
    beam_search: bool = False,
) -> Dict[str, List[str]]:
    _require_extension()
    prepared = _coerce_image(image)
    try:
        result = _ocrs_extension.run_ocr(prepared, allowed_chars, beam_search)
    except RuntimeError as exc:
        raise OcrsError(str(exc)) from exc
    return {key: list(value) for key, value in result.items()}


def models_available() -> bool:
    detection_env = os.getenv("OCRS_DETECTION_MODEL")
    recognition_env = os.getenv("OCRS_RECOGNITION_MODEL")
    if detection_env and recognition_env:
        return Path(detection_env).exists() and Path(recognition_env).exists()

    candidate_dirs = []
    model_dir = os.getenv("OCRS_MODEL_DIR")
    if model_dir:
        candidate_dirs.append(Path(model_dir))
    candidate_dirs.append(Path.home() / ".cache" / "ocrs")

    for directory in candidate_dirs:
        if all((directory / MODEL_FILENAMES[key]).exists() for key in MODEL_FILENAMES):
            return True
    return False


def ensure_ready() -> bool:
    if _ocrs_extension is None:
        LOGGER.debug("OCRS extension is missing; skipping warm-up")
        return False
    if not models_available():
        return False
    try:
        dummy = np.zeros((2, 2, 3), dtype=np.uint8)
        _ocrs_extension.run_ocr(dummy, None, False)
    except RuntimeError as exc:
        LOGGER.debug("OCRS warm-up failed: %s", exc)
        return False
    return True
