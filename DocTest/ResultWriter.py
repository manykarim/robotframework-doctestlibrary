"""Machine-readable comparison result sidecars (schema v1).

When ``result_json`` is enabled on ``VisualTest`` or ``PdfTest``, every
comparison writes a JSON sidecar to ``{OUTPUT_DIR}/doctest_results/`` and
logs a single ``DOCTEST_RESULT: <relative path>`` message. The sidecar is
the machine-readable counterpart of the human-oriented log and is consumed
by external tooling such as the doctest-dashboard.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

SCHEMA_VERSION = 1
RESULTS_DIR_NAME = "doctest_results"
RESULT_LOG_PREFIX = "DOCTEST_RESULT:"


def _json_safe(value: Any) -> Any:
    """Fallback serializer for values json cannot encode natively."""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):  # numpy scalars
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


class ComparisonResultWriter:
    """Collects per-page data during one comparison and writes the sidecar."""

    def __init__(
        self,
        output_directory,
        keyword: str,
        library: str,
        pabot_index: Optional[str] = None,
    ):
        self.output_directory = Path(output_directory)
        self.keyword = keyword
        self.library = library
        uid = str(uuid.uuid1())
        if pabot_index is not None:
            uid = f"{pabot_index}-{uid}"
        self.uid = uid
        self.results_dir = self.output_directory / RESULTS_DIR_NAME
        self.image_dir = self.results_dir / self.uid
        self.started = datetime.now().isoformat(timespec="seconds")
        self._t0 = time.monotonic()
        self.pages: List[Dict[str, Any]] = []

    def save_page_image(self, image, page_number, kind: str) -> Optional[str]:
        """Save a page rendering as lossless PNG.

        Returns the path relative to the output directory, or None if the
        image could not be written.
        """
        if image is None:
            return None
        self.image_dir.mkdir(parents=True, exist_ok=True)
        rel = Path(RESULTS_DIR_NAME) / self.uid / f"page_{page_number}_{kind}.png"
        if cv2.imwrite(str(self.output_directory / rel), image):
            return rel.as_posix()
        return None

    def add_page(
        self,
        page_number,
        status: str,
        score: Optional[float] = None,
        threshold: Optional[float] = None,
        diff_regions: Optional[List[Dict[str, int]]] = None,
        images: Optional[Dict[str, str]] = None,
        notes: Optional[List[str]] = None,
        resolved_masks: Optional[List[Dict[str, int]]] = None,
    ) -> None:
        self.pages.append(
            {
                "page": int(page_number) if str(page_number).isdigit() else page_number,
                "status": status,
                "score": score,
                "threshold": threshold,
                "diff_regions": diff_regions or [],
                "images": images or {},
                "notes": notes or [],
                "resolved_masks": resolved_masks or [],
            }
        )

    def write(
        self,
        status: str,
        reference: Dict[str, Any],
        candidate: Dict[str, Any],
        settings: Dict[str, Any],
        masks: Optional[Dict[str, Any]] = None,
        llm: Optional[Dict[str, Any]] = None,
        notes: Optional[List[str]] = None,
    ) -> str:
        """Write the sidecar JSON and return its path relative to OUTPUT_DIR."""
        result = {
            "schema_version": SCHEMA_VERSION,
            "keyword": self.keyword,
            "library": self.library,
            "status": status,
            "reference": reference,
            "candidate": candidate,
            "settings": settings,
            "masks": masks or {},
            "pages": self.pages,
            "llm": llm,
            "notes": notes or [],
            "timing": {
                "started": self.started,
                "elapsed_ms": int((time.monotonic() - self._t0) * 1000),
            },
        }
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / f"{self.uid}.json"
        with open(path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, default=_json_safe)
        return (Path(RESULTS_DIR_NAME) / f"{self.uid}.json").as_posix()
