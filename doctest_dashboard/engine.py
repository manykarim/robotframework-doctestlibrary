"""Embedded comparison engine.

The dashboard imports the DocTest library directly (verified to work
outside a Robot Framework run) for two features:

- mask preview: resolve mask definitions to pixel boxes for a page
- recompare: re-run a stored comparison with adjusted masks/settings into
  a scratch directory, never touching the original run artifacts

Comparisons are CPU-bound (OpenCV/OCR), so jobs run in a
``ProcessPoolExecutor`` with per-job timeouts; identical requests are
served from an in-memory cache.
"""

import hashlib
import json
import logging
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

JOB_TIMEOUT_SECONDS = 120
MAX_WORKERS = 2


# --- worker functions (run in subprocesses, must stay module-level) ----------

def _mask_preview_job(file_path: str, page: int, masks: Any, dpi: Optional[int],
                      ocr_engine: Optional[str], force_ocr: bool) -> Dict[str, Any]:
    from DocTest.DocumentRepresentation import DocumentRepresentation

    kwargs: Dict[str, Any] = {"ignore_area": masks}
    if dpi:
        kwargs["dpi"] = dpi
    if ocr_engine:
        kwargs["ocr_engine"] = ocr_engine
    if force_ocr:
        kwargs["force_ocr"] = True
    document = DocumentRepresentation(file_path, **kwargs)
    try:
        pages = document.pages
        if page < 1 or page > len(pages):
            raise ValueError(f"Page {page} out of range (document has {len(pages)} pages)")
        target = pages[page - 1]
        return {
            "page": page,
            "dpi": target.dpi,
            "page_count": len(pages),
            "resolved_areas": list(target.pixel_ignore_areas),
            "image_size": {"height": target.image.shape[0], "width": target.image.shape[1]},
        }
    finally:
        document.close()


def _page_image_job(file_path: str, page: int, dpi: Optional[int],
                    target_png: str) -> Dict[str, Any]:
    import cv2

    from DocTest.DocumentRepresentation import DocumentRepresentation

    kwargs: Dict[str, Any] = {}
    if dpi:
        kwargs["dpi"] = dpi
    document = DocumentRepresentation(file_path, **kwargs)
    try:
        pages = document.pages
        if page < 1 or page > len(pages):
            raise ValueError(f"Page {page} out of range (document has {len(pages)} pages)")
        target = pages[page - 1]
        cv2.imwrite(target_png, target.image)
        return {
            "page": page,
            "page_count": len(pages),
            "dpi": target.dpi,
            "image_size": {"height": target.image.shape[0], "width": target.image.shape[1]},
        }
    finally:
        document.close()


def _recompare_job(reference: str, candidate: str, masks: Any,
                   settings: Dict[str, Any], scratch_dir: str) -> Dict[str, Any]:
    os.chdir(scratch_dir)
    from DocTest.VisualTest import VisualTest

    visual_tester = VisualTest(result_json=True)
    kwargs: Dict[str, Any] = {}
    if masks is not None:
        kwargs["mask"] = masks
    for key in ("threshold", "move_tolerance", "DPI", "force_ocr", "blur",
                "block_based_ssim", "check_text_content"):
        if settings.get(key) is not None:
            kwargs[key] = settings[key]
    status = "PASS"
    try:
        visual_tester.compare_images(reference, candidate, **kwargs)
    except AssertionError:
        status = "FAIL"
    sidecars = sorted(Path(scratch_dir).glob("doctest_results/*.json"))
    if not sidecars:
        raise RuntimeError("Comparison produced no sidecar")
    with open(sidecars[-1], encoding="utf-8") as file:
        sidecar = json.load(file)
    sidecar["status"] = status
    return sidecar


# --- service -----------------------------------------------------------------

class EngineService:
    def __init__(self, scratch_root: Path):
        self.scratch_root = Path(scratch_root)
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        self._pool: Optional[ProcessPoolExecutor] = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.capabilities = self._check_capabilities()

    @staticmethod
    def _check_capabilities() -> Dict[str, Any]:
        try:
            from DocTest.CapabilityCheck import check_all_capabilities

            return check_all_capabilities()
        except Exception as error:  # pragma: no cover - defensive
            LOG.warning("Capability check failed: %s", error)
            return {}

    @property
    def ocr_available(self) -> bool:
        tesseract = self.capabilities.get("tesseract", {})
        return bool(tesseract.get("available", False))

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            # spawn: the server process is multi-threaded, fork is unsafe
            self._pool = ProcessPoolExecutor(
                max_workers=MAX_WORKERS,
                mp_context=multiprocessing.get_context("spawn"))
        return self._pool

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    @staticmethod
    def _cache_key(*parts: Any) -> str:
        digest = hashlib.sha256()
        for part in parts:
            digest.update(json.dumps(part, sort_keys=True, default=str).encode())
        return digest.hexdigest()

    @staticmethod
    def _file_fingerprint(path: str) -> Dict[str, Any]:
        stat = os.stat(path)
        return {"path": path, "size": stat.st_size, "mtime": stat.st_mtime_ns}

    def mask_preview(self, file_path: str, page: int, masks: Any,
                     dpi: Optional[int] = None, ocr_engine: Optional[str] = None,
                     force_ocr: bool = False) -> Dict[str, Any]:
        key = self._cache_key("preview", self._file_fingerprint(file_path),
                              page, masks, dpi, ocr_engine, force_ocr)
        if key in self._cache:
            return {**self._cache[key], "cached": True}
        future = self._ensure_pool().submit(
            _mask_preview_job, file_path, page, masks, dpi, ocr_engine, force_ocr)
        result = future.result(timeout=JOB_TIMEOUT_SECONDS)
        self._cache[key] = result
        return {**result, "cached": False}

    def page_image(self, file_path: str, page: int, dpi: Optional[int] = None) -> Dict[str, Any]:
        """Render a document/image page to a PNG in the scratch area."""
        key = self._cache_key("page-image", self._file_fingerprint(file_path), page, dpi)
        if key in self._cache:
            return {**self._cache[key], "cached": True}
        scratch_dir = tempfile.mkdtemp(prefix="page_", dir=self.scratch_root)
        target_png = str(Path(scratch_dir) / f"page_{page}.png")
        future = self._ensure_pool().submit(
            _page_image_job, file_path, page, dpi, target_png)
        info = future.result(timeout=JOB_TIMEOUT_SECONDS)
        result = {**info, "png_path": target_png}
        self._cache[key] = result
        return {**result, "cached": False}

    def recompare(self, reference: str, candidate: str, masks: Any,
                  settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        settings = settings or {}
        key = self._cache_key("recompare", self._file_fingerprint(reference),
                              self._file_fingerprint(candidate), masks, settings)
        if key in self._cache:
            return {**self._cache[key], "cached": True}
        scratch_dir = tempfile.mkdtemp(prefix="recompare_", dir=self.scratch_root)
        future = self._ensure_pool().submit(
            _recompare_job, reference, candidate, masks, settings, scratch_dir)
        try:
            sidecar = future.result(timeout=JOB_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            raise TimeoutError(f"Recompare timed out after {JOB_TIMEOUT_SECONDS}s")
        result = {"scratch_dir": scratch_dir, "result": sidecar}
        self._cache[key] = result
        return {**result, "cached": False}
