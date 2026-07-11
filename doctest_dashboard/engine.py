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
from collections import OrderedDict
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
MAX_CACHE_ENTRIES = 256


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


def _region_text_job(reference: str, candidate: str, page_no: int,
                     region: Dict[str, int], dpi: Optional[int],
                     force_ocr: bool) -> Dict[str, Any]:
    """Extract and compare the text inside one region of one page, using the
    library's own region comparison (exact engine parity with
    check_text_content). Region pixels are relative to the sidecar DPI."""
    from DocTest.DocumentRepresentation import DocumentRepresentation

    kwargs: Dict[str, Any] = {}
    if dpi:
        kwargs["dpi"] = dpi
    ref_doc = DocumentRepresentation(reference, **kwargs)
    cand_doc = DocumentRepresentation(candidate, **kwargs)
    try:
        if page_no < 1 or page_no > len(ref_doc.pages) or page_no > len(cand_doc.pages):
            raise ValueError(f"Page {page_no} out of range")
        ref_page = ref_doc.pages[page_no - 1]
        cand_page = cand_doc.pages[page_no - 1]
        same, ref_text, cand_text = ref_page._compare_text_content_in_area_with(
            cand_page, region, force_ocr)
        return {
            "same": bool(same),
            "reference_text": ref_text or "",
            "candidate_text": cand_text or "",
        }
    finally:
        ref_doc.close()
        cand_doc.close()


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
    sidecars = sorted(
        f for f in Path(scratch_dir).glob("doctest_results/*.json")
        if f.name != "run.json")
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
        self._cache: 'OrderedDict[str, Dict[str, Any]]' = OrderedDict()
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

    def _cache_get(self, key: str):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def _cache_put(self, key: str, value) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > MAX_CACHE_ENTRIES:
            self._cache.popitem(last=False)

    @staticmethod
    def _cache_key(*parts: Any) -> str:
        digest = hashlib.sha256()
        for part in parts:
            digest.update(json.dumps(part, sort_keys=True, default=str).encode())
        return digest.hexdigest()

    @staticmethod
    def _file_fingerprint(path: str) -> Dict[str, Any]:
        stat = os.stat(path)  # NOSONAR: paths are confined to configured roots (config.is_within_roots, symlink-safe resolve) and covered by traversal tests
        return {"path": path, "size": stat.st_size, "mtime": stat.st_mtime_ns}

    def mask_preview(self, file_path: str, page: int, masks: Any,
                     dpi: Optional[int] = None, ocr_engine: Optional[str] = None,
                     force_ocr: bool = False) -> Dict[str, Any]:
        key = self._cache_key("preview", self._file_fingerprint(file_path),
                              page, masks, dpi, ocr_engine, force_ocr)
        cached = self._cache_get(key)
        if cached is not None:
            return {**cached, "cached": True}
        future = self._ensure_pool().submit(
            _mask_preview_job, file_path, page, masks, dpi, ocr_engine, force_ocr)
        result = future.result(timeout=JOB_TIMEOUT_SECONDS)
        self._cache_put(key, result)
        return {**result, "cached": False}

    def page_image(self, file_path: str, page: int, dpi: Optional[int] = None) -> Dict[str, Any]:
        """Render a document/image page to a PNG in the scratch area."""
        key = self._cache_key("page-image", self._file_fingerprint(file_path), page, dpi)
        cached = self._cache_get(key)
        if cached is not None:
            return {**cached, "cached": True}
        scratch_dir = tempfile.mkdtemp(prefix="page_", dir=self.scratch_root)
        target_png = str(Path(scratch_dir) / f"page_{page}.png")
        future = self._ensure_pool().submit(
            _page_image_job, file_path, page, dpi, target_png)
        info = future.result(timeout=JOB_TIMEOUT_SECONDS)
        result = {**info, "png_path": target_png}
        self._cache_put(key, result)
        return {**result, "cached": False}

    def region_text(self, reference: str, candidate: str, page_no: int,
                    region: Dict[str, int], dpi: Optional[int] = None,
                    force_ocr: bool = False) -> Dict[str, Any]:
        key = self._cache_key("region-text", self._file_fingerprint(reference),
                              self._file_fingerprint(candidate), page_no, region,
                              dpi, force_ocr)
        cached = self._cache_get(key)
        if cached is not None:
            return {**cached, "cached": True}
        future = self._ensure_pool().submit(
            _region_text_job, reference, candidate, page_no, region, dpi, force_ocr)
        result = future.result(timeout=JOB_TIMEOUT_SECONDS)
        self._cache_put(key, result)
        return {**result, "cached": False}

    def recompare(self, reference: str, candidate: str, masks: Any,
                  settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        settings = settings or {}
        key = self._cache_key("recompare", self._file_fingerprint(reference),
                              self._file_fingerprint(candidate), masks, settings)
        cached = self._cache_get(key)
        if cached is not None:
            return {**cached, "cached": True}
        scratch_dir = tempfile.mkdtemp(prefix="recompare_", dir=self.scratch_root)
        future = self._ensure_pool().submit(
            _recompare_job, reference, candidate, masks, settings, scratch_dir)
        try:
            sidecar = future.result(timeout=JOB_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            raise TimeoutError(f"Recompare timed out after {JOB_TIMEOUT_SECONDS}s")
        result = {"scratch_dir": scratch_dir, "result": sidecar}
        self._cache_put(key, result)
        return {**result, "cached": False}
