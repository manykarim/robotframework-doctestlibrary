"""Embedded comparison engine endpoints: preview, page images, recompare."""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from doctest_dashboard.ingest import asset_token
from doctest_dashboard.masks import MaskError, validate_pattern_masks
from doctest_dashboard.server.schemas import (
    MaskPreviewRequest,
    RecompareBatchRequest,
    RecompareRequest,
    RegionTextRequest,
)
from doctest_dashboard.server.state import AppState, get_state, require_token

router = APIRouter(dependencies=[Depends(require_token)])

PATTERN_TYPES = {"pattern", "line_pattern", "word_pattern"}


def _as_mask_list(masks) -> list:
    if masks is None:
        return []
    if isinstance(masks, dict):
        return [masks]
    if isinstance(masks, list):
        return masks
    return []


@router.get("/api/capabilities")
def capabilities(state: AppState = Depends(get_state)):
    return {"capabilities": state.engine.capabilities,
            "ocr_available": state.engine.ocr_available}


@router.post("/api/mask-preview")
def mask_preview(request: MaskPreviewRequest, state: AppState = Depends(get_state)):
    path = Path(request.file)
    if not state.config.is_within_roots(path):
        raise HTTPException(status_code=403, detail="File outside configured roots")
    try:
        validate_pattern_masks(_as_mask_list(request.masks))
    except MaskError as error:
        raise HTTPException(status_code=422, detail=str(error))
    mask_types = {m.get("type") for m in _as_mask_list(request.masks)}
    if (mask_types & PATTERN_TYPES) and path.suffix.lower() not in (
        ".pdf",) and not state.engine.ocr_available:
        raise HTTPException(
            status_code=409,
            detail="Pattern mask preview requires OCR (tesseract), "
                   "which is not available in this environment")
    try:
        return state.engine.mask_preview(
            str(path), request.page, request.masks,
            dpi=request.dpi, ocr_engine=request.ocr_engine,
            force_ocr=request.force_ocr)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error))


@router.get("/api/page-image")
def page_image(file: str, page: int = 1, dpi: Optional[int] = None,
               state: AppState = Depends(get_state)):
    path = Path(file)
    if not state.config.is_within_roots(path):
        raise HTTPException(status_code=403, detail="File outside configured roots")
    try:
        info = state.engine.page_image(str(path), page, dpi)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    token = asset_token(info["png_path"])
    state.db.register_asset(token, info["png_path"])
    return {
        "page": info["page"],
        "page_count": info["page_count"],
        "dpi": info["dpi"],
        "image_size": info["image_size"],
        "image": token,
    }


def _run_recompare(state: AppState, comparison: dict, masks, settings) -> dict:
    try:
        validate_pattern_masks(_as_mask_list(masks))
    except MaskError as error:
        raise HTTPException(status_code=422, detail=str(error))
    reference = comparison["reference_path"]
    candidate = comparison["candidate_path"]
    if not reference or not candidate:
        raise HTTPException(
            status_code=409,
            detail="Comparison has no recorded paths (degraded record)")
    for path in (reference, candidate):
        if not state.config.is_within_roots(Path(path)):
            raise HTTPException(status_code=403, detail="Path outside configured roots")
    try:
        outcome = state.engine.recompare(reference, candidate, masks, settings)
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error))
    sidecar = outcome["result"]
    scratch = Path(outcome["scratch_dir"])
    for page in sidecar.get("pages", []):
        tokens = {}
        for kind, rel in page.get("images", {}).items():
            absolute = (scratch / rel).resolve()
            token = asset_token(str(absolute))
            state.db.register_asset(token, str(absolute))
            tokens[kind] = token
        page["images"] = tokens
    return {
        "comparison_id": comparison["id"],
        "status": sidecar["status"],
        "pages": sidecar.get("pages", []),
        "cached": outcome.get("cached", False),
    }


@router.post("/api/comparisons/{comparison_id}/region-text")
def region_text(comparison_id: int, request: RegionTextRequest,
                state: AppState = Depends(get_state)):
    """Explain a diff region: extract reference vs candidate text inside it."""
    comparison = state.db.query_one(
        "SELECT * FROM comparisons WHERE id = ?", (comparison_id,))
    if not comparison:
        raise HTTPException(status_code=404, detail="Comparison not found")
    reference = comparison["reference_path"]
    candidate = comparison["candidate_path"]
    if not reference or not candidate:
        raise HTTPException(
            status_code=409,
            detail="Comparison has no recorded paths (degraded record)")
    for path in (reference, candidate):
        if not state.config.is_within_roots(Path(path)):
            raise HTTPException(status_code=403, detail="Path outside configured roots")
    dpi = None
    if comparison["sidecar_json"]:
        dpi = (json.loads(comparison["sidecar_json"])
               .get("reference", {}).get("dpi"))
    region = {
        "x": request.region.x, "y": request.region.y,
        "width": request.region.width, "height": request.region.height,
    }
    try:
        return state.engine.region_text(
            reference, candidate, request.page_no, region,
            dpi=dpi, force_ocr=request.force_ocr)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error))


@router.post("/api/recompare")
def recompare(request: RecompareRequest, state: AppState = Depends(get_state)):
    comparison = state.db.query_one(
        "SELECT * FROM comparisons WHERE id = ?", (request.comparison_id,))
    if not comparison:
        raise HTTPException(status_code=404, detail="Comparison not found")
    return _run_recompare(state, comparison, request.masks, request.settings)


@router.post("/api/recompare-batch")
def recompare_batch(request: RecompareBatchRequest, state: AppState = Depends(get_state)):
    db = state.db
    if request.comparison_ids:
        comparisons = [
            row for cid in request.comparison_ids
            if (row := db.query_one("SELECT * FROM comparisons WHERE id = ?", (cid,)))
        ]
    elif request.masks_file:
        comparisons = [
            row for row in db.query(
                "SELECT * FROM comparisons WHERE sidecar_json IS NOT NULL")
            if json.loads(row["sidecar_json"]).get("masks", {}).get("placeholder_file")
            == request.masks_file
        ]
    else:
        raise HTTPException(
            status_code=400, detail="Provide comparison_ids or masks_file")
    results = []
    for comparison in comparisons:
        try:
            results.append(_run_recompare(state, comparison, request.masks, request.settings))
        except HTTPException as error:
            results.append({
                "comparison_id": comparison["id"],
                "status": "ERROR",
                "error": error.detail,
            })
    return {"results": results}
