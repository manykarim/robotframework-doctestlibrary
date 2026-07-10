"""masks.json read/write endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from doctest_dashboard.masks import (
    MaskError,
    load_mask_file,
    normalize_masks,
    save_mask_file,
    validate_pattern_masks,
)
from doctest_dashboard.server.schemas import MaskSaveRequest
from doctest_dashboard.server.state import AppState, get_state, require_token

router = APIRouter(dependencies=[Depends(require_token)])


@router.get("/api/masks")
def get_masks(file: str, state: AppState = Depends(get_state)):
    path = Path(file)
    if not state.config.is_within_roots(path):
        raise HTTPException(status_code=403, detail="Mask file outside configured roots")
    try:
        return {"file": str(path), "masks": normalize_masks(load_mask_file(path))}
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except MaskError as error:
        raise HTTPException(status_code=422, detail=str(error))


@router.put("/api/masks")
def put_masks(request: MaskSaveRequest, state: AppState = Depends(get_state)):
    path = Path(request.file)  # NOSONAR: paths are confined to configured roots (config.is_within_roots, symlink-safe resolve) and covered by traversal tests
    probe = path if path.exists() else path.parent
    if not state.config.is_within_roots(probe):
        raise HTTPException(status_code=403, detail="Mask file outside configured roots")
    try:
        masks = normalize_masks(request.masks)
        validate_pattern_masks(masks)
    except MaskError as error:
        raise HTTPException(status_code=422, detail=str(error))
    file_hash = save_mask_file(path, masks)
    state.db.execute(
        "INSERT INTO mask_files (path, last_seen_hash, updated_at) VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(path) DO UPDATE SET last_seen_hash = excluded.last_seen_hash, "
        "updated_at = excluded.updated_at",
        (str(path), file_hash))
    return {"file": str(path), "masks": masks, "sha256": file_hash}
