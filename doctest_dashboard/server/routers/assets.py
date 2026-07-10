"""Token-based, root-confined asset serving."""

import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from doctest_dashboard.server.state import AppState, get_state, require_token

router = APIRouter(dependencies=[Depends(require_token)])


@router.get("/api/assets/{token}")
def get_asset(token: str, state: AppState = Depends(get_state)):
    path = state.db.resolve_asset(token)
    if not path:
        raise HTTPException(status_code=404, detail="Unknown asset")
    if not state.config.is_within_roots(Path(path)):
        raise HTTPException(status_code=403, detail="Asset outside configured roots")
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return FileResponse(
        path, media_type=media_type,
        headers={"Cache-Control": "private, max-age=86400"})
