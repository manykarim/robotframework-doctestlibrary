"""Uploads from the browser: single workspace files and whole result folders."""

import re
import uuid
from pathlib import Path, PurePosixPath
from typing import List

import anyio
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from doctest_dashboard.ingest import ingest_output_xml
from doctest_dashboard.server.state import AppState, get_state, require_token

router = APIRouter(dependencies=[Depends(require_token)])

UPLOAD_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".pdf", ".ps", ".pcl",
    ".tif", ".tiff", ".bmp", ".gif", ".json",
}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
RESULTS_UPLOAD_EXTENSIONS = UPLOAD_EXTENSIONS | {".xml", ".txt", ".log"}
MAX_RESULTS_UPLOAD_BYTES = 500 * 1024 * 1024


@router.post("/api/upload")
async def upload(file: UploadFile, state: AppState = Depends(get_state)):
    """Store a file from the user's machine in the dashboard workspace.

    The workspace (``{data_dir}/uploads``) is a configured root, so the
    stored file is immediately usable in the mask editor and browsable
    in the file picker — no --root setup required.
    """
    original = Path(file.filename or "upload")
    if original.suffix.lower() not in UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{original.suffix}'. "
                   f"Allowed: {', '.join(sorted(UPLOAD_EXTENSIONS))}")
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", original.stem)[:80] or "upload"
    target_dir = state.uploads_dir / uuid.uuid4().hex[:8]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{safe_stem}{original.suffix.lower()}"
    size = 0
    try:
        async with await anyio.open_file(target, "wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit")
                await out.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    return {"path": str(target), "name": target.name, "size": size}


@router.post("/api/upload-results")
async def upload_results(files: List[UploadFile], state: AppState = Depends(get_state)):
    """Ingest a Robot Framework results folder uploaded from the browser.

    The folder picker (webkitdirectory) sends every file with its path
    relative to the chosen folder; the tree is stored in the workspace
    with that structure intact — which is what output.xml, screenshot
    references and sidecars rely on — then ingested in place.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files in upload")
    target_root = state.uploads_dir / f"run_{uuid.uuid4().hex[:8]}"
    stored, skipped, total = 0, 0, 0
    output_xmls = []
    for file in files:
        relative = PurePosixPath((file.filename or "").replace("\\", "/"))
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.lower() not in RESULTS_UPLOAD_EXTENSIONS
        ):
            skipped += 1
            continue
        target = target_root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with await anyio.open_file(target, "wb") as out:
            while chunk := await file.read(1 << 20):
                total += len(chunk)
                if total > MAX_RESULTS_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the "
                               f"{MAX_RESULTS_UPLOAD_BYTES // (1024 * 1024)} MB limit")
                await out.write(chunk)
        stored += 1
        if relative.name == "output.xml":
            output_xmls.append(target)
    if not output_xmls:
        raise HTTPException(
            status_code=422,
            detail="The selected folder contains no output.xml — pick the "
                   "Robot Framework output directory of a run")
    summaries = [
        ingest_output_xml(state.db, state.config, output_xml)
        for output_xml in sorted(output_xmls, key=lambda p: len(p.parts))
    ]
    return {"stored": stored, "skipped": skipped, "runs": summaries}
