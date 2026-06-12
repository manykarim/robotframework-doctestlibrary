"""FastAPI application factory."""

import json
import logging
import mimetypes

import anyio
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from doctest_dashboard import __version__
from doctest_dashboard.config import AppConfig
from doctest_dashboard.db import Database
from doctest_dashboard.ingest import ingest_output_xml
from doctest_dashboard.review import (
    ReviewError,
    accept_comparison,
    accept_page,
    build_bug_bundle,
    reject_comparison,
)

LOG = logging.getLogger(__name__)

# Capabilities of this backend build, advertised via /api/health. The UI
# checks them at load time and tells the user to restart the server when it
# is older than the served frontend (static files are re-read from disk, the
# Python process is not).
API_FEATURES = [
    "ingest", "review", "masks", "engine", "recompare", "browse", "upload",
    "upload-results",
]

# The built web UI lives inside the package (vite builds straight into it);
# wheels ship it, dev builds land in the same place.
STATIC_DIR = Path(__file__).parent.parent / "static"


class IngestRequest(BaseModel):
    output_xml: str


class DecisionRequest(BaseModel):
    actor: Optional[str] = None
    reason: Optional[str] = None


class MaskSaveRequest(BaseModel):
    file: str
    masks: object


class MaskPreviewRequest(BaseModel):
    file: str
    page: int = 1
    masks: object = None
    dpi: Optional[int] = None
    ocr_engine: Optional[str] = None
    force_ocr: bool = False


class RecompareRequest(BaseModel):
    comparison_id: int
    masks: object = None
    settings: Optional[dict] = None


class RecompareBatchRequest(BaseModel):
    masks: object = None
    comparison_ids: Optional[list] = None
    masks_file: Optional[str] = None
    settings: Optional[dict] = None


def create_app(config: AppConfig, database: Optional[Database] = None) -> FastAPI:
    from doctest_dashboard.engine import EngineService
    from doctest_dashboard.ingest import asset_token

    config.data_dir.mkdir(parents=True, exist_ok=True)
    db = database or Database(config.db_path)
    engine = EngineService(scratch_root=config.data_dir / "scratch")
    config.add_root(config.data_dir / "scratch")
    uploads_dir = config.data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    config.add_root(uploads_dir)

    app = FastAPI(title="doctest-dashboard", version=__version__)
    app.state.config = config
    app.state.db = db
    app.state.engine = engine

    def require_token(request: Request) -> None:
        if config.token is None:
            return
        header = request.headers.get("authorization", "")
        if header != f"Bearer {config.token}":
            raise HTTPException(status_code=401, detail="Missing or invalid token")

    api = Depends(require_token)

    @app.get("/api/health", dependencies=[api])
    def health():
        return {"status": "ok", "version": __version__, "features": API_FEATURES}

    @app.post("/api/ingest", dependencies=[api])
    def ingest(request: IngestRequest):
        try:
            summary = ingest_output_xml(db, config, request.output_xml)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error))
        return summary

    @app.get("/api/runs", dependencies=[api])
    def list_runs():
        runs = db.query("SELECT * FROM runs ORDER BY imported_at DESC")
        for run in runs:
            counts = db.query_one(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN c.review_state = 'unresolved' THEN 1 ELSE 0 END) AS unresolved, "
                "SUM(CASE WHEN c.status = 'FAIL' THEN 1 ELSE 0 END) AS failed "
                "FROM comparisons c JOIN tests t ON c.test_id = t.id WHERE t.run_id = ?",
                (run["id"],))
            run["comparisons"] = counts["total"] or 0
            run["unresolved"] = counts["unresolved"] or 0
            run["failed"] = counts["failed"] or 0
        return runs

    @app.get("/api/runs/{run_id}/tests", dependencies=[api])
    def list_tests(run_id: int, status: Optional[str] = None, review_state: Optional[str] = None):
        rows = db.query(
            "SELECT t.id AS test_id, t.suite, t.name, t.status AS test_status, "
            "c.id AS comparison_id, c.keyword, c.library, c.status, c.degraded, "
            "c.review_state, c.identity "
            "FROM tests t JOIN comparisons c ON c.test_id = t.id "
            "WHERE t.run_id = ? ORDER BY t.id, c.id",
            (run_id,))
        if status:
            rows = [row for row in rows if row["status"] == status.upper()]
        if review_state:
            rows = [row for row in rows if row["review_state"] == review_state]
        for row in rows:
            thumb = db.query_one(
                "SELECT images_json FROM pages WHERE comparison_id = ? AND status = 'FAIL' "
                "ORDER BY page_no LIMIT 1", (row["comparison_id"],))
            images = json.loads(thumb["images_json"]) if thumb and thumb["images_json"] else {}
            row["thumbnail"] = images.get("diff") or images.get("candidate")
        return rows

    @app.get("/api/comparisons/{comparison_id}", dependencies=[api])
    def get_comparison(comparison_id: int):
        comparison = db.query_one("SELECT * FROM comparisons WHERE id = ?", (comparison_id,))
        if not comparison:
            raise HTTPException(status_code=404, detail="Comparison not found")
        comparison["sidecar_json"] = (
            json.loads(comparison["sidecar_json"]) if comparison["sidecar_json"] else None
        )
        comparison["images"] = (
            json.loads(comparison["images_json"]) if comparison["images_json"] else []
        )
        pages = db.query(
            "SELECT * FROM pages WHERE comparison_id = ? ORDER BY page_no", (comparison_id,))
        for page in pages:
            page["regions"] = json.loads(page["regions_json"]) if page["regions_json"] else []
            page["images"] = json.loads(page["images_json"]) if page["images_json"] else {}
            del page["regions_json"], page["images_json"]
        comparison["pages"] = pages
        return comparison

    def _review(action, *args, **kwargs):
        try:
            return action(db, config, *args, **kwargs)
        except ReviewError as error:
            detail = {"message": str(error), **error.payload}
            raise HTTPException(status_code=error.status_code, detail=detail)

    @app.post("/api/comparisons/{comparison_id}/accept", dependencies=[api])
    def comparison_accept(comparison_id: int, decision: DecisionRequest):
        return _review(accept_comparison, comparison_id,
                       actor=decision.actor, reason=decision.reason)

    @app.post("/api/pages/{page_id}/accept", dependencies=[api])
    def page_accept(page_id: int, decision: DecisionRequest):
        return _review(accept_page, page_id,
                       actor=decision.actor, reason=decision.reason)

    @app.post("/api/comparisons/{comparison_id}/reject", dependencies=[api])
    def comparison_reject(comparison_id: int, decision: DecisionRequest):
        return _review(reject_comparison, comparison_id,
                       actor=decision.actor, reason=decision.reason)

    @app.get("/api/comparisons/{comparison_id}/bugdata", dependencies=[api])
    def comparison_bugdata(comparison_id: int):
        from fastapi.responses import Response

        data = _review(build_bug_bundle, comparison_id)
        return Response(
            content=data, media_type="application/zip",
            headers={"Content-Disposition":
                     f'attachment; filename="bugdata_comparison_{comparison_id}.zip"'})

    @app.get("/api/comparisons/{comparison_id}/decisions", dependencies=[api])
    def comparison_decisions(comparison_id: int):
        return db.query(
            "SELECT * FROM decisions WHERE comparison_id = ? ORDER BY id", (comparison_id,))

    # -- uploads ---------------------------------------------------------------

    UPLOAD_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".pdf", ".ps", ".pcl",
        ".tif", ".tiff", ".bmp", ".gif", ".json",
    }
    MAX_UPLOAD_BYTES = 100 * 1024 * 1024

    @app.post("/api/upload", dependencies=[api])
    async def upload(file: UploadFile):
        """Store a file from the user's machine in the dashboard workspace.

        The workspace (``{data_dir}/uploads``) is a configured root, so the
        stored file is immediately usable in the mask editor and browsable
        in the file picker — no --root setup required.
        """
        import re
        import uuid as uuid_module

        original = Path(file.filename or "upload")
        if original.suffix.lower() not in UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{original.suffix}'. "
                       f"Allowed: {', '.join(sorted(UPLOAD_EXTENSIONS))}")
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", original.stem)[:80] or "upload"
        target_dir = uploads_dir / uuid_module.uuid4().hex[:8]
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

    RESULTS_UPLOAD_EXTENSIONS = UPLOAD_EXTENSIONS | {".xml", ".txt", ".log"}
    MAX_RESULTS_UPLOAD_BYTES = 500 * 1024 * 1024

    @app.post("/api/upload-results", dependencies=[api])
    async def upload_results(files: List[UploadFile]):
        """Ingest a Robot Framework results folder uploaded from the browser.

        The folder picker (webkitdirectory) sends every file with its path
        relative to the chosen folder; the tree is stored in the workspace
        with that structure intact — which is what output.xml, screenshot
        references and sidecars rely on — then ingested in place.
        """
        import uuid as uuid_module
        from pathlib import PurePosixPath

        if not files:
            raise HTTPException(status_code=400, detail="No files in upload")
        target_root = uploads_dir / f"run_{uuid_module.uuid4().hex[:8]}"
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
            ingest_output_xml(db, config, output_xml)
            for output_xml in sorted(output_xmls, key=lambda p: len(p.parts))
        ]
        return {"stored": stored, "skipped": skipped, "runs": summaries}

    # -- file browsing --------------------------------------------------------

    @app.get("/api/browse", dependencies=[api])
    def browse(path: Optional[str] = None):
        """List configured roots, or the contents of a directory under them.

        Powers the file picker in the UI; every path is validated against
        the configured roots like any other filesystem access.
        """
        if not path:
            # Hide the internal recompare scratch area; the uploads workspace
            # stays visible so uploaded files can be browsed again.
            visible = [root for root in config.roots
                       if root != config.data_dir / "scratch"]
            return {"roots": [{"name": root.name or str(root), "path": str(root)}
                              for root in visible]}
        directory = Path(path)  # NOSONAR: paths are confined to configured roots (config.is_within_roots, symlink-safe resolve) and covered by traversal tests
        if not config.is_within_roots(directory):
            raise HTTPException(status_code=403, detail="Path outside configured roots")
        if not directory.is_dir():
            raise HTTPException(status_code=404, detail="Not a directory")
        entries = []
        try:
            children = sorted(
                directory.iterdir(),
                key=lambda child: (not child.is_dir(), child.name.lower()))
        except OSError as error:
            raise HTTPException(status_code=400, detail=str(error))
        for child in children:
            if child.name.startswith("."):
                continue
            try:
                is_dir = child.is_dir()
                entries.append({
                    "name": child.name,
                    "path": str(child),
                    "type": "dir" if is_dir else "file",
                    "size": None if is_dir else child.stat().st_size,
                })
            except OSError:
                continue
        parent = directory.parent
        return {
            "path": str(directory),
            "parent": str(parent) if config.is_within_roots(parent) else None,
            "entries": entries,
        }

    # -- masks ----------------------------------------------------------------

    @app.get("/api/masks", dependencies=[api])
    def get_masks(file: str):
        from doctest_dashboard.masks import MaskError, load_mask_file, normalize_masks

        path = Path(file)
        if not config.is_within_roots(path):
            raise HTTPException(status_code=403, detail="Mask file outside configured roots")
        try:
            return {"file": str(path), "masks": normalize_masks(load_mask_file(path))}
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error))
        except MaskError as error:
            raise HTTPException(status_code=422, detail=str(error))

    @app.put("/api/masks", dependencies=[api])
    def put_masks(request: MaskSaveRequest):
        from doctest_dashboard.masks import (
            MaskError,
            normalize_masks,
            save_mask_file,
            validate_pattern_masks,
        )

        path = Path(request.file)  # NOSONAR: paths are confined to configured roots (config.is_within_roots, symlink-safe resolve) and covered by traversal tests
        probe = path if path.exists() else path.parent
        if not config.is_within_roots(probe):
            raise HTTPException(status_code=403, detail="Mask file outside configured roots")
        try:
            masks = normalize_masks(request.masks)
            validate_pattern_masks(masks)
        except MaskError as error:
            raise HTTPException(status_code=422, detail=str(error))
        file_hash = save_mask_file(path, masks)
        db.execute(
            "INSERT INTO mask_files (path, last_seen_hash, updated_at) VALUES (?, ?, datetime('now')) "
            "ON CONFLICT(path) DO UPDATE SET last_seen_hash = excluded.last_seen_hash, "
            "updated_at = excluded.updated_at",
            (str(path), file_hash))
        return {"file": str(path), "masks": masks, "sha256": file_hash}

    # -- embedded engine ------------------------------------------------------

    @app.get("/api/capabilities", dependencies=[api])
    def capabilities():
        return {"capabilities": engine.capabilities, "ocr_available": engine.ocr_available}

    @app.post("/api/mask-preview", dependencies=[api])
    def mask_preview(request: MaskPreviewRequest):
        from doctest_dashboard.masks import MaskError, validate_pattern_masks

        path = Path(request.file)
        if not config.is_within_roots(path):
            raise HTTPException(status_code=403, detail="File outside configured roots")
        try:
            validate_pattern_masks(_as_mask_list(request.masks))
        except MaskError as error:
            raise HTTPException(status_code=422, detail=str(error))
        mask_types = {m.get("type") for m in _as_mask_list(request.masks)}
        pattern_types = {"pattern", "line_pattern", "word_pattern"}
        if (mask_types & pattern_types) and path.suffix.lower() not in (
            ".pdf",) and not engine.ocr_available:
            raise HTTPException(
                status_code=409,
                detail="Pattern mask preview requires OCR (tesseract), "
                       "which is not available in this environment")
        try:
            return engine.mask_preview(
                str(path), request.page, request.masks,
                dpi=request.dpi, ocr_engine=request.ocr_engine,
                force_ocr=request.force_ocr)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=400, detail=str(error))

    @app.get("/api/page-image", dependencies=[api])
    def page_image(file: str, page: int = 1, dpi: Optional[int] = None):
        path = Path(file)
        if not config.is_within_roots(path):
            raise HTTPException(status_code=403, detail="File outside configured roots")
        try:
            info = engine.page_image(str(path), page, dpi)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=400, detail=str(error))
        token = asset_token(info["png_path"])
        db.register_asset(token, info["png_path"])
        return {
            "page": info["page"],
            "page_count": info["page_count"],
            "dpi": info["dpi"],
            "image_size": info["image_size"],
            "image": token,
        }

    def _as_mask_list(masks) -> list:
        if masks is None:
            return []
        if isinstance(masks, dict):
            return [masks]
        if isinstance(masks, list):
            return masks
        return []

    def _run_recompare(comparison: dict, masks, settings) -> dict:
        from doctest_dashboard.masks import MaskError, validate_pattern_masks

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
            if not config.is_within_roots(Path(path)):
                raise HTTPException(status_code=403, detail="Path outside configured roots")
        try:
            outcome = engine.recompare(reference, candidate, masks, settings)
        except TimeoutError as error:
            raise HTTPException(status_code=504, detail=str(error))
        sidecar = outcome["result"]
        scratch = Path(outcome["scratch_dir"])
        for page in sidecar.get("pages", []):
            tokens = {}
            for kind, rel in page.get("images", {}).items():
                absolute = (scratch / rel).resolve()
                token = asset_token(str(absolute))
                db.register_asset(token, str(absolute))
                tokens[kind] = token
            page["images"] = tokens
        return {
            "comparison_id": comparison["id"],
            "status": sidecar["status"],
            "pages": sidecar.get("pages", []),
            "cached": outcome.get("cached", False),
        }

    @app.post("/api/recompare", dependencies=[api])
    def recompare(request: RecompareRequest):
        comparison = db.query_one(
            "SELECT * FROM comparisons WHERE id = ?", (request.comparison_id,))
        if not comparison:
            raise HTTPException(status_code=404, detail="Comparison not found")
        return _run_recompare(comparison, request.masks, request.settings)

    @app.post("/api/recompare-batch", dependencies=[api])
    def recompare_batch(request: RecompareBatchRequest):
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
                results.append(_run_recompare(comparison, request.masks, request.settings))
            except HTTPException as error:
                results.append({
                    "comparison_id": comparison["id"],
                    "status": "ERROR",
                    "error": error.detail,
                })
        return {"results": results}

    @app.get("/api/assets/{token}", dependencies=[api])
    def get_asset(token: str):
        path = db.resolve_asset(token)
        if not path:
            raise HTTPException(status_code=404, detail="Unknown asset")
        if not config.is_within_roots(Path(path)):
            raise HTTPException(status_code=403, detail="Asset outside configured roots")
        media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return FileResponse(
            path, media_type=media_type,
            headers={"Cache-Control": "private, max-age=86400"})

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app
