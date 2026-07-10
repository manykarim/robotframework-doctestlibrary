"""FastAPI application factory: wiring, health, static UI."""

import logging
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from doctest_dashboard import __version__
from doctest_dashboard.config import AppConfig
from doctest_dashboard.db import Database
from doctest_dashboard.server.features import API_FEATURES
from doctest_dashboard.server.state import AppState, require_token
from doctest_dashboard.storage import run_gc

LOG = logging.getLogger(__name__)

# The built web UI lives inside the package (vite builds straight into it);
# wheels ship it, dev builds land in the same place.
STATIC_DIR = Path(__file__).parent.parent / "static"


def create_app(config: AppConfig, database: Optional[Database] = None) -> FastAPI:
    from doctest_dashboard.engine import EngineService
    from doctest_dashboard.server.routers import (
        assets, browse, engine as engine_router, masks, review, runs, uploads,
    )

    config.data_dir.mkdir(parents=True, exist_ok=True)
    db = database or Database(config.db_path)
    engine = EngineService(scratch_root=config.data_dir / "scratch")
    config.add_root(config.data_dir / "scratch")
    uploads_dir = config.data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    config.add_root(uploads_dir)

    swept = run_gc(config)
    if any(swept.values()):
        LOG.info("storage GC removed %s", swept)

    app = FastAPI(title="doctest-dashboard", version=__version__)
    app.state.ctx = AppState(config=config, db=db, engine=engine, uploads_dir=uploads_dir)

    @app.get("/api/health", dependencies=[Depends(require_token)])
    def health():
        return {"status": "ok", "version": __version__, "features": API_FEATURES}

    for router_module in (runs, review, uploads, browse, masks, engine_router, assets):
        app.include_router(router_module.router)

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app
