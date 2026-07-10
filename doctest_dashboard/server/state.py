"""Shared application state and dependencies for the API routers."""

from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request

from doctest_dashboard.config import AppConfig
from doctest_dashboard.db import Database
from doctest_dashboard.engine import EngineService


@dataclass
class AppState:
    config: AppConfig
    db: Database
    engine: EngineService
    uploads_dir: Path


def get_state(request: Request) -> AppState:
    return request.app.state.ctx


def require_token(request: Request) -> None:
    config: AppConfig = request.app.state.ctx.config
    if config.token is None:
        return
    header = request.headers.get("authorization", "")
    if header != f"Bearer {config.token}":
        raise HTTPException(status_code=401, detail="Missing or invalid token")
