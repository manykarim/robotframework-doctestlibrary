"""Runs: ingestion, listing, grid, comparison detail, deletion."""

import json

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from doctest_dashboard.ingest import ingest_output_xml
from doctest_dashboard.server.schemas import IngestRequest, RunLabelRequest
from doctest_dashboard.server.state import AppState, get_state, require_token

router = APIRouter(dependencies=[Depends(require_token)])


@router.post("/api/ingest")
def ingest(request: IngestRequest, state: AppState = Depends(get_state)):
    try:
        return ingest_output_xml(state.db, state.config, request.output_xml)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))


@router.get("/api/runs")
def list_runs(limit: int = 50, offset: int = 0, state: AppState = Depends(get_state)):
    runs, total = state.db.list_runs(limit=limit, offset=offset)
    return {"runs": runs, "total": total}


@router.get("/api/runs/{run_id}/tests")
def list_tests(
    run_id: int,
    status: Optional[str] = None,
    review_state: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    state: AppState = Depends(get_state),
):
    rows, total = state.db.list_grid(
        run_id, status=status, review_state=review_state, limit=limit, offset=offset)
    return {"rows": rows, "total": total}


@router.get("/api/runs/{run_id}/groups")
def list_groups(run_id: int, state: AppState = Depends(get_state)):
    """Similarity groups of unresolved failures (Percy-style matching diffs)."""
    if not state.db.query_one("SELECT id FROM runs WHERE id = ?", (run_id,)):
        raise HTTPException(status_code=404, detail="Run not found")
    return state.db.list_groups(run_id)


@router.patch("/api/runs/{run_id}")
def rename_run(run_id: int, request: RunLabelRequest, state: AppState = Depends(get_state)):
    if not state.db.query_one("SELECT id FROM runs WHERE id = ?", (run_id,)):
        raise HTTPException(status_code=404, detail="Run not found")
    label = (request.label or "").strip() or None
    state.db.set_run_label(run_id, label)
    return {"id": run_id, "label": label}


@router.delete("/api/runs/{run_id}")
def delete_run(run_id: int, state: AppState = Depends(get_state)):
    if not state.db.query_one("SELECT id FROM runs WHERE id = ?", (run_id,)):
        raise HTTPException(status_code=404, detail="Run not found")
    pruned = state.db.delete_run(run_id)
    return {"deleted": run_id, "assets_pruned": pruned}


@router.get("/api/flaky")
def flaky(window: int = 10, state: AppState = Depends(get_state)):
    """Comparison identities whose status flipped across recent runs."""
    return {"flaky": state.db.flaky_identities(window=window)}


@router.get("/api/comparisons/{comparison_id}/history")
def comparison_history(comparison_id: int, state: AppState = Depends(get_state)):
    history = state.db.comparison_history(comparison_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Comparison not found")
    return {"history": history}


@router.get("/api/comparisons/{comparison_id}")
def get_comparison(comparison_id: int, state: AppState = Depends(get_state)):
    db = state.db
    comparison = db.query_one(
        "SELECT c.*, t.name AS test_name, t.suite AS suite "
        "FROM comparisons c JOIN tests t ON c.test_id = t.id WHERE c.id = ?",
        (comparison_id,))
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
