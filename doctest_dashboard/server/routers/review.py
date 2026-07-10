"""Review actions: accept (single/batch), reject, decisions, bug bundles."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from doctest_dashboard.review import (
    ReviewError,
    accept_comparison,
    accept_many,
    accept_page,
    build_bug_bundle,
    reject_comparison,
)
from doctest_dashboard.server.schemas import BatchDecisionRequest, DecisionRequest
from doctest_dashboard.server.state import AppState, get_state, require_token

router = APIRouter(dependencies=[Depends(require_token)])


def _review(state: AppState, action, *args, **kwargs):
    try:
        return action(state.db, state.config, *args, **kwargs)
    except ReviewError as error:
        detail = {"message": str(error), **error.payload}
        raise HTTPException(status_code=error.status_code, detail=detail)


@router.post("/api/comparisons/{comparison_id}/accept")
def comparison_accept(comparison_id: int, decision: DecisionRequest,
                      state: AppState = Depends(get_state)):
    return _review(state, accept_comparison, comparison_id,
                   actor=decision.actor, reason=decision.reason)


@router.post("/api/comparisons/accept-batch")
def comparison_accept_batch(request: BatchDecisionRequest,
                            state: AppState = Depends(get_state)):
    return accept_many(state.db, state.config, request.ids,
                       actor=request.actor, reason=request.reason)


@router.post("/api/runs/{run_id}/accept")
def run_accept(run_id: int, decision: DecisionRequest,
               state: AppState = Depends(get_state)):
    ids = [row["id"] for row in state.db.query(
        "SELECT c.id AS id FROM comparisons c JOIN tests t ON c.test_id = t.id "
        "WHERE t.run_id = ? AND c.status = 'FAIL' AND c.review_state = 'unresolved' "
        "ORDER BY c.id", (run_id,))]
    return accept_many(state.db, state.config, ids,
                       actor=decision.actor, reason=decision.reason)


@router.post("/api/pages/{page_id}/accept")
def page_accept(page_id: int, decision: DecisionRequest,
                state: AppState = Depends(get_state)):
    return _review(state, accept_page, page_id,
                   actor=decision.actor, reason=decision.reason)


@router.post("/api/comparisons/{comparison_id}/reject")
def comparison_reject(comparison_id: int, decision: DecisionRequest,
                      state: AppState = Depends(get_state)):
    return _review(state, reject_comparison, comparison_id,
                   actor=decision.actor, reason=decision.reason)


@router.get("/api/comparisons/{comparison_id}/bugdata")
def comparison_bugdata(comparison_id: int, state: AppState = Depends(get_state)):
    data = _review(state, build_bug_bundle, comparison_id)
    return Response(
        content=data, media_type="application/zip",
        headers={"Content-Disposition":
                 f'attachment; filename="bugdata_comparison_{comparison_id}.zip"'})


@router.get("/api/comparisons/{comparison_id}/decisions")
def comparison_decisions(comparison_id: int, state: AppState = Depends(get_state)):
    return state.db.query(
        "SELECT * FROM decisions WHERE comparison_id = ? ORDER BY id", (comparison_id,))
