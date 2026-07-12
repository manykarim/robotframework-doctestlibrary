"""Review actions: accept (baseline promotion), reject, bug-data bundles.

Accept copies the candidate source file over the reference file — exactly
the layout a ``REFERENCE_RUN`` produces — and records SHA-256 of the file
before and after in the audit table. All writes are confined to the
configured roots.
"""

import hashlib
import io
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from doctest_dashboard.config import AppConfig
from doctest_dashboard.db import Database

DOCUMENT_SUFFIXES = {".pdf", ".ps", ".pcl"}


class ReviewError(Exception):
    """Review action failed; ``status_code`` maps to the HTTP layer."""

    def __init__(self, message: str, status_code: int = 400, **payload):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _get_comparison(database: Database, comparison_id: int) -> Dict[str, Any]:
    comparison = database.query_one(
        "SELECT * FROM comparisons WHERE id = ?", (comparison_id,))
    if not comparison:
        raise ReviewError("Comparison not found", status_code=404)
    return comparison


def _paths_for_promotion(config: AppConfig, comparison: Dict[str, Any]) -> tuple:
    reference = comparison["reference_path"]
    candidate = comparison["candidate_path"]
    if not reference or not candidate:
        raise ReviewError(
            "Comparison has no recorded reference/candidate paths "
            "(degraded record — re-run with result_json enabled)",
            status_code=409)
    reference_path = Path(reference)
    candidate_path = Path(candidate)
    if not candidate_path.is_file():
        raise ReviewError(f"Candidate file no longer exists: {candidate}", status_code=409)
    if not config.is_within_roots(candidate_path):
        raise ReviewError("Candidate path outside configured roots", status_code=403)
    # The reference may not exist yet; validate its parent directory instead.
    probe = reference_path if reference_path.exists() else reference_path.parent
    if not config.is_within_roots(probe):
        raise ReviewError("Reference path outside configured roots", status_code=403)
    return reference_path, candidate_path


def is_document_artifact(path: Optional[str]) -> bool:
    return bool(path) and Path(path).suffix.lower() in DOCUMENT_SUFFIXES


def accept_comparison(
    database: Database,
    config: AppConfig,
    comparison_id: int,
    actor: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Promote the candidate document/image to be the new reference."""
    comparison = _get_comparison(database, comparison_id)
    reference_path, candidate_path = _paths_for_promotion(config, comparison)
    prev_hash = _sha256(reference_path)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate_path, reference_path)
    new_hash = _sha256(reference_path)
    database.insert_decision(
        action="accept", actor=actor, reason=reason,
        comparison_id=comparison_id, prev_sha256=prev_hash, new_sha256=new_hash)
    database.set_comparison_state(comparison_id, "accepted")
    for page in database.query(
            "SELECT id FROM pages WHERE comparison_id = ? AND review_state = 'unresolved'",
            (comparison_id,)):
        database.set_page_state(page["id"], "accepted")
    return {
        "comparison_id": comparison_id,
        "reference_path": str(reference_path),
        "prev_sha256": prev_hash,
        "new_sha256": new_hash,
        "review_state": "accepted",
    }


def accept_many(
    database: Database,
    config: AppConfig,
    comparison_ids,
    actor: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Batch accept: promote each comparison, skipping the impossible ones
    (degraded, outside roots, missing candidate) with a reason instead of
    aborting the batch. Every promotion keeps its own audit row."""
    accepted, skipped = [], []
    for comparison_id in comparison_ids:
        try:
            accepted.append(
                accept_comparison(database, config, comparison_id, actor, reason))
        except ReviewError as error:
            skipped.append({"comparison_id": comparison_id, "reason": str(error)})
    return {"accepted": accepted, "skipped": skipped}


def accept_page(
    database: Database,
    config: AppConfig,
    page_id: int,
    actor: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Accept a single page.

    For single-image artifacts this is a file copy. For multi-page document
    artifacts (PDF/PS/PCL) page-level promotion is impossible at file level
    — respond with the documented alternatives instead of pretending.
    """
    page = database.query_one("SELECT * FROM pages WHERE id = ?", (page_id,))
    if not page:
        raise ReviewError("Page not found", status_code=404)
    comparison = _get_comparison(database, page["comparison_id"])
    page_count = len(database.query(
        "SELECT id FROM pages WHERE comparison_id = ?", (comparison["id"],)))
    if is_document_artifact(comparison["candidate_path"]) and page_count > 1:
        raise ReviewError(
            "Page-level accept is not possible for multi-page documents: the "
            "reference is a single file. Accept the whole document, or mask "
            "the intended change instead.",
            status_code=409,
            alternatives=["accept_document", "create_mask"])
    result = accept_comparison(database, config, comparison["id"], actor, reason)
    database.set_page_state(page_id, "accepted")
    result["page_id"] = page_id
    return result


def reject_comparison(
    database: Database,
    config: AppConfig,
    comparison_id: int,
    actor: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    comparison = _get_comparison(database, comparison_id)
    database.insert_decision(
        action="reject", actor=actor, reason=reason, comparison_id=comparison_id)
    database.set_comparison_state(comparison_id, "rejected")
    for page in database.query(
            "SELECT id FROM pages WHERE comparison_id = ? AND review_state = 'unresolved'",
            (comparison_id,)):
        database.set_page_state(page["id"], "rejected")
    return {"comparison_id": comparison_id, "review_state": "rejected"}


def build_bug_bundle(database: Database, config: AppConfig, comparison_id: int) -> bytes:
    """ZIP with reference, candidate, diff images, sidecar, and metadata."""
    comparison = _get_comparison(database, comparison_id)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for label in ("reference_path", "candidate_path"):
            value = comparison[label]
            if value and Path(value).is_file() and config.is_within_roots(Path(value)):
                bundle.write(value, f"{label.replace('_path', '')}/{Path(value).name}")
        if comparison["sidecar_path"] and Path(comparison["sidecar_path"]).is_file():
            bundle.write(comparison["sidecar_path"], "comparison_result.json")
        for page in database.query(
                "SELECT * FROM pages WHERE comparison_id = ? AND status = 'FAIL'",
                (comparison_id,)):
            images = json.loads(page["images_json"]) if page["images_json"] else {}
            for kind, token in images.items():
                path = database.resolve_asset(token)
                if path and Path(path).is_file() and config.is_within_roots(Path(path)):
                    bundle.write(path, f"pages/page_{page['page_no']}_{kind}{Path(path).suffix}")
        decisions = database.query(
            "SELECT * FROM decisions WHERE comparison_id = ? ORDER BY id", (comparison_id,))
        metadata = {
            "comparison_id": comparison_id,
            "identity": comparison["identity"],
            "keyword": comparison["keyword"],
            "status": comparison["status"],
            "review_state": comparison["review_state"],
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "decisions": decisions,
        }
        bundle.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))
    return buffer.getvalue()
