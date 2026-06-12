"""SQLite persistence: thin DAL over WAL-mode sqlite3.

Write rates are tiny (ingests and review decisions), so plain sqlite3 with
a small helper class is deliberate — no ORM.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    output_xml_path TEXT NOT NULL UNIQUE,
    name TEXT,
    started TEXT,
    imported_at TEXT NOT NULL,
    rf_version TEXT
);
CREATE TABLE IF NOT EXISTS tests (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    suite TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT
);
CREATE TABLE IF NOT EXISTS comparisons (
    id INTEGER PRIMARY KEY,
    test_id INTEGER NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
    keyword TEXT NOT NULL,
    library TEXT,
    status TEXT NOT NULL,
    degraded INTEGER NOT NULL DEFAULT 0,
    review_state TEXT NOT NULL DEFAULT 'unresolved',
    sidecar_path TEXT,
    sidecar_json TEXT,
    reference_path TEXT,
    candidate_path TEXT,
    identity TEXT NOT NULL,
    images_json TEXT
);
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY,
    comparison_id INTEGER NOT NULL REFERENCES comparisons(id) ON DELETE CASCADE,
    page_no INTEGER NOT NULL,
    status TEXT NOT NULL,
    score REAL,
    threshold REAL,
    regions_json TEXT,
    images_json TEXT,
    review_state TEXT NOT NULL DEFAULT 'unresolved',
    content_key TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    comparison_id INTEGER REFERENCES comparisons(id) ON DELETE SET NULL,
    page_id INTEGER REFERENCES pages(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    actor TEXT,
    reason TEXT,
    prev_sha256 TEXT,
    new_sha256 TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mask_files (
    path TEXT PRIMARY KEY,
    last_seen_hash TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS assets (
    token TEXT PRIMARY KEY,
    path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tests_run ON tests(run_id);
CREATE INDEX IF NOT EXISTS idx_comparisons_test ON comparisons(test_id);
CREATE INDEX IF NOT EXISTS idx_comparisons_identity ON comparisons(identity);
CREATE INDEX IF NOT EXISTS idx_pages_comparison ON pages(comparison_id);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Database:
    """Thread-safe-enough sqlite wrapper (one connection, one lock)."""

    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return cursor

    def query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def query_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # -- runs ---------------------------------------------------------------

    def upsert_run(self, output_xml_path: str, name: str, started: Optional[str],
                   rf_version: Optional[str]) -> int:
        """Insert the run or, when re-ingesting the same output.xml, wipe and
        re-create its children so ingestion stays idempotent."""
        existing = self.query_one(
            "SELECT id FROM runs WHERE output_xml_path = ?", (output_xml_path,))
        if existing:
            run_id = existing["id"]
            self.execute("DELETE FROM tests WHERE run_id = ?", (run_id,))
            self.execute(
                "UPDATE runs SET name = ?, started = ?, imported_at = ?, rf_version = ? WHERE id = ?",
                (name, started, _now(), rf_version, run_id))
            return run_id
        cursor = self.execute(
            "INSERT INTO runs (output_xml_path, name, started, imported_at, rf_version) "
            "VALUES (?, ?, ?, ?, ?)",
            (output_xml_path, name, started, _now(), rf_version))
        return cursor.lastrowid

    def insert_test(self, run_id: int, suite: str, name: str, status: str,
                    message: str) -> int:
        cursor = self.execute(
            "INSERT INTO tests (run_id, suite, name, status, message) VALUES (?, ?, ?, ?, ?)",
            (run_id, suite, name, status, message))
        return cursor.lastrowid

    def insert_comparison(self, test_id: int, keyword: str, library: Optional[str],
                          status: str, degraded: bool, identity: str,
                          sidecar_path: Optional[str] = None,
                          sidecar_json: Optional[Dict[str, Any]] = None,
                          reference_path: Optional[str] = None,
                          candidate_path: Optional[str] = None,
                          images: Optional[List[str]] = None) -> int:
        cursor = self.execute(
            "INSERT INTO comparisons (test_id, keyword, library, status, degraded, "
            "review_state, sidecar_path, sidecar_json, reference_path, candidate_path, "
            "identity, images_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (test_id, keyword, library, status, int(degraded),
             "unresolved" if status == "FAIL" else "passed",
             sidecar_path,
             json.dumps(sidecar_json) if sidecar_json is not None else None,
             reference_path, candidate_path, identity,
             json.dumps(images) if images is not None else None))
        return cursor.lastrowid

    def insert_page(self, comparison_id: int, page_no: int, status: str,
                    score: Optional[float], threshold: Optional[float],
                    regions: List[Dict[str, int]], images: Dict[str, str],
                    content_key: Optional[str]) -> int:
        cursor = self.execute(
            "INSERT INTO pages (comparison_id, page_no, status, score, threshold, "
            "regions_json, images_json, review_state, content_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (comparison_id, page_no, status, score, threshold,
             json.dumps(regions), json.dumps(images),
             "unresolved" if status == "FAIL" else "passed", content_key))
        return cursor.lastrowid

    # -- decisions ----------------------------------------------------------

    def insert_decision(self, action: str, actor: Optional[str], reason: Optional[str],
                        comparison_id: Optional[int] = None, page_id: Optional[int] = None,
                        prev_sha256: Optional[str] = None,
                        new_sha256: Optional[str] = None) -> int:
        cursor = self.execute(
            "INSERT INTO decisions (comparison_id, page_id, action, actor, reason, "
            "prev_sha256, new_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (comparison_id, page_id, action, actor, reason, prev_sha256, new_sha256, _now()))
        return cursor.lastrowid

    def set_comparison_state(self, comparison_id: int, state: str) -> None:
        self.execute(
            "UPDATE comparisons SET review_state = ? WHERE id = ?", (state, comparison_id))

    def set_page_state(self, page_id: int, state: str) -> None:
        self.execute("UPDATE pages SET review_state = ? WHERE id = ?", (state, page_id))

    # -- assets ---------------------------------------------------------------

    def register_asset(self, token: str, path: str) -> None:
        self.execute(
            "INSERT INTO assets (token, path) VALUES (?, ?) "
            "ON CONFLICT(token) DO UPDATE SET path = excluded.path",
            (token, path))

    def resolve_asset(self, token: str) -> Optional[str]:
        row = self.query_one("SELECT path FROM assets WHERE token = ?", (token,))
        return row["path"] if row else None
