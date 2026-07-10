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
    images_json TEXT,
    group_key TEXT
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
CREATE INDEX IF NOT EXISTS idx_comparisons_review ON comparisons(review_state);
CREATE INDEX IF NOT EXISTS idx_comparisons_group ON comparisons(group_key);
CREATE INDEX IF NOT EXISTS idx_pages_comp_status ON pages(comparison_id, status);
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
            # pre-existing dev databases (unreleased product): additive column
            try:
                self._conn.execute("ALTER TABLE comparisons ADD COLUMN group_key TEXT")
            except sqlite3.OperationalError:
                pass
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
                          images: Optional[List[str]] = None,
                          group_key: Optional[str] = None) -> int:
        cursor = self.execute(
            "INSERT INTO comparisons (test_id, keyword, library, status, degraded, "
            "review_state, sidecar_path, sidecar_json, reference_path, candidate_path, "
            "identity, images_json, group_key) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (test_id, keyword, library, status, int(degraded),
             "unresolved" if status == "FAIL" else "passed",
             sidecar_path,
             json.dumps(sidecar_json) if sidecar_json is not None else None,
             reference_path, candidate_path, identity,
             json.dumps(images) if images is not None else None,
             group_key))
        return cursor.lastrowid

    def list_groups(self, run_id: int):
        """Similarity groups (size >= 2) of unresolved failures + ungrouped count."""
        rows = self.query(
            "SELECT c.id AS comparison_id, c.group_key, t.name, "
            "(SELECT p.images_json FROM pages p "
            " WHERE p.comparison_id = c.id AND p.status = 'FAIL' "
            " ORDER BY p.page_no LIMIT 1) AS thumb_json "
            "FROM comparisons c JOIN tests t ON c.test_id = t.id "
            "WHERE t.run_id = ? AND c.status = 'FAIL' AND c.review_state = 'unresolved' "
            "ORDER BY c.id", (run_id,))
        by_key: Dict[str, list] = {}
        ungrouped = 0
        for row in rows:
            images = json.loads(row["thumb_json"]) if row["thumb_json"] else {}
            member = {
                "comparison_id": row["comparison_id"],
                "name": row["name"],
                "thumbnail": images.get("diff") or images.get("candidate"),
            }
            if row["group_key"]:
                by_key.setdefault(row["group_key"], []).append(member)
            else:
                ungrouped += 1
        groups = []
        for key, members in by_key.items():
            if len(members) < 2:
                ungrouped += 1
                continue
            groups.append({
                "group_key": key,
                "count": len(members),
                "thumbnail": members[0]["thumbnail"],
                "members": members,
            })
        groups.sort(key=lambda g: -g["count"])
        return {"groups": groups, "ungrouped": ungrouped}

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

    # -- aggregate list queries ------------------------------------------------

    def list_runs(self, limit: int = 50, offset: int = 0):
        """Runs with review counts in a constant number of queries."""
        rows = self.query(
            "SELECT r.*, COALESCE(a.total, 0) AS comparisons, "
            "COALESCE(a.unresolved, 0) AS unresolved, COALESCE(a.failed, 0) AS failed "
            "FROM runs r LEFT JOIN ("
            "  SELECT t.run_id AS run_id, COUNT(*) AS total, "
            "    SUM(CASE WHEN c.review_state = 'unresolved' THEN 1 ELSE 0 END) AS unresolved, "
            "    SUM(CASE WHEN c.status = 'FAIL' THEN 1 ELSE 0 END) AS failed "
            "  FROM comparisons c JOIN tests t ON c.test_id = t.id GROUP BY t.run_id"
            ") a ON a.run_id = r.id "
            "ORDER BY r.imported_at DESC, r.id DESC LIMIT ? OFFSET ?",
            (limit, offset))
        total = self.query_one("SELECT COUNT(*) AS n FROM runs")["n"]
        return rows, total

    def list_grid(self, run_id: int, status: Optional[str] = None,
                  review_state: Optional[str] = None,
                  limit: int = 50, offset: int = 0):
        """Grid rows with SQL-side filters and a subquery thumbnail."""
        where = ["t.run_id = ?"]
        params: list = [run_id]
        if status:
            where.append("c.status = ?")
            params.append(status.upper())
        if review_state:
            where.append("c.review_state = ?")
            params.append(review_state)
        clause = " AND ".join(where)
        total = self.query_one(
            f"SELECT COUNT(*) AS n FROM tests t JOIN comparisons c ON c.test_id = t.id "
            f"WHERE {clause}", tuple(params))["n"]
        rows = self.query(
            "SELECT t.id AS test_id, t.suite, t.name, t.status AS test_status, "
            "c.id AS comparison_id, c.keyword, c.library, c.status, c.degraded, "
            "c.review_state, c.identity, "
            "(SELECT p.images_json FROM pages p "
            " WHERE p.comparison_id = c.id AND p.status = 'FAIL' "
            " ORDER BY p.page_no LIMIT 1) AS thumb_json "
            "FROM tests t JOIN comparisons c ON c.test_id = t.id "
            f"WHERE {clause} ORDER BY t.id, c.id LIMIT ? OFFSET ?",
            tuple(params) + (limit, offset))
        for row in rows:
            images = json.loads(row["thumb_json"]) if row["thumb_json"] else {}
            row["thumbnail"] = images.get("diff") or images.get("candidate")
            del row["thumb_json"]
        return rows, total

    def delete_run(self, run_id: int) -> int:
        """Delete a run (children cascade) and prune its asset registrations.

        Decision rows survive with nulled references (audit history).
        Returns the number of pruned assets.
        """
        tokens: set = set()
        for row in self.query(
                "SELECT p.images_json FROM pages p "
                "JOIN comparisons c ON p.comparison_id = c.id "
                "JOIN tests t ON c.test_id = t.id WHERE t.run_id = ?", (run_id,)):
            if row["images_json"]:
                tokens.update(json.loads(row["images_json"]).values())
        for row in self.query(
                "SELECT c.images_json FROM comparisons c "
                "JOIN tests t ON c.test_id = t.id WHERE t.run_id = ?", (run_id,)):
            if row["images_json"]:
                tokens.update(json.loads(row["images_json"]))
        pruned = 0
        for token in tokens:
            self.execute("DELETE FROM assets WHERE token = ?", (token,))
            pruned += 1
        self.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return pruned

    def comparison_history(self, comparison_id: int):
        """Timeline of a comparison's identity across all ingested runs."""
        comparison = self.query_one(
            "SELECT identity FROM comparisons WHERE id = ?", (comparison_id,))
        if not comparison:
            return None
        return self.query(
            "SELECT c.id AS comparison_id, c.status, c.review_state, "
            "r.id AS run_id, r.name AS run_name, r.imported_at, "
            "(SELECT p.score FROM pages p WHERE p.comparison_id = c.id "
            " AND p.status = 'FAIL' ORDER BY p.page_no LIMIT 1) AS score "
            "FROM comparisons c JOIN tests t ON c.test_id = t.id "
            "JOIN runs r ON t.run_id = r.id "
            "WHERE c.identity = ? ORDER BY r.imported_at DESC, r.id DESC",
            (comparison["identity"],))

    def flaky_identities(self, window: int = 10, min_flips: int = 1):
        """Identities whose status flipped across their recent occurrences."""
        rows = self.query(
            "SELECT c.identity, c.status, t.name, r.imported_at "
            "FROM comparisons c JOIN tests t ON c.test_id = t.id "
            "JOIN runs r ON t.run_id = r.id "
            "ORDER BY c.identity, r.imported_at DESC, r.id DESC")
        results = []
        current: Dict[str, Any] = {}
        for row in rows:
            if row["identity"] != current.get("identity"):
                if current and current["flips"] >= min_flips:
                    results.append(current)
                current = {"identity": row["identity"], "name": row["name"],
                           "statuses": [], "flips": 0}
            if len(current["statuses"]) < window:
                if current["statuses"] and current["statuses"][-1] != row["status"]:
                    current["flips"] += 1
                current["statuses"].append(row["status"])
        if current and current["flips"] >= min_flips:
            results.append(current)
        for item in results:
            item["occurrences"] = len(item["statuses"])
            item["last_status"] = item["statuses"][0] if item["statuses"] else None
        results.sort(key=lambda item: -item["flips"])
        return results

    # -- assets ---------------------------------------------------------------

    def register_asset(self, token: str, path: str) -> None:
        self.execute(
            "INSERT INTO assets (token, path) VALUES (?, ?) "
            "ON CONFLICT(token) DO UPDATE SET path = excluded.path",
            (token, path))

    def resolve_asset(self, token: str) -> Optional[str]:
        row = self.query_one("SELECT path FROM assets WHERE token = ?", (token,))
        return row["path"] if row else None
