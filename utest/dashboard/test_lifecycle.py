"""Lifecycle tests: run deletion, storage GC, bounded caches, query counts."""

import os
import time

from doctest_dashboard.engine import MAX_CACHE_ENTRIES, EngineService
from doctest_dashboard.ingest import ingest_output_xml
from doctest_dashboard.storage import sweep_directory

from test_review import _failing_comparison, _ingest, _make_image_run


def test_delete_run_cascades_and_prunes_assets(client, config, database, tmp_path):
    output_xml, reference, candidate, artifacts = _make_image_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)
    detail = client.get(f"/api/comparisons/{comparison_id}").json()
    token = detail["pages"][0]["images"]["diff"]
    assert client.get(f"/api/assets/{token}").status_code == 200

    response = client.delete(f"/api/runs/{run_id}")
    assert response.status_code == 200
    assert response.json()["assets_pruned"] > 0

    assert client.get(f"/api/comparisons/{comparison_id}").status_code == 404
    assert client.get(f"/api/assets/{token}").status_code == 404
    assert client.get("/api/runs").json()["total"] == 0
    # files on disk are untouched — deletion is a dashboard-side action
    assert reference.exists() and candidate.exists()


def test_delete_run_leaves_other_runs_alone(client, config, tmp_path):
    xml1, _, _, artifacts1 = _make_image_run(tmp_path / "a")
    xml2, _, _, artifacts2 = _make_image_run(tmp_path / "b")
    run1 = _ingest(client, config, xml1, artifacts1)
    run2 = _ingest(client, config, xml2, artifacts2)
    client.delete(f"/api/runs/{run1}")
    body = client.get("/api/runs").json()
    assert body["total"] == 1
    assert body["runs"][0]["id"] == run2
    assert _failing_comparison(client, run2)  # still fully readable


def test_delete_unknown_run_404(client):
    assert client.delete("/api/runs/9999").status_code == 404


def test_sweep_directory_removes_only_aged_entries(tmp_path):
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "f.png").write_bytes(b"x")
    aged = time.time() - 10 * 86400
    os.utime(old_dir, (aged, aged))

    removed = sweep_directory(tmp_path, ttl_days=7)
    assert removed == 1
    assert not old_dir.exists()
    assert new_dir.exists()


def test_sweep_missing_root_is_noop(tmp_path):
    assert sweep_directory(tmp_path / "nope", ttl_days=1) == 0


def test_engine_cache_is_bounded(tmp_path):
    engine = EngineService(scratch_root=tmp_path)
    for index in range(MAX_CACHE_ENTRIES + 50):
        engine._cache_put(f"key-{index}", {"value": index})
    assert len(engine._cache) == MAX_CACHE_ENTRIES
    # oldest entries were evicted, newest survive
    assert engine._cache_get("key-0") is None
    assert engine._cache_get(f"key-{MAX_CACHE_ENTRIES + 49}") is not None


def test_list_endpoints_use_constant_query_count(client, config, database, sidecar_output_xml,
                                                 monkeypatch):
    ingest_output_xml(database, config, sidecar_output_xml)
    calls = {"n": 0}
    original = type(database).query

    def counting_query(self, sql, params=()):
        calls["n"] += 1
        return original(self, sql, params)

    monkeypatch.setattr(type(database), "query", counting_query)

    calls["n"] = 0
    client.get("/api/runs")
    runs_queries = calls["n"]

    run_id = 1
    calls["n"] = 0
    client.get(f"/api/runs/{run_id}/tests")
    grid_queries = calls["n"]

    # aggregate queries: no per-row loops (a couple of queries incl. totals)
    assert runs_queries <= 3, runs_queries
    assert grid_queries <= 3, grid_queries
