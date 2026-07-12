"""API tests: ingest endpoint, run/test queries, asset confinement, auth."""

import os
from pathlib import Path

from fastapi.testclient import TestClient

from doctest_dashboard.config import AppConfig
from doctest_dashboard.db import Database
from doctest_dashboard.ingest import asset_token
from doctest_dashboard.server.app import create_app


def _ingest_via_api(client, output_xml):
    response = client.post("/api/ingest", json={"output_xml": str(output_xml)})
    assert response.status_code == 200, response.text
    return response.json()


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_ingest_endpoint_and_run_listing(client, sidecar_output_xml):
    summary = _ingest_via_api(client, sidecar_output_xml)
    assert summary["comparisons"] == 3
    body = client.get("/api/runs").json()
    runs = body["runs"]
    assert body["total"] == 1
    assert len(runs) == 1
    assert runs[0]["comparisons"] == 3
    assert runs[0]["failed"] == 2
    assert runs[0]["unresolved"] == 2


def test_ingest_missing_file_404(client, tmp_path):
    response = client.post("/api/ingest", json={"output_xml": str(tmp_path / "nope.xml")})
    assert response.status_code == 404


def test_test_listing_with_filters(client, sidecar_output_xml):
    run_id = _ingest_via_api(client, sidecar_output_xml)["run_id"]
    body = client.get(f"/api/runs/{run_id}/tests").json()
    assert len(body["rows"]) == 3
    assert body["total"] == 3
    failed = client.get(f"/api/runs/{run_id}/tests", params={"status": "fail"}).json()
    assert len(failed["rows"]) == 2
    assert failed["total"] == 2
    assert all(row["thumbnail"] for row in failed["rows"])
    unresolved = client.get(
        f"/api/runs/{run_id}/tests", params={"review_state": "unresolved"}).json()
    assert len(unresolved["rows"]) == 2
    paged = client.get(f"/api/runs/{run_id}/tests", params={"limit": 2, "offset": 2}).json()
    assert len(paged["rows"]) == 1
    assert paged["total"] == 3


def test_comparison_detail(client, sidecar_output_xml):
    run_id = _ingest_via_api(client, sidecar_output_xml)["run_id"]
    rows = client.get(f"/api/runs/{run_id}/tests", params={"status": "fail"}).json()["rows"]
    detail = client.get(f"/api/comparisons/{rows[0]['comparison_id']}").json()
    assert detail["sidecar_json"]["schema_version"] == 1
    assert len(detail["pages"]) >= 1
    page = detail["pages"][0]
    assert page["regions"]
    assert set(page["images"]) >= {"reference", "candidate", "diff"}


def test_asset_served_with_cache_headers(client, sidecar_output_xml):
    run_id = _ingest_via_api(client, sidecar_output_xml)["run_id"]
    rows = client.get(f"/api/runs/{run_id}/tests", params={"status": "fail"}).json()["rows"]
    detail = client.get(f"/api/comparisons/{rows[0]['comparison_id']}").json()
    token = detail["pages"][0]["images"]["diff"]
    response = client.get(f"/api/assets/{token}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert "max-age" in response.headers["cache-control"]


def test_unknown_asset_404(client):
    assert client.get("/api/assets/deadbeef").status_code == 404


def test_asset_outside_roots_403(client, database, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("nope")
    token = asset_token(str(secret))
    database.register_asset(token, str(secret))
    assert client.get(f"/api/assets/{token}").status_code == 403


def test_asset_symlink_escape_403(client, config, database, sidecar_output_xml, tmp_path):
    # A symlink under an allowed root pointing outside must be rejected
    _ingest_via_api(client, sidecar_output_xml)
    root = config.roots[0]
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = Path(root) / "sneaky_link"
    try:
        os.symlink(outside, link)
    except OSError:
        return  # symlinks unsupported on this filesystem
    try:
        token = asset_token(str(link))
        database.register_asset(token, str(link))
        assert client.get(f"/api/assets/{token}").status_code == 403
    finally:
        link.unlink(missing_ok=True)


def test_token_auth_enforced(tmp_path, sidecar_output_xml):
    config = AppConfig(data_dir=tmp_path / "data", token="s3cret")
    config.data_dir.mkdir(parents=True, exist_ok=True)
    database = Database(config.db_path)
    client = TestClient(create_app(config, database))
    assert client.get("/api/runs").status_code == 401
    assert client.get(
        "/api/runs", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(
        "/api/runs", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    database.close()
