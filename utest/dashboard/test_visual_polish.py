"""dashboard-visual-polish: thumbnails, run/comparison labels, rename API."""

import json

import pytest

from doctest_dashboard.ingest import ingest_output_xml
from helpers import CAND_IMAGE, REF_IMAGE, run_robot_suite

NAMED_SUITE = f"""
*** Settings ***
Library    DocTest.VisualTest    result_json=true    take_screenshots=false

*** Test Cases ***
Some Test Name
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images    {REF_IMAGE}    {CAND_IMAGE}    name=Invoice header
"""

# Same labeled comparison under a DIFFERENT test name: the label must keep
# the identity stable so history joins across the rename.
RENAMED_SUITE = NAMED_SUITE.replace("Some Test Name", "A Renamed Test")


@pytest.fixture(scope="session")
def named_output_xml(tmp_path_factory):
    return run_robot_suite(NAMED_SUITE, tmp_path_factory.mktemp("named_run"))


@pytest.fixture(scope="session")
def renamed_output_xml(tmp_path_factory):
    return run_robot_suite(RENAMED_SUITE, tmp_path_factory.mktemp("renamed_run"))


def test_run_name_includes_source_folder(database, config, sidecar_output_xml):
    ingest_output_xml(database, config, sidecar_output_xml)
    run = database.query_one("SELECT name FROM runs")
    suite, _, folder = run["name"].partition(" · ")
    assert suite == "Suite"
    assert folder == sidecar_output_xml.parent.name


def test_grid_thumbnail_prefers_highlighted_thumb(database, config, sidecar_output_xml):
    ingest_output_xml(database, config, sidecar_output_xml)
    rows, _ = database.list_grid(1)
    failing = [r for r in rows if r["status"] == "FAIL"]
    assert failing
    for row in failing:
        page = database.query_one(
            "SELECT images_json FROM pages WHERE comparison_id = ? AND status = 'FAIL' "
            "ORDER BY page_no LIMIT 1", (row["comparison_id"],))
        images = json.loads(page["images_json"])
        assert "thumb" in images, "v1.1 sidecars must ship a thumb rendering"
        assert row["thumbnail"] == images["thumb"]


def test_sidecar_name_becomes_label_and_identity(database, config, named_output_xml):
    ingest_output_xml(database, config, named_output_xml)
    row = database.query_one("SELECT label, identity FROM comparisons")
    assert row["label"] == "Invoice header"
    assert row["identity"] == "name::Invoice header"


def test_label_identity_survives_test_rename(database, config, named_output_xml,
                                             renamed_output_xml):
    ingest_output_xml(database, config, named_output_xml)
    ingest_output_xml(database, config, renamed_output_xml)
    identities = {r["identity"] for r in database.query("SELECT identity FROM comparisons")}
    assert identities == {"name::Invoice header"}
    first = database.query_one("SELECT id FROM comparisons ORDER BY id LIMIT 1")
    history = database.comparison_history(first["id"])
    assert len(history) == 2


def test_rename_run_roundtrip(client, sidecar_output_xml):
    client.post("/api/ingest", json={"output_xml": str(sidecar_output_xml)})
    run_id = client.get("/api/runs").json()["runs"][0]["id"]

    response = client.patch(f"/api/runs/{run_id}", json={"label": "  Release 1.4 smoke  "})
    assert response.status_code == 200
    assert response.json()["label"] == "Release 1.4 smoke"
    run = client.get("/api/runs").json()["runs"][0]
    assert run["label"] == "Release 1.4 smoke"
    assert " · " in run["name"]  # original name kept as fallback metadata

    # empty label clears back to the derived name
    assert client.patch(f"/api/runs/{run_id}", json={"label": ""}).json()["label"] is None
    assert client.get("/api/runs").json()["runs"][0]["label"] is None


def test_rename_missing_run_404(client):
    assert client.patch("/api/runs/999", json={"label": "x"}).status_code == 404


def test_comparison_detail_names_the_test(client, sidecar_output_xml):
    client.post("/api/ingest", json={"output_xml": str(sidecar_output_xml)})
    grid = client.get("/api/runs/1/tests").json()["rows"]
    detail = client.get(f"/api/comparisons/{grid[0]['comparison_id']}").json()
    assert detail["test_name"] == grid[0]["name"]
    assert detail["suite"]
