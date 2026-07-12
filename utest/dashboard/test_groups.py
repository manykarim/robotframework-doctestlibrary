"""Similarity-grouping tests: key determinism, groups endpoint, group accept."""

import shutil

from helpers import CAND_IMAGE, REF_IMAGE, UTESTDATA, run_robot_suite

from test_review import _ingest

DIFFERENT_CANDIDATE = UTESTDATA / "birthday_1080_noise_001.png"

SUITE_HEADER = """*** Settings ***
Library    DocTest.VisualTest    result_json=true    take_screenshots=false

*** Test Cases ***
"""


def _grouping_run(base_dir, pairs):
    """Robot run with one failing comparison per (ref_src, cand_src) pair."""
    artifacts = base_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    lines = [SUITE_HEADER]
    created = []
    for index, (ref_src, cand_src) in enumerate(pairs, start=1):
        reference = artifacts / f"ref_{index}.png"
        candidate = artifacts / f"cand_{index}.png"
        shutil.copyfile(ref_src, reference)
        shutil.copyfile(cand_src, candidate)
        created.append((reference, candidate))
        lines.append(
            f"Comparison {index}\n"
            "    Run Keyword And Expect Error    The compared images are different.\n"
            f"    ...    Compare Images    {reference}    {candidate}\n")
    output_xml = run_robot_suite("\n".join(lines), base_dir / "run")
    return output_xml, artifacts, created


def test_identical_diffs_share_group_key(client, config, database, tmp_path):
    output_xml, artifacts, _ = _grouping_run(
        tmp_path,
        [(REF_IMAGE, CAND_IMAGE), (REF_IMAGE, CAND_IMAGE), (REF_IMAGE, DIFFERENT_CANDIDATE)])
    _ingest(client, config, output_xml, artifacts)

    keys = [row["group_key"] for row in database.query(
        "SELECT group_key FROM comparisons WHERE status = 'FAIL' ORDER BY id")]
    assert len(keys) == 3
    assert keys[0] is not None
    assert keys[0] == keys[1], "identical differences must share a key"
    assert keys[2] != keys[0], "different differences must not group"


def test_passing_and_degraded_have_no_group_key(client, config, database,
                                                sidecar_output_xml, legacy_output_xml):
    from doctest_dashboard.ingest import ingest_output_xml

    ingest_output_xml(database, config, sidecar_output_xml)
    ingest_output_xml(database, config, legacy_output_xml)
    rows = database.query("SELECT status, degraded, group_key FROM comparisons")
    for row in rows:
        if row["status"] == "PASS" or row["degraded"]:
            assert row["group_key"] is None


def test_groups_endpoint_and_group_accept(client, config, tmp_path):
    output_xml, artifacts, created = _grouping_run(
        tmp_path,
        [(REF_IMAGE, CAND_IMAGE), (REF_IMAGE, CAND_IMAGE), (REF_IMAGE, DIFFERENT_CANDIDATE)])
    run_id = _ingest(client, config, output_xml, artifacts)

    body = client.get(f"/api/runs/{run_id}/groups").json()
    assert len(body["groups"]) == 1
    group = body["groups"][0]
    assert group["count"] == 2
    assert group["thumbnail"]
    assert body["ungrouped"] == 1
    member_ids = [member["comparison_id"] for member in group["members"]]

    response = client.post("/api/comparisons/accept-batch", json={"ids": member_ids})
    assert response.status_code == 200
    assert len(response.json()["accepted"]) == 2
    # both group members' baselines were promoted
    for reference, candidate in created[:2]:
        assert reference.read_bytes() == candidate.read_bytes()

    after = client.get(f"/api/runs/{run_id}/groups").json()
    assert after["groups"] == []
    assert after["ungrouped"] == 1


def test_individually_resolved_member_leaves_group(client, config, tmp_path):
    output_xml, artifacts, created = _grouping_run(
        tmp_path, [(REF_IMAGE, CAND_IMAGE), (REF_IMAGE, CAND_IMAGE)])
    run_id = _ingest(client, config, output_xml, artifacts)
    body = client.get(f"/api/runs/{run_id}/groups").json()
    member_ids = [m["comparison_id"] for m in body["groups"][0]["members"]]

    client.post(f"/api/comparisons/{member_ids[0]}/accept", json={})
    after = client.get(f"/api/runs/{run_id}/groups").json()
    # a group of one is folded into the ungrouped remainder
    assert after["groups"] == []
    assert after["ungrouped"] == 1


def test_groups_unknown_run_404(client):
    assert client.get("/api/runs/999/groups").status_code == 404
