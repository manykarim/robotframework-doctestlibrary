"""Review workflow tests: accept/reject/bundle/reset on real ingested runs."""

import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from helpers import CAND_IMAGE, REF_IMAGE, REPO_ROOT, run_robot_suite
from doctest_dashboard.ingest import ingest_output_xml

ROOT_TESTDATA = REPO_ROOT / "testdata"

PROMOTION_SUITE = """
*** Settings ***
Library    DocTest.VisualTest    result_json=true    take_screenshots=false

*** Test Cases ***
Comparison To Review
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images    {reference}    {candidate}
"""

PDF_PROMOTION_SUITE = """
*** Settings ***
Library    DocTest.VisualTest    result_json=true    take_screenshots=false

*** Test Cases ***
Pdf Comparison To Review
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images    {reference}    {candidate}
"""


def _make_image_run(base_dir: Path, candidate_src: Path = CAND_IMAGE):
    artifacts = base_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    reference = artifacts / "reference.png"
    candidate = artifacts / "candidate.png"
    shutil.copyfile(REF_IMAGE, reference)
    shutil.copyfile(candidate_src, candidate)
    suite = PROMOTION_SUITE.format(reference=reference, candidate=candidate)
    output_xml = run_robot_suite(suite, base_dir / "run")
    return output_xml, reference, candidate, artifacts


def _ingest(client, config, output_xml, artifacts):
    config.add_root(artifacts)
    response = client.post("/api/ingest", json={"output_xml": str(output_xml)})
    assert response.status_code == 200
    return response.json()["run_id"]


def _failing_comparison(client, run_id):
    rows = client.get(f"/api/runs/{run_id}/tests", params={"status": "fail"}).json()
    assert rows
    return rows[0]["comparison_id"]


def test_accept_promotes_reference_with_audit(client, config, tmp_path):
    output_xml, reference, candidate, artifacts = _make_image_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)

    prev_bytes = reference.read_bytes()
    response = client.post(f"/api/comparisons/{comparison_id}/accept",
                           json={"actor": "tester", "reason": "intended change"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert reference.read_bytes() == candidate.read_bytes() != prev_bytes
    assert body["prev_sha256"] != body["new_sha256"]

    decisions = client.get(f"/api/comparisons/{comparison_id}/decisions").json()
    assert len(decisions) == 1
    assert decisions[0]["action"] == "accept"
    assert decisions[0]["actor"] == "tester"
    assert decisions[0]["prev_sha256"] and decisions[0]["new_sha256"]

    detail = client.get(f"/api/comparisons/{comparison_id}").json()
    assert detail["review_state"] == "accepted"
    assert all(p["review_state"] == "accepted" for p in detail["pages"] if p["status"] == "FAIL")


def test_accept_matches_reference_run_layout(client, config, tmp_path):
    """Dashboard accept and the library's reference_run produce identical files."""
    output_xml, reference, candidate, artifacts = _make_image_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)
    client.post(f"/api/comparisons/{comparison_id}/accept", json={})

    from DocTest.ReferencePromotion import promote_candidate_to_reference
    reference_run_target = tmp_path / "refrun" / "reference.png"
    promote_candidate_to_reference(str(reference_run_target), str(candidate))
    assert reference.read_bytes() == reference_run_target.read_bytes()


def test_page_accept_on_single_image(client, config, tmp_path):
    output_xml, reference, candidate, artifacts = _make_image_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)
    detail = client.get(f"/api/comparisons/{comparison_id}").json()
    page_id = detail["pages"][0]["id"]
    response = client.post(f"/api/pages/{page_id}/accept", json={})
    assert response.status_code == 200
    assert reference.read_bytes() == candidate.read_bytes()


def test_page_accept_on_multipage_pdf_redirects(client, config, tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    reference = artifacts / "reference.pdf"
    candidate = artifacts / "candidate.pdf"
    shutil.copyfile(ROOT_TESTDATA / "sample.pdf", reference)
    shutil.copyfile(ROOT_TESTDATA / "sample_changed.pdf", candidate)
    suite = PDF_PROMOTION_SUITE.format(reference=reference, candidate=candidate)
    output_xml = run_robot_suite(suite, tmp_path / "run")
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)
    detail = client.get(f"/api/comparisons/{comparison_id}").json()
    assert len(detail["pages"]) == 2
    failing_page = next(p for p in detail["pages"] if p["status"] == "FAIL")

    response = client.post(f"/api/pages/{failing_page['id']}/accept", json={})
    assert response.status_code == 409
    assert "create_mask" in response.json()["detail"]["alternatives"]
    assert reference.read_bytes() != candidate.read_bytes()  # nothing was written

    # document-level accept works and is offered as the alternative
    response = client.post(f"/api/comparisons/{comparison_id}/accept", json={})
    assert response.status_code == 200
    assert reference.read_bytes() == candidate.read_bytes()


def test_accept_outside_roots_403(client, config, tmp_path, database):
    output_xml, reference, candidate, artifacts = _make_image_run(tmp_path)
    # Ingest WITHOUT adding the artifacts dir as a root
    response = client.post("/api/ingest", json={"output_xml": str(output_xml)})
    run_id = response.json()["run_id"]
    comparison_id = _failing_comparison(client, run_id)
    response = client.post(f"/api/comparisons/{comparison_id}/accept", json={})
    assert response.status_code == 403
    assert reference.read_bytes() != candidate.read_bytes()


def test_reject_with_bug_bundle(client, config, tmp_path):
    output_xml, reference, candidate, artifacts = _make_image_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)

    response = client.post(f"/api/comparisons/{comparison_id}/reject",
                           json={"reason": "real bug", "actor": "tester"})
    assert response.status_code == 200
    assert response.json()["review_state"] == "rejected"

    bundle = client.get(f"/api/comparisons/{comparison_id}/bugdata")
    assert bundle.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(bundle.content))
    names = archive.namelist()
    assert "reference/reference.png" in names
    assert "candidate/candidate.png" in names
    assert "comparison_result.json" in names
    assert any(name.startswith("pages/") and "diff" in name for name in names)
    metadata = json.loads(archive.read("metadata.json"))
    assert metadata["review_state"] == "rejected"
    assert metadata["decisions"][0]["reason"] == "real bug"


def test_newer_run_with_changed_candidate_resets_to_unresolved(client, config, tmp_path):
    output_xml, reference, candidate, artifacts = _make_image_run(tmp_path / "v1")
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)
    client.post(f"/api/comparisons/{comparison_id}/accept", json={})

    # Newer run, same identity, different candidate content
    output_xml2, _, _, artifacts2 = _make_image_run(
        tmp_path / "v2", candidate_src=REF_IMAGE.parent / "birthday_1080_noise_001.png")
    run_id2 = _ingest(client, config, output_xml2, artifacts2)
    rows2 = client.get(f"/api/runs/{run_id2}/tests", params={"status": "fail"}).json()
    detail2 = client.get(f"/api/comparisons/{rows2[0]['comparison_id']}").json()
    assert detail2["pages"][0]["review_state"] == "unresolved"

    # The earlier decision is still queryable
    decisions = client.get(f"/api/comparisons/{comparison_id}/decisions").json()
    assert len(decisions) == 1


def test_newer_run_with_same_candidate_inherits_acceptance(client, config, tmp_path):
    output_xml, reference, candidate, artifacts = _make_image_run(tmp_path / "v1")
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)
    detail = client.get(f"/api/comparisons/{comparison_id}").json()
    client.post(f"/api/pages/{detail['pages'][0]['id']}/accept", json={})

    output_xml2, _, _, artifacts2 = _make_image_run(tmp_path / "v2")
    run_id2 = _ingest(client, config, output_xml2, artifacts2)
    rows2 = client.get(f"/api/runs/{run_id2}/tests").json()
    detail2 = client.get(f"/api/comparisons/{rows2[0]['comparison_id']}").json()
    assert detail2["pages"][0]["review_state"] == "accepted"
