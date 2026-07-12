"""CI gate + history + flakiness tests."""

import sys

from doctest_dashboard.cli import main as cli_main

from helpers import REF_IMAGE
from test_review import _failing_comparison, _ingest, _make_image_run


def test_gate_fails_then_passes_after_review(client, config, tmp_path):
    output_xml, reference, candidate, artifacts = _make_image_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    data_dir = str(config.data_dir)

    assert cli_main(["--data-dir", data_dir, "gate", str(run_id)]) == 1
    assert cli_main(["--data-dir", data_dir, "gate", "latest"]) == 1

    comparison_id = _failing_comparison(client, run_id)
    client.post(f"/api/comparisons/{comparison_id}/accept", json={})

    assert cli_main(["--data-dir", data_dir, "gate", str(run_id)]) == 0
    assert cli_main(["--data-dir", data_dir, "gate", "latest"]) == 0


def test_gate_rejected_also_passes(client, config, tmp_path):
    output_xml, *_rest, artifacts = _make_image_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)
    client.post(f"/api/comparisons/{comparison_id}/reject", json={"reason": "bug"})
    assert cli_main(["--data-dir", str(config.data_dir), "gate", str(run_id)]) == 0


def test_gate_unknown_run_and_missing_db_exit_2(client, config, tmp_path):
    assert cli_main(["--data-dir", str(tmp_path / "empty"), "gate", "latest"]) == 2
    output_xml, *_rest, artifacts = _make_image_run(tmp_path)
    _ingest(client, config, output_xml, artifacts)
    assert cli_main(["--data-dir", str(config.data_dir), "gate", "999"]) == 2
    assert cli_main(["--data-dir", str(config.data_dir), "gate", "not-a-number"]) == 2


def test_gate_works_without_dashboard_extra(client, config, tmp_path, monkeypatch):
    """The gate must not trip the [dashboard] dependency guard."""
    output_xml, *_rest, artifacts = _make_image_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    monkeypatch.setitem(sys.modules, "fastapi", None)  # simulate base install
    assert cli_main(["--data-dir", str(config.data_dir), "gate", str(run_id)]) == 1


def test_history_across_runs(client, config, tmp_path):
    xml1, *_r1, artifacts1 = _make_image_run(tmp_path / "v1")
    xml2, *_r2, artifacts2 = _make_image_run(tmp_path / "v2")
    run1 = _ingest(client, config, xml1, artifacts1)
    run2 = _ingest(client, config, xml2, artifacts2)
    comparison2 = _failing_comparison(client, run2)

    body = client.get(f"/api/comparisons/{comparison2}/history").json()
    history = body["history"]
    assert len(history) == 2
    assert {entry["run_id"] for entry in history} == {run1, run2}
    assert all(entry["status"] == "FAIL" for entry in history)
    assert history[0]["score"] is not None


def test_history_unknown_comparison_404(client):
    assert client.get("/api/comparisons/9999/history").status_code == 404


def _passing_run(base_dir):
    """Same suite/test identity as _make_image_run, but the comparison passes."""
    import shutil
    from helpers import run_robot_suite

    artifacts = base_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    reference = artifacts / "reference.png"
    candidate = artifacts / "candidate.png"
    shutil.copyfile(REF_IMAGE, reference)
    shutil.copyfile(REF_IMAGE, candidate)
    suite = f"""*** Settings ***
Library    DocTest.VisualTest    result_json=true    take_screenshots=false

*** Test Cases ***
Comparison To Review
    Compare Images    {reference}    {candidate}
"""
    return run_robot_suite(suite, base_dir / "run"), artifacts


def test_flaky_lists_flipping_identity(client, config, tmp_path):
    # same suite/test identity: one passing run, one failing run
    xml_pass, artifacts_pass = _passing_run(tmp_path / "pass")
    xml_fail, *_f, artifacts_fail = _make_image_run(tmp_path / "fail")
    _ingest(client, config, xml_pass, artifacts_pass)
    _ingest(client, config, xml_fail, artifacts_fail)

    body = client.get("/api/flaky").json()
    assert len(body["flaky"]) == 1
    entry = body["flaky"][0]
    assert entry["flips"] >= 1
    assert entry["occurrences"] == 2


def test_flaky_stable_identity_not_listed(client, config, tmp_path):
    xml1, *_r1, artifacts1 = _make_image_run(tmp_path / "a")
    xml2, *_r2, artifacts2 = _make_image_run(tmp_path / "b")
    _ingest(client, config, xml1, artifacts1)
    _ingest(client, config, xml2, artifacts2)
    assert client.get("/api/flaky").json()["flaky"] == []
