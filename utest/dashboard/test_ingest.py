"""Ingestion tests against genuine robot-generated output.xml files."""

from doctest_dashboard.ingest import ingest_output_xml


def test_sidecar_run_ingests_fully(database, config, sidecar_output_xml):
    summary = ingest_output_xml(database, config, sidecar_output_xml)
    assert summary.tests == 3
    assert summary.comparisons == 3
    assert summary.sidecar_comparisons == 3
    assert summary.degraded_comparisons == 0


def test_expect_error_wrapper_does_not_hide_failure(database, config, sidecar_output_xml):
    ingest_output_xml(database, config, sidecar_output_xml)
    row = database.query_one(
        "SELECT c.status AS comparison_status, t.status AS test_status "
        "FROM comparisons c JOIN tests t ON c.test_id = t.id "
        "WHERE t.name = 'Failing Comparison Expected'")
    assert row["test_status"] == "PASS"
    assert row["comparison_status"] == "FAIL"


def test_pages_and_regions_stored(database, config, sidecar_output_xml):
    ingest_output_xml(database, config, sidecar_output_xml)
    page = database.query_one(
        "SELECT p.* FROM pages p JOIN comparisons c ON p.comparison_id = c.id "
        "JOIN tests t ON c.test_id = t.id "
        "WHERE t.name = 'Failing Comparison Expected' AND p.status = 'FAIL'")
    assert page is not None
    assert page["score"] is not None
    assert '"x"' in page["regions_json"]
    assert page["review_state"] == "unresolved"


def test_reingest_is_idempotent(database, config, sidecar_output_xml):
    first = ingest_output_xml(database, config, sidecar_output_xml)
    second = ingest_output_xml(database, config, sidecar_output_xml)
    assert first.run_id == second.run_id
    assert len(database.query("SELECT id FROM runs")) == 1
    assert len(database.query("SELECT id FROM tests")) == 3
    assert len(database.query("SELECT id FROM comparisons")) == 3


def test_legacy_run_is_degraded_with_images(database, config, legacy_output_xml):
    summary = ingest_output_xml(database, config, legacy_output_xml)
    assert summary.comparisons == 1
    assert summary.degraded_comparisons == 1
    comparison = database.query_one("SELECT * FROM comparisons")
    assert comparison["degraded"] == 1
    assert comparison["status"] == "FAIL"
    assert comparison["images_json"] is not None
    tokens = database.query("SELECT token FROM assets")
    assert len(tokens) > 0


def test_passing_comparison_state_is_passed(database, config, sidecar_output_xml):
    ingest_output_xml(database, config, sidecar_output_xml)
    row = database.query_one(
        "SELECT c.review_state FROM comparisons c JOIN tests t ON c.test_id = t.id "
        "WHERE t.name = 'Passing Comparison'")
    assert row["review_state"] == "passed"
