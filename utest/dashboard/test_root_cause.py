"""Region text explanation tests: engine parity, confinement, degraded."""

from doctest_dashboard.ingest import ingest_output_xml

from test_review import _failing_comparison, _ingest, _make_image_run


def _first_region(client, comparison_id):
    detail = client.get(f"/api/comparisons/{comparison_id}").json()
    page = next(p for p in detail["pages"] if p["status"] == "FAIL")
    assert page["regions"]
    return page["page_no"], page["regions"][0]


def test_region_text_explains_changed_text(client, config, tmp_path):
    output_xml, reference, candidate, artifacts = _make_image_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)
    page_no, region = _first_region(client, comparison_id)

    response = client.post(f"/api/comparisons/{comparison_id}/region-text",
                           json={"page_no": page_no, "region": region})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["same"] is False
    assert body["reference_text"] != body["candidate_text"]
    # cached on repeat
    again = client.post(f"/api/comparisons/{comparison_id}/region-text",
                        json={"page_no": page_no, "region": region}).json()
    assert again["cached"] is True


def test_region_text_degraded_409(client, config, database, legacy_output_xml):
    ingest_output_xml(database, config, legacy_output_xml)
    comparison = database.query_one("SELECT id FROM comparisons")
    response = client.post(f"/api/comparisons/{comparison['id']}/region-text",
                           json={"page_no": 1,
                                 "region": {"x": 0, "y": 0, "width": 10, "height": 10}})
    assert response.status_code == 409


def test_region_text_outside_roots_403(client, config, tmp_path):
    output_xml, reference, candidate, artifacts = _make_image_run(tmp_path)
    # ingest WITHOUT adding the artifacts dir as a root
    response = client.post("/api/ingest", json={"output_xml": str(output_xml)})
    run_id = response.json()["run_id"]
    comparison_id = _failing_comparison(client, run_id)
    response = client.post(f"/api/comparisons/{comparison_id}/region-text",
                           json={"page_no": 1,
                                 "region": {"x": 0, "y": 0, "width": 10, "height": 10}})
    assert response.status_code == 403


def test_region_text_page_out_of_range_400(client, config, tmp_path):
    output_xml, reference, candidate, artifacts = _make_image_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)
    response = client.post(f"/api/comparisons/{comparison_id}/region-text",
                           json={"page_no": 99,
                                 "region": {"x": 0, "y": 0, "width": 10, "height": 10}})
    assert response.status_code == 400
