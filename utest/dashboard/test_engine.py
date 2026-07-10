"""Embedded-engine tests: mask preview, recompare, caching, degradation."""

import shutil

from helpers import REF_IMAGE, REPO_ROOT, UTESTDATA, run_robot_suite
from test_review import PROMOTION_SUITE, _failing_comparison, _ingest

BEACH_IMAGE = UTESTDATA / "Beach_date.png"


def test_capabilities_endpoint(client):
    body = client.get("/api/capabilities").json()
    assert "capabilities" in body
    assert isinstance(body["ocr_available"], bool)


def test_mask_preview_resolves_area_mask(client, config, tmp_path):
    config.add_root(UTESTDATA)
    response = client.post("/api/mask-preview", json={
        "file": str(BEACH_IMAGE),
        "page": 1,
        "masks": {"page": "all", "type": "area", "location": "top", "percent": 10},
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["page"] == 1
    assert len(body["resolved_areas"]) == 1
    area = body["resolved_areas"][0]
    assert area["y"] == 0
    assert area["height"] == body["image_size"]["height"] // 10
    assert body["dpi"] > 0


def test_mask_preview_pattern_returns_text_boxes(client, config):
    if not client.get("/api/capabilities").json()["ocr_available"]:
        import pytest
        pytest.skip("tesseract not available")
    config.add_root(UTESTDATA)
    response = client.post("/api/mask-preview", json={
        "file": str(BEACH_IMAGE),
        "page": 1,
        "masks": {"page": "all", "type": "pattern",
                  "pattern": "[0-9]{2}-[a-zA-Z]{3}-[0-9]{4}"},
    })
    assert response.status_code == 200, response.text
    areas = response.json()["resolved_areas"]
    assert len(areas) >= 1
    assert all(area["width"] > 0 and area["height"] > 0 for area in areas)


def test_mask_preview_invalid_regex_422(client, config):
    """Mid-typing patterns ("[", "[0-9]{2}[") must yield 422, never a 500."""
    config.add_root(UTESTDATA)
    for pattern in ("[", "[0-9]{2}[", "(unclosed"):
        response = client.post("/api/mask-preview", json={
            "file": str(BEACH_IMAGE),
            "page": 1,
            "masks": {"page": "all", "type": "pattern", "pattern": pattern},
        })
        assert response.status_code == 422, (pattern, response.text)
        assert "Invalid regular expression" in response.json()["detail"]


def test_mask_preview_empty_pattern_422(client, config):
    config.add_root(UTESTDATA)
    response = client.post("/api/mask-preview", json={
        "file": str(BEACH_IMAGE),
        "page": 1,
        "masks": {"page": "all", "type": "pattern", "pattern": ""},
    })
    assert response.status_code == 422


def test_recompare_invalid_regex_422(client, config, tmp_path):
    output_xml, reference, candidate, artifacts = _make_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)
    response = client.post("/api/recompare", json={
        "comparison_id": comparison_id,
        "masks": [{"page": "all", "type": "pattern", "pattern": "["}]})
    assert response.status_code == 422


def test_save_masks_invalid_regex_422(client, config, tmp_path):
    config.add_root(tmp_path)
    response = client.put("/api/masks", json={
        "file": str(tmp_path / "masks.json"),
        "masks": [{"page": "all", "type": "pattern", "pattern": "["}]})
    assert response.status_code == 422
    assert not (tmp_path / "masks.json").exists()


def test_mask_preview_outside_roots_403(client, tmp_path):
    secret = tmp_path / "image.png"
    shutil.copyfile(REF_IMAGE, secret)
    response = client.post("/api/mask-preview", json={
        "file": str(secret), "page": 1, "masks": []})
    assert response.status_code == 403


def test_mask_preview_pattern_degrades_without_ocr(client, config, monkeypatch):
    config.add_root(UTESTDATA)
    app_engine = client.app.state.ctx.engine
    monkeypatch.setattr(type(app_engine), "ocr_available", property(lambda self: False))
    response = client.post("/api/mask-preview", json={
        "file": str(BEACH_IMAGE),
        "page": 1,
        "masks": {"type": "pattern", "pattern": ".*"},
    })
    assert response.status_code == 409
    assert "OCR" in response.json()["detail"]
    # coordinate masks still work without OCR
    response = client.post("/api/mask-preview", json={
        "file": str(BEACH_IMAGE),
        "page": 1,
        "masks": {"type": "coordinates", "x": 0, "y": 0, "width": 10, "height": 10},
    })
    assert response.status_code == 200


def test_recompare_with_region_masks_flips_to_pass(client, config, tmp_path):
    output_xml, reference, candidate, artifacts = _make_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)
    detail = client.get(f"/api/comparisons/{comparison_id}").json()
    regions = detail["pages"][0]["regions"]
    assert regions, "failing page must expose diff regions"

    # create-mask-from-diff: cover every detected region with padding
    masks = [{
        "page": "all", "type": "coordinates",
        "x": max(0, region["x"] - 5), "y": max(0, region["y"] - 5),
        "width": region["width"] + 10, "height": region["height"] + 10,
    } for region in regions]

    response = client.post("/api/recompare", json={
        "comparison_id": comparison_id, "masks": masks})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "PASS"
    assert body["cached"] is False

    # original artifacts untouched, original record unchanged
    assert reference.read_bytes() != candidate.read_bytes()
    detail_after = client.get(f"/api/comparisons/{comparison_id}").json()
    assert detail_after["status"] == "FAIL"

    # identical request is served from cache
    response = client.post("/api/recompare", json={
        "comparison_id": comparison_id, "masks": masks})
    assert response.json()["cached"] is True


def test_recompare_without_mask_still_fails(client, config, tmp_path):
    output_xml, reference, candidate, artifacts = _make_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)
    response = client.post("/api/recompare", json={"comparison_id": comparison_id})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAIL"
    # recompare diff images are served as assets
    failing = next(p for p in body["pages"] if p["status"] == "FAIL")
    asset = client.get(f"/api/assets/{failing['images']['diff']}")
    assert asset.status_code == 200


def test_recompare_batch_reports_would_be_status(client, config, tmp_path):
    output_xml, reference, candidate, artifacts = _make_run(tmp_path)
    run_id = _ingest(client, config, output_xml, artifacts)
    comparison_id = _failing_comparison(client, run_id)
    detail = client.get(f"/api/comparisons/{comparison_id}").json()
    regions = detail["pages"][0]["regions"]
    masks = [{
        "page": "all", "type": "coordinates",
        "x": max(0, r["x"] - 5), "y": max(0, r["y"] - 5),
        "width": r["width"] + 10, "height": r["height"] + 10,
    } for r in regions]
    response = client.post("/api/recompare-batch", json={
        "comparison_ids": [comparison_id], "masks": masks})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["status"] == "PASS"


def _make_run(tmp_path):
    from test_review import _make_image_run

    return _make_image_run(tmp_path)
