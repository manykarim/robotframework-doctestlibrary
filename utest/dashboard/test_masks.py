"""Mask I/O round-trip contract tests against the library's own parser."""

import json
import shutil

import pytest

from helpers import REPO_ROOT, UTESTDATA
from doctest_dashboard.masks import dumps_masks, load_mask_file, normalize_masks, save_mask_file

ROOT_TESTDATA = REPO_ROOT / "testdata"

LIBRARY_MASK_FILES = [
    ROOT_TESTDATA / "masks.json",
    UTESTDATA / "area_mask.json",
    ROOT_TESTDATA / "pattern_mask.json",
    ROOT_TESTDATA / "pdf_pattern_mask.json",
    ROOT_TESTDATA / "mask_logo.json",
]


@pytest.mark.parametrize("mask_file", LIBRARY_MASK_FILES, ids=lambda p: p.name)
def test_library_testdata_roundtrip(mask_file, tmp_path):
    """load -> save -> IgnoreAreaManager parses identically to the original."""
    from DocTest.IgnoreAreaManager import IgnoreAreaManager

    original = IgnoreAreaManager(ignore_area_file=str(mask_file)).read_ignore_areas()
    loaded = load_mask_file(mask_file)
    target = tmp_path / mask_file.name
    save_mask_file(target, loaded)
    reparsed = IgnoreAreaManager(ignore_area_file=str(target)).read_ignore_areas()
    assert reparsed == original


@pytest.mark.parametrize("mask_file", LIBRARY_MASK_FILES, ids=lambda p: p.name)
def test_save_is_byte_stable(mask_file, tmp_path):
    """parse -> save -> parse -> save produces identical bytes."""
    loaded = load_mask_file(mask_file)
    first = tmp_path / "first.json"
    save_mask_file(first, loaded)
    second = tmp_path / "second.json"
    save_mask_file(second, load_mask_file(first))
    assert first.read_bytes() == second.read_bytes()


def test_shorthand_import_normalizes_to_area_masks():
    masks = normalize_masks("top:10;bottom:5")
    assert masks == [
        {"page": "all", "type": "area", "location": "top", "percent": "10"},
        {"page": "all", "type": "area", "location": "bottom", "percent": "5"},
    ]


def test_inline_dict_and_json_string_accepted():
    entry = {"page": "all", "type": "coordinates", "x": 1, "y": 2, "width": 3, "height": 4}
    assert normalize_masks(entry) == [entry]
    assert normalize_masks(json.dumps([entry])) == [entry]


def test_invalid_mask_rejected():
    from doctest_dashboard.masks import MaskError

    with pytest.raises(MaskError):
        normalize_masks([{"page": "all"}])  # no type


def test_export_uses_stable_key_order():
    content = dumps_masks([{
        "height": 4, "width": 3, "y": 2, "x": 1,
        "type": "coordinates", "page": "all", "unit": "mm",
    }])
    keys = list(json.loads(content)[0].keys())
    assert keys == ["page", "type", "x", "y", "width", "height", "unit"]


def test_save_creates_backup(tmp_path):
    target = tmp_path / "masks.json"
    save_mask_file(target, normalize_masks("top:10"))
    original_bytes = target.read_bytes()
    save_mask_file(target, normalize_masks("bottom:20"))
    backup = tmp_path / "masks.json.bak"
    assert backup.read_bytes() == original_bytes
    assert b"bottom" in target.read_bytes()


def test_masks_api_roundtrip(client, config, tmp_path):
    config.add_root(tmp_path)
    target = tmp_path / "masks.json"
    shutil.copyfile(ROOT_TESTDATA / "masks.json", target)

    response = client.get("/api/masks", params={"file": str(target)})
    assert response.status_code == 200
    masks = response.json()["masks"]
    assert masks

    response = client.put("/api/masks", json={"file": str(target), "masks": masks})
    assert response.status_code == 200
    assert response.json()["sha256"]
    assert (tmp_path / "masks.json.bak").exists()

    from DocTest.IgnoreAreaManager import IgnoreAreaManager
    reparsed = IgnoreAreaManager(ignore_area_file=str(target)).read_ignore_areas()
    assert reparsed == masks


def test_masks_api_shorthand_import(client, config, tmp_path):
    config.add_root(tmp_path)
    target = tmp_path / "new_masks.json"
    response = client.put("/api/masks", json={"file": str(target), "masks": "top:10;bottom:10"})
    assert response.status_code == 200
    saved = json.loads(target.read_text())
    assert all(entry["type"] == "area" for entry in saved)


def test_masks_api_roots_confinement(client, tmp_path):
    outside = tmp_path / "outside.json"
    response = client.get("/api/masks", params={"file": str(outside)})
    assert response.status_code == 403
    response = client.put("/api/masks", json={"file": str(outside), "masks": "top:10"})
    assert response.status_code == 403


def test_masks_api_invalid_input_422(client, config, tmp_path):
    config.add_root(tmp_path)
    response = client.put("/api/masks", json={
        "file": str(tmp_path / "masks.json"), "masks": [{"page": "all"}]})
    assert response.status_code == 422


@pytest.mark.parametrize("dpi,mm,expected_px", [(200, 25.4, 200), (300, 10, 118), (96, 50.8, 192)])
def test_unit_dpi_fidelity_via_preview(client, config, dpi, mm, expected_px):
    """Editor unit conversion contract: mm masks resolve via the library at
    the rendering DPI exactly as the editor will display them. Uses a PDF
    because the dpi argument controls PDF rendering, while plain images
    carry their own intrinsic DPI."""
    config.add_root(UTESTDATA)
    response = client.post("/api/mask-preview", json={
        "file": str(UTESTDATA / "sample_1_page.pdf"),
        "page": 1,
        "dpi": dpi,
        "masks": {"page": "all", "type": "coordinates",
                  "x": 0, "y": 0, "width": mm, "height": mm, "unit": "mm"},
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dpi"] == dpi
    area = body["resolved_areas"][0]
    assert area["width"] == expected_px
