import json
from pathlib import Path

import pytest

from DocTest.PdfTest import PdfTest
from DocTest.VisualTest import VisualTest

ROOT_TESTDATA = Path(__file__).parent.parent.resolve() / "testdata"


def _sidecars(directory):
    return sorted((Path(directory) / "doctest_results").glob("*.json"))


def _load(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture
def visual_tester(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return VisualTest(result_json=True)


def test_no_sidecar_by_default(tmp_path, monkeypatch, testdata_dir):
    monkeypatch.chdir(tmp_path)
    visual_tester = VisualTest()
    visual_tester.compare_images(
        str(testdata_dir / 'birthday_1080.png'), str(testdata_dir / 'birthday_1080.png'))
    assert not (tmp_path / "doctest_results").exists()


def test_failing_comparison_emits_fail_sidecar(visual_tester, tmp_path, testdata_dir):
    with pytest.raises(AssertionError):
        visual_tester.compare_images(
            str(testdata_dir / 'birthday_1080.png'),
            str(testdata_dir / 'birthday_1080_date_id.png'))
    sidecars = _sidecars(tmp_path)
    assert len(sidecars) == 1
    result = _load(sidecars[0])
    assert result["schema_version"] == 1
    assert result["status"] == "FAIL"
    assert result["keyword"] == "Compare Images"
    assert result["library"] == "DocTest.VisualTest"
    assert result["reference"]["pages"] == 1
    assert result["timing"]["elapsed_ms"] >= 0
    page = result["pages"][0]
    assert page["status"] == "FAIL"
    assert page["score"] is not None
    assert len(page["diff_regions"]) > 0
    assert all(set(region) == {"x", "y", "width", "height"} for region in page["diff_regions"])
    for kind in ("reference", "candidate", "diff", "combined_with_diff"):
        assert (tmp_path / page["images"][kind]).is_file(), kind


def test_passing_comparison_emits_pass_sidecar(visual_tester, tmp_path, testdata_dir):
    visual_tester.compare_images(
        str(testdata_dir / 'birthday_1080.png'), str(testdata_dir / 'birthday_1080.png'))
    sidecars = _sidecars(tmp_path)
    assert len(sidecars) == 1
    result = _load(sidecars[0])
    assert result["status"] == "PASS"
    page = result["pages"][0]
    assert page["status"] == "PASS"
    assert page["diff_regions"] == []
    assert "diff" not in page["images"]
    assert (tmp_path / page["images"]["reference"]).is_file()
    assert (tmp_path / page["images"]["candidate"]).is_file()


def test_one_sidecar_per_comparison(visual_tester, tmp_path, testdata_dir):
    reference = str(testdata_dir / 'birthday_1080.png')
    visual_tester.compare_images(reference, reference)
    visual_tester.compare_images(reference, reference)
    assert len(_sidecars(tmp_path)) == 2


def test_masked_comparison_records_resolved_masks(visual_tester, tmp_path, testdata_dir):
    with pytest.raises(AssertionError):
        visual_tester.compare_images(
            str(testdata_dir / 'sample_1_page.pdf'),
            str(testdata_dir / 'sample_1_page_moved.pdf'),
            placeholder_file=str(testdata_dir / 'pdf_area_mask.json'))
    result = _load(_sidecars(tmp_path)[0])
    assert result["masks"]["placeholder_file"].endswith("pdf_area_mask.json")
    assert len(result["masks"]["abstract"]) == 1
    assert result["masks"]["abstract"][0]["type"] == "area"
    page = result["pages"][0]
    assert len(page["resolved_masks"]) == 1
    assert set(page["resolved_masks"][0]) == {"x", "y", "width", "height"}


def test_multipage_pdf_records_every_page(visual_tester, tmp_path, testdata_dir):
    with pytest.raises(AssertionError):
        visual_tester.compare_images(
            str(testdata_dir / 'sample.pdf'),
            str(ROOT_TESTDATA / 'sample_changed.pdf'))
    result = _load(_sidecars(tmp_path)[0])
    assert result["reference"]["pages"] == 2
    assert len(result["pages"]) == 2
    statuses = {page["page"]: page["status"] for page in result["pages"]}
    assert "FAIL" in statuses.values()
    for page in result["pages"]:
        assert (tmp_path / page["images"]["reference"]).is_file()
        assert (tmp_path / page["images"]["candidate"]).is_file()
        if page["status"] == "FAIL":
            assert (tmp_path / page["images"]["diff"]).is_file()


def test_pabot_index_prefixes_sidecar_name(visual_tester, tmp_path, testdata_dir):
    visual_tester.PABOTQUEUEINDEX = "7"
    reference = str(testdata_dir / 'birthday_1080.png')
    visual_tester.compare_images(reference, reference)
    sidecars = _sidecars(tmp_path)
    assert sidecars[0].name.startswith("7-")


def test_set_result_json_keyword_toggles(tmp_path, monkeypatch, testdata_dir):
    monkeypatch.chdir(tmp_path)
    visual_tester = VisualTest()
    visual_tester.set_result_json(True)
    reference = str(testdata_dir / 'birthday_1080.png')
    visual_tester.compare_images(reference, reference)
    assert len(_sidecars(tmp_path)) == 1
    visual_tester.set_result_json(False)
    visual_tester.compare_images(reference, reference)
    assert len(_sidecars(tmp_path)) == 1


def test_reference_run_promotion_writes_pass_sidecar(tmp_path, monkeypatch, testdata_dir):
    monkeypatch.chdir(tmp_path)
    visual_tester = VisualTest(result_json=True)
    visual_tester.set_reference_run(True)
    reference = tmp_path / 'new_reference.png'
    visual_tester.compare_images(str(reference), str(testdata_dir / 'birthday_1080.png'))
    assert reference.is_file()
    result = _load(_sidecars(tmp_path)[0])
    assert result["status"] == "PASS"
    assert any("Reference run" in note for note in result["notes"])


def test_pdftest_failing_sidecar_is_document_level(tmp_path, monkeypatch, testdata_dir):
    monkeypatch.chdir(tmp_path)
    pdf_tester = PdfTest(result_json=True)
    with pytest.raises(AssertionError):
        pdf_tester.compare_pdf_documents(
            str(testdata_dir / 'sample_1_page.pdf'),
            str(testdata_dir / 'sample_1_page_different_text.pdf'))
    sidecars = _sidecars(tmp_path)
    assert len(sidecars) == 1
    result = _load(sidecars[0])
    assert result["status"] == "FAIL"
    assert result["keyword"] == "Compare Pdf Documents"
    assert result["library"] == "DocTest.PdfTest"
    assert result["pages"] == []
    assert any("[text]" in note for note in result["notes"])


def test_pdftest_passing_sidecar(tmp_path, monkeypatch, testdata_dir):
    monkeypatch.chdir(tmp_path)
    pdf_tester = PdfTest(result_json=True)
    pdf_tester.compare_pdf_documents(
        str(testdata_dir / 'sample_1_page.pdf'), str(testdata_dir / 'sample_1_page.pdf'))
    result = _load(_sidecars(tmp_path)[0])
    assert result["status"] == "PASS"


def test_pdftest_sidecar_carries_structured_facets(tmp_path, monkeypatch, testdata_dir):
    monkeypatch.chdir(tmp_path)
    pdf_tester = PdfTest(result_json=True)
    with pytest.raises(AssertionError):
        pdf_tester.compare_pdf_documents(
            str(testdata_dir / 'sample_1_page.pdf'),
            str(testdata_dir / 'sample_1_page_different_text.pdf'))
    result = _load(_sidecars(tmp_path)[0])
    facets = result["facets"]
    assert facets, "failing PdfTest comparison must expose structured facets"
    facet_names = {facet["facet"] for facet in facets}
    assert "text" in facet_names
    text_facet = next(facet for facet in facets if facet["facet"] == "text")
    assert text_facet["description"]
    assert text_facet["details"]
