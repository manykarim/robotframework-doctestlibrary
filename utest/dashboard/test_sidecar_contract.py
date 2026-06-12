"""Contract test: the real library's sidecars validate against our models.

Runs the core library directly (no Robot Framework) and parses every
emitted sidecar with the dashboard's pydantic schema. If the library's
``ResultWriter`` and the dashboard's ``models.sidecar`` drift apart, this
fails.
"""

import json
from pathlib import Path

import pytest

from doctest_dashboard.models.sidecar import SUPPORTED_SCHEMA_VERSION, parse_sidecar

from helpers import CAND_IMAGE, PDF_MASK, PDF_REF, REF_IMAGE


@pytest.fixture
def sidecars_from_real_library(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from DocTest.VisualTest import VisualTest

    visual_tester = VisualTest(result_json=True)
    visual_tester.compare_images(str(REF_IMAGE), str(REF_IMAGE))
    with pytest.raises(AssertionError):
        visual_tester.compare_images(str(REF_IMAGE), str(CAND_IMAGE))
    with pytest.raises(AssertionError):
        visual_tester.compare_images(
            str(PDF_REF), str(PDF_REF).replace("sample_1_page", "sample_1_page_moved"),
            placeholder_file=str(PDF_MASK))
    return sorted((tmp_path / "doctest_results").glob("*.json"))


def test_real_sidecars_validate(sidecars_from_real_library, tmp_path):
    assert len(sidecars_from_real_library) == 3
    statuses = []
    for path in sidecars_from_real_library:
        result = parse_sidecar(json.loads(path.read_text(encoding="utf-8")))
        statuses.append(result.status)
        assert result.schema_version == SUPPORTED_SCHEMA_VERSION
        for page in result.pages:
            for image_rel in page.images.values():
                assert (tmp_path / image_rel).is_file()
    assert statuses.count("PASS") == 1
    assert statuses.count("FAIL") == 2


def test_unknown_schema_version_rejected():
    with pytest.raises(ValueError, match="schema_version"):
        parse_sidecar({"schema_version": 99})
