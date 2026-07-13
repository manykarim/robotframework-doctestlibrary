"""Canonical performance benchmarks for the render/compare hot paths.

Run explicitly with:  uv run pytest utest/benchmarks --benchmark-only
Under a normal pytest run each benchmark executes once (fast, still a test).
"""

from pathlib import Path

import pytest

from DocTest.DocumentRepresentation import DocumentRepresentation
from DocTest.VisualTest import VisualTest

TESTDATA = Path(__file__).parent.parent / "testdata"


@pytest.mark.benchmark(group="pdf-load")
def test_benchmark_pdf_document_load(benchmark):
    """PDF load: render all pages + text handling at 200 DPI."""

    def _load():
        doc = DocumentRepresentation(str(TESTDATA / "sample.pdf"), dpi=200)
        doc.close()
        return doc.page_count

    pages = benchmark(_load)
    assert pages > 0


@pytest.mark.benchmark(group="visual-compare")
def test_benchmark_visual_compare_moved(benchmark, tmp_path, monkeypatch):
    """Single-page comparison incl. movement detection (worst common case)."""
    monkeypatch.chdir(tmp_path)
    tester = VisualTest()

    def _compare():
        tester.compare_images(
            str(TESTDATA / "sample_1_page.pdf"),
            str(TESTDATA / "sample_1_page_moved.pdf"),
            move_tolerance=20,
        )

    benchmark(_compare)


@pytest.mark.benchmark(group="visual-compare")
def test_benchmark_visual_compare_identical(benchmark, tmp_path, monkeypatch):
    """Best case: identical documents short-circuit."""
    monkeypatch.chdir(tmp_path)
    tester = VisualTest()

    def _compare():
        tester.compare_images(
            str(TESTDATA / "sample_1_page.pdf"),
            str(TESTDATA / "sample_1_page.pdf"),
        )

    benchmark(_compare)
