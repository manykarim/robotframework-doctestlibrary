"""Regression tests for the fix-silent-failures-and-correctness change.

One test (group) per defect from docs/ultradeep-analysis-solution-proposal.md §2.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from DocTest.DocumentRepresentation import DocumentRepresentation, Page
from DocTest.PdfTest import PdfTest
from DocTest.VisualTest import VisualTest


def _sidecars(directory):
    return sorted(
        path for path in (Path(directory) / "doctest_results").glob("*.json")
        if path.name != "run.json")


class TestLlmStringFlagCoercion:
    """§2.1 — Robot string 'False' must disable the LLM flags, not enable them."""

    def _run_compare(self, testdata_dir, monkeypatch, **llm_kwargs):
        calls = []

        def _fake_runtime():
            calls.append(True)
            from DocTest.PdfTest import LLMDependencyError
            raise LLMDependencyError()

        monkeypatch.setattr("DocTest.PdfTest._load_pdf_llm_runtime", _fake_runtime)
        tester = PdfTest()
        with pytest.raises(AssertionError):
            tester.compare_pdf_documents(
                str(testdata_dir / "sample_1_page.pdf"),
                str(testdata_dir / "sample_1_page_different_text.pdf"),
                compare="text",
                **llm_kwargs,
            )
        return calls

    def test_string_false_disables_llm(self, testdata_dir, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        calls = self._run_compare(
            testdata_dir, monkeypatch, llm="False", llm_override="False"
        )
        assert calls == [], "LLM runtime must not be loaded for llm='False'"

    def test_string_true_enables_llm(self, testdata_dir, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        calls = self._run_compare(testdata_dir, monkeypatch, llm="True")
        assert calls, "LLM runtime should be requested for llm='True'"


class TestContainsBarcodes:
    """§2.2 — contains_barcodes must be forwarded and content-checked."""

    def test_flag_is_forwarded_to_documents(self, testdata_dir, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        captured = []
        real_ctor = DocumentRepresentation

        class _Recorder(DocumentRepresentation):
            def __init__(self, *args, **kwargs):
                captured.append(kwargs.get("contains_barcodes"))
                super().__init__(*args, **kwargs)

        monkeypatch.setattr("DocTest.VisualTest.DocumentRepresentation", _Recorder)
        tester = VisualTest()
        tester.compare_images(
            str(testdata_dir / "birthday_left.png"),
            str(testdata_dir / "birthday_left.png"),
            contains_barcodes=True,
        )
        assert captured == [True, True]

    def test_differing_barcode_values_fail(self, testdata_dir, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        injected = iter([
            [{"x": 0, "y": 0, "width": 10, "height": 10, "value": "REF-VALUE"}],
            [{"x": 0, "y": 0, "width": 10, "height": 10, "value": "CAND-VALUE"}],
        ])

        def _fake_identify(page_self):
            page_self.barcodes.extend(next(injected, []))

        monkeypatch.setattr(Page, "identify_barcodes", _fake_identify)
        tester = VisualTest()
        with pytest.raises(AssertionError):
            tester.compare_images(
                str(testdata_dir / "birthday_left.png"),
                str(testdata_dir / "birthday_left.png"),
                contains_barcodes=True,
            )

    def test_equal_barcode_values_pass(self, testdata_dir, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)

        def _fake_identify(page_self):
            page_self.barcodes.append(
                {"x": 0, "y": 0, "width": 10, "height": 10, "value": "SAME"}
            )

        monkeypatch.setattr(Page, "identify_barcodes", _fake_identify)
        tester = VisualTest()
        tester.compare_images(
            str(testdata_dir / "birthday_left.png"),
            str(testdata_dir / "birthday_left.png"),
            contains_barcodes=True,
        )


class TestBlockBasedSsimInversion:
    """§2.3 — an inverted block (SSIM ≈ -1) must fail block-based SSIM."""

    def test_inverted_block_fails(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        other = image.copy()
        other[:32, :32] = 255 - other[:32, :32]

        page = Page(image, page_number=1, dpi=72)
        result, lowest_score = page.block_based_ssim_comparison(
            other, threshold=0.3, block_size=32
        )
        assert result is False
        assert lowest_score < 0, "inverted block must produce a negative SSIM"

    def test_identical_images_pass(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        page = Page(image, page_number=1, dpi=72)
        result, lowest_score = page.block_based_ssim_comparison(
            image.copy(), threshold=0.3, block_size=32
        )
        assert result is True
        assert lowest_score == pytest.approx(1.0)


class TestPrintJobErrors:
    """§2.4/§2.5 — clear errors instead of UnboundLocalError/TypeError."""

    def test_unsupported_type_raises_value_error(self):
        from DocTest.PrintJobTests import compare_print_jobs

        with pytest.raises(ValueError, match="afp"):
            compare_print_jobs("afp", "ref.afp", "cand.afp")
        with pytest.raises(ValueError, match="pclx"):
            compare_print_jobs("pclx", "ref.pcl", "cand.pcl")

    def test_missing_property_is_reported_difference(self):
        from DocTest.PrintJobTests import compare_properties

        reference = SimpleNamespace(
            properties=[{"property": "trailer", "value": ["x"]}]
        )
        candidate = SimpleNamespace(properties=[])
        with pytest.raises(AssertionError, match="different"):
            compare_properties(reference, candidate)

    def test_extra_candidate_property_is_reported_difference(self):
        from DocTest.PrintJobTests import compare_properties

        reference = SimpleNamespace(properties=[])
        candidate = SimpleNamespace(
            properties=[{"property": "trailer", "value": ["x"]}]
        )
        with pytest.raises(AssertionError, match="different"):
            compare_properties(reference, candidate)


class TestEastResizeDimensions:
    """§2.6 — _resize_image must not swap width and height."""

    def test_non_square_dimensions(self):
        from DocTest.Ocr import EastTextExtractor

        extractor = EastTextExtractor.__new__(EastTextExtractor)
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        resized, width, height, ratio_w, ratio_h = extractor._resize_image(
            image, 320, 640
        )
        assert (width, height) == (320, 640)
        assert resized.shape[:2] == (640, 320)
        assert ratio_w == pytest.approx(200 / 320)
        assert ratio_h == pytest.approx(100 / 640)


class TestSidecarMaskNotShadowed:
    """§2.7 — sidecar masks.mask must be the caller's mask, not watermark data."""

    def test_mask_metadata_survives_watermark_loop(
        self, testdata_dir, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        mask = {"page": "all", "type": "area", "location": "top", "percent": 5}
        tester = VisualTest(result_json=True)
        with pytest.raises(AssertionError):
            tester.compare_images(
                str(testdata_dir / "birthday_1080.png"),
                str(testdata_dir / "birthday_1080_date_id.png"),
                mask=mask,
                watermark_file=str(testdata_dir / "watermark.pdf"),
            )
        sidecars = _sidecars(tmp_path)
        assert len(sidecars) == 1
        with open(sidecars[0], encoding="utf-8") as file:
            result = json.load(file)
        assert result["masks"]["mask"] == mask


class TestFitzDocumentsClosed:
    """§2.8 — PdfTest text keywords must close their fitz documents."""

    def _capture_open(self, monkeypatch):
        import DocTest.PdfTest as pdftest_module

        opened = []
        real_open = pdftest_module.fitz.open

        def _open(*args, **kwargs):
            doc = real_open(*args, **kwargs)
            opened.append(doc)
            return doc

        monkeypatch.setattr(pdftest_module.fitz, "open", _open)
        return opened

    def test_pdf_should_contain_strings_closes_doc(self, testdata_dir, monkeypatch):
        opened = self._capture_open(monkeypatch)
        tester = PdfTest()
        with pytest.raises(AssertionError):
            tester.PDF_should_contain_strings(
                ["this text certainly does not exist"],
                str(testdata_dir / "sample_1_page.pdf"),
            )
        assert opened and all(doc.is_closed for doc in opened)

    def test_pdf_should_not_contain_strings_closes_doc(self, testdata_dir, monkeypatch):
        opened = self._capture_open(monkeypatch)
        tester = PdfTest()
        tester.PDF_should_not_contain_strings(
            ["this text certainly does not exist"],
            str(testdata_dir / "sample_1_page.pdf"),
        )
        assert opened and all(doc.is_closed for doc in opened)

    def test_check_text_content_closes_doc(self, testdata_dir, monkeypatch):
        opened = self._capture_open(monkeypatch)
        tester = PdfTest()
        with pytest.raises(AssertionError):
            tester.check_text_content(
                ["this text certainly does not exist"],
                str(testdata_dir / "sample_1_page.pdf"),
            )
        assert opened and all(doc.is_closed for doc in opened)


class TestPclPageNumbersAreIntegers:
    """§2.9 — PCL/PS loaders must use int page numbers so page masks apply."""

    def test_pcl_page_scoped_mask_applies(self, monkeypatch, tmp_path):
        import shutil as shutil_module
        import subprocess as subprocess_module

        pcl_file = Path(__file__).parent.parent / "testdata" / "invoice.pcl"
        rendered = np.full((200, 200, 3), 255, dtype=np.uint8)

        real_which = shutil_module.which
        monkeypatch.setattr(
            shutil_module,
            "which",
            lambda name: "/usr/bin/pcl6" if name.startswith(("pcl", "gpcl")) else real_which(name),
        )

        def _fake_run(args, **kwargs):
            output_arg = next(a for a in args if str(a).startswith("-sOutputFile="))
            pattern = str(output_arg)[len("-sOutputFile="):]
            cv2.imwrite(pattern % 1, rendered)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(subprocess_module, "run", _fake_run)

        doc = DocumentRepresentation(
            str(pcl_file),
            ignore_area={"page": 1, "type": "area", "location": "top", "percent": 10},
        )
        assert doc.pages, "PCL loader produced no pages"
        page = doc.pages[0]
        assert isinstance(page.page_number, int)
        assert page.pixel_ignore_areas, (
            "page-scoped mask must resolve to pixel ignore areas on PCL pages"
        )
