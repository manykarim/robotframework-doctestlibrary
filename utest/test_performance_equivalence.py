"""Output-equivalence tests for optimize-render-and-compare-hot-paths.

Every optimization in that change must be observationally identical to the
code it replaced; these tests pin that down.
"""

from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest

from DocTest.DocumentRepresentation import DocumentRepresentation, Page


class TestPixmapConversionBitExact:
    """Direct pix.samples conversion must equal the old PNG round-trip."""

    @pytest.mark.parametrize("dpi", [72, 200])
    def test_bit_exact_against_png_roundtrip(self, testdata_dir, dpi):
        with fitz.open(str(testdata_dir / "sample.pdf")) as doc:
            for page_num in range(len(doc)):
                pix = doc.load_page(page_num).get_pixmap(dpi=dpi)
                via_png = cv2.imdecode(
                    np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR
                )
                direct = DocumentRepresentation._pixmap_to_bgr(pix)
                assert np.array_equal(via_png, direct), (
                    f"page {page_num} differs at dpi={dpi}"
                )

    def test_grayscale_pixmap(self, testdata_dir):
        with fitz.open(str(testdata_dir / "sample_1_page.pdf")) as doc:
            pix = doc.load_page(0).get_pixmap(dpi=100, colorspace=fitz.csGRAY)
            via_png = cv2.imdecode(
                np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR
            )
            direct = DocumentRepresentation._pixmap_to_bgr(pix)
            assert np.array_equal(via_png, direct)

    def test_alpha_pixmap(self, testdata_dir):
        with fitz.open(str(testdata_dir / "sample_1_page.pdf")) as doc:
            pix = doc.load_page(0).get_pixmap(dpi=100, alpha=True)
            direct = DocumentRepresentation._pixmap_to_bgr(pix)
            opaque = DocumentRepresentation._pixmap_to_bgr(
                doc.load_page(0).get_pixmap(dpi=100)
            )
            # The PDF background is opaque white either way
            assert direct.shape == opaque.shape


class TestLazyTextExtraction:
    """Lazy pdf_text_* values must equal the former eager extraction."""

    def test_lazy_equals_eager_after_close(self, testdata_dir):
        pdf = str(testdata_dir / "sample_1_page.pdf")
        doc = DocumentRepresentation(pdf)  # fitz handle closed after load
        page = doc.pages[0]

        with fitz.open(pdf) as ref_doc:
            ref_page = ref_doc.load_page(0)
            expected = {
                "text": ref_page.get_text("text", sort=True),
                "dict": ref_page.get_text("dict", sort=True),
                "words": ref_page.get_text("words", sort=True),
                "blocks": ref_page.get_text("blocks", sort=True),
            }

        assert page.pdf_text_data == expected["text"]
        assert page.pdf_text_dict == expected["dict"]
        assert page.pdf_text_words == expected["words"]
        assert page.pdf_text_blocks == expected["blocks"]

    def test_pixel_only_compare_skips_expensive_extraction(
        self, testdata_dir, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        calls = []
        original = Page._extract_pdf_text

        def _spy(page_self, fmt):
            calls.append(fmt)
            return original(page_self, fmt)

        monkeypatch.setattr(Page, "_extract_pdf_text", _spy)

        from DocTest.VisualTest import VisualTest

        tester = VisualTest()
        tester.compare_images(
            str(testdata_dir / "sample_1_page.pdf"),
            str(testdata_dir / "sample_1_page.pdf"),
        )
        assert calls == [], f"pixel-only compare extracted {calls}"

    def test_masked_lazy_dict_is_filtered(self, testdata_dir):
        pdf = str(testdata_dir / "sample_1_page.pdf")
        mask = {"page": "all", "type": "area", "location": "top", "percent": 50}
        masked = DocumentRepresentation(pdf, ignore_area=mask)
        unmasked = DocumentRepresentation(pdf)

        def _span_texts(text_dict):
            return [
                span["text"]
                for block in text_dict.get("blocks", [])
                if block.get("type") == 0
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ]

        masked_spans = _span_texts(masked.pages[0].pdf_text_dict)
        unmasked_spans = _span_texts(unmasked.pages[0].pdf_text_dict)
        assert masked_spans != unmasked_spans, "top-half mask must remove spans"
        assert set(masked_spans).issubset(set(unmasked_spans))

    def test_setter_still_works(self, testdata_dir):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        page = Page(image, page_number=1, dpi=72)
        assert page.pdf_text_dict is None  # non-PDF page: no source, no crash
        page.pdf_text_dict = {"blocks": []}
        assert page.pdf_text_dict == {"blocks": []}
        page.pdf_text_data = "hello"
        assert page.pdf_text_data == "hello"


class TestCountRealDifferencePixelsEquivalence:
    """uint8 morphology must count exactly like the int64 implementation."""

    def test_equivalent_counts(self, testdata_dir):
        from DocTest.VisualTest import count_real_difference_pixels

        ref = cv2.cvtColor(
            cv2.imread(str(testdata_dir / "birthday_1080.png")), cv2.COLOR_BGR2GRAY
        )
        cand = cv2.cvtColor(
            cv2.imread(str(testdata_dir / "birthday_1080_date_id.png")),
            cv2.COLOR_BGR2GRAY,
        )

        kernel = np.ones((3, 3), np.uint8)

        def old_local_range(gray):
            return cv2.dilate(gray, kernel).astype(int) - cv2.erode(gray, kernel).astype(int)

        intensity_threshold, edge_range = 20, 6
        diff = cv2.absdiff(ref, cand)
        differing = diff > intensity_threshold
        on_edge = (old_local_range(ref) > edge_range) & (old_local_range(cand) > edge_range)
        expected = (
            int(differing.sum()),
            int((differing & on_edge).sum()),
            int((differing & ~on_edge).sum()),
        )

        actual = count_real_difference_pixels(
            ref, cand, intensity_threshold=intensity_threshold, edge_range=edge_range
        )
        assert actual == expected


class TestLlmPayloadStillReceivesImages:
    """Conditional diff copies must not starve the LLM payload path."""

    def test_llm_path_gets_diff_images(self, testdata_dir, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        from DocTest.VisualTest import VisualTest

        captured = {}

        def _fake_handle(self, **kwargs):
            captured["differences"] = kwargs.get("differences")
            return None

        monkeypatch.setattr(
            VisualTest, "_handle_llm_for_visual_differences", _fake_handle
        )
        monkeypatch.setattr(
            "DocTest.VisualTest._load_visual_llm_runtime",
            lambda: (None, None, None),
        )

        tester = VisualTest()
        with pytest.raises(AssertionError):
            tester.compare_images(
                str(testdata_dir / "birthday_1080.png"),
                str(testdata_dir / "birthday_1080_date_id.png"),
                llm=True,
            )
        differences = captured["differences"]
        assert differences and differences[0]["absolute_diff"] is not None
