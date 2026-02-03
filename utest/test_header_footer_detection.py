"""Unit tests for HeaderFooterDetector module (ADR-002).

Tests cover repetition-based detection of headers/footers, stripping of
detected lines, digit normalization for page numbers, and the convenience
filter_headers_footers function.
"""

import pytest

from DocTest.HeaderFooterDetector import (
    DetectionResult,
    HeaderFooterConfig,
    _normalize_for_grouping,
    detect_repeating_headers_footers,
    filter_headers_footers,
    strip_detected_headers_footers,
)
from DocTest.PdfStructureModels import (
    DocumentStructure,
    PageStructure,
    StructureExtractionConfig,
    TextBlock,
    TextLine,
    flatten_document_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page(page_number, lines_data, width=612, height=792):
    """Create a PageStructure from line data.

    Args:
        page_number: The 1-based page number.
        lines_data: list of (text, y_top, y_bottom) tuples.
            Each line gets bbox = (0, y_top, width, y_bottom).
        width: Page width in PDF points.
        height: Page height in PDF points.

    Returns:
        A PageStructure suitable for testing.
    """
    text_lines = []
    for i, (text, y_top, y_bottom) in enumerate(lines_data):
        text_lines.append(
            TextLine(
                index=i,
                text=text,
                bbox=(0.0, float(y_top), float(width), float(y_bottom)),
            )
        )
    block = TextBlock(index=0, bbox=(0, 0, width, height), lines=text_lines)
    return PageStructure(
        page_number=page_number, width=width, height=height, blocks=[block]
    )


def _make_doc(*pages):
    """Create a DocumentStructure from PageStructure objects."""
    config = StructureExtractionConfig()
    return DocumentStructure(pages=list(pages), config=config)


# ---------------------------------------------------------------------------
# Normalization helper tests
# ---------------------------------------------------------------------------


class TestNormalizeForGrouping:
    """Tests for the _normalize_for_grouping helper."""

    def test_replaces_single_digit(self):
        assert _normalize_for_grouping("Page 1") == "Page #"

    def test_replaces_multiple_digit_runs(self):
        assert _normalize_for_grouping("Page 1 of 5") == "Page # of #"

    def test_no_digits_unchanged(self):
        assert _normalize_for_grouping("ACME Corp") == "ACME Corp"

    def test_multi_digit_run(self):
        assert _normalize_for_grouping("2024-01-15") == "#-#-#"

    def test_empty_string(self):
        assert _normalize_for_grouping("") == ""

    def test_standalone_page_number(self):
        assert _normalize_for_grouping("42") == "#"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestHeaderFooterConfig:
    """Tests for HeaderFooterConfig properties."""

    def test_detection_disabled_when_scan_height_zero(self):
        """Both scan heights 0 means detection is disabled."""
        config = HeaderFooterConfig(header_scan_height=0, footer_scan_height=0)
        assert config.enabled is False

    def test_config_enabled_with_header_only(self):
        """Detection is enabled when only header_scan_height > 0."""
        config = HeaderFooterConfig(header_scan_height=50, footer_scan_height=0)
        assert config.enabled is True

    def test_config_enabled_with_footer_only(self):
        """Detection is enabled when only footer_scan_height > 0."""
        config = HeaderFooterConfig(header_scan_height=0, footer_scan_height=50)
        assert config.enabled is True

    def test_config_enabled_with_both(self):
        """Detection is enabled when both scan heights > 0."""
        config = HeaderFooterConfig(header_scan_height=50, footer_scan_height=50)
        assert config.enabled is True


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


class TestDetectRepeatingHeadersFooters:
    """Tests for detect_repeating_headers_footers."""

    def test_disabled_config_returns_empty_result(self):
        """When config.enabled is False, detection returns empty result."""
        doc = _make_doc(
            _make_page(1, [("ACME Corp", 10, 25), ("Body text", 100, 115)]),
            _make_page(2, [("ACME Corp", 10, 25), ("More text", 100, 115)]),
        )
        config = HeaderFooterConfig(header_scan_height=0, footer_scan_height=0)
        result = detect_repeating_headers_footers(doc, config)
        assert result.header_keys == frozenset()
        assert result.footer_keys == frozenset()
        assert result.has_detections is False

    def test_detects_identical_header_on_all_pages(self):
        """Identical text in header region on all pages is detected."""
        doc = _make_doc(
            _make_page(1, [("ACME Corp", 10, 25), ("Body page 1", 100, 115)]),
            _make_page(2, [("ACME Corp", 10, 25), ("Body page 2", 100, 115)]),
            _make_page(3, [("ACME Corp", 10, 25), ("Body page 3", 100, 115)]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=2)
        result = detect_repeating_headers_footers(doc, config)
        assert "ACME Corp" in result.header_keys
        assert result.has_detections is True

    def test_does_not_detect_non_repeating_text(self):
        """Unique text in header region across pages is not detected."""
        doc = _make_doc(
            _make_page(1, [("Title A", 10, 25), ("Body 1", 100, 115)]),
            _make_page(2, [("Title B", 10, 25), ("Body 2", 100, 115)]),
            _make_page(3, [("Title C", 10, 25), ("Body 3", 100, 115)]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=2)
        result = detect_repeating_headers_footers(doc, config)
        assert result.header_keys == frozenset()
        assert result.has_detections is False

    def test_detects_header_with_page_numbers(self):
        """Page-number variants normalize to the same key and are detected."""
        doc = _make_doc(
            _make_page(1, [("Page 1 of 5", 10, 25), ("Body 1", 100, 115)]),
            _make_page(2, [("Page 2 of 5", 10, 25), ("Body 2", 100, 115)]),
            _make_page(3, [("Page 3 of 5", 10, 25), ("Body 3", 100, 115)]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=2)
        result = detect_repeating_headers_footers(doc, config)
        assert "Page # of #" in result.header_keys

    def test_respects_repeat_threshold_below(self):
        """Text repeating on fewer pages than threshold is not detected."""
        doc = _make_doc(
            _make_page(1, [("Header", 10, 25), ("Body 1", 100, 115)]),
            _make_page(2, [("Header", 10, 25), ("Body 2", 100, 115)]),
            _make_page(3, [("Header", 10, 25), ("Body 3", 100, 115)]),
            _make_page(4, [("Unique", 10, 25), ("Body 4", 100, 115)]),
            _make_page(5, [("Unique2", 10, 25), ("Body 5", 100, 115)]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=4)
        result = detect_repeating_headers_footers(doc, config)
        # "Header" only on 3 pages, threshold is 4
        assert "Header" not in result.header_keys

    def test_respects_repeat_threshold_at_boundary(self):
        """Text repeating on exactly threshold pages is detected."""
        doc = _make_doc(
            _make_page(1, [("Header", 10, 25), ("Body 1", 100, 115)]),
            _make_page(2, [("Header", 10, 25), ("Body 2", 100, 115)]),
            _make_page(3, [("Header", 10, 25), ("Body 3", 100, 115)]),
            _make_page(4, [("Unique", 10, 25), ("Body 4", 100, 115)]),
            _make_page(5, [("Unique2", 10, 25), ("Body 5", 100, 115)]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=3)
        result = detect_repeating_headers_footers(doc, config)
        assert "Header" in result.header_keys

    def test_single_page_no_detection(self):
        """Single page document never reaches threshold=2."""
        doc = _make_doc(
            _make_page(1, [("Header", 10, 25), ("Body", 100, 115)]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=2)
        result = detect_repeating_headers_footers(doc, config)
        assert result.header_keys == frozenset()
        assert result.has_detections is False

    def test_footer_detection(self):
        """Text in footer region repeating across pages is detected."""
        # Page height = 792, footer_scan_height = 50 -> boundary at 742
        # Lines at y_bottom=770 are past 742 -> in footer region
        doc = _make_doc(
            _make_page(1, [("Body 1", 100, 115), ("Copyright 2024", 755, 770)]),
            _make_page(2, [("Body 2", 100, 115), ("Copyright 2024", 755, 770)]),
            _make_page(3, [("Body 3", 100, 115), ("Copyright 2024", 755, 770)]),
        )
        config = HeaderFooterConfig(footer_scan_height=50, repeat_threshold=2)
        result = detect_repeating_headers_footers(doc, config)
        assert "Copyright #" in result.footer_keys
        assert result.has_detections is True

    def test_header_and_footer_simultaneously(self):
        """Both header and footer can be detected independently."""
        doc = _make_doc(
            _make_page(1, [("ACME Corp", 10, 25), ("Body 1", 400, 415), ("Page 1", 760, 775)]),
            _make_page(2, [("ACME Corp", 10, 25), ("Body 2", 400, 415), ("Page 2", 760, 775)]),
            _make_page(3, [("ACME Corp", 10, 25), ("Body 3", 400, 415), ("Page 3", 760, 775)]),
        )
        config = HeaderFooterConfig(
            header_scan_height=50, footer_scan_height=50, repeat_threshold=2
        )
        result = detect_repeating_headers_footers(doc, config)
        assert "ACME Corp" in result.header_keys
        assert "Page #" in result.footer_keys

    def test_standalone_page_number_detection(self):
        """Standalone page numbers like '1', '2', '3' normalize to '#'."""
        doc = _make_doc(
            _make_page(1, [("1", 10, 25), ("Body 1", 100, 115)]),
            _make_page(2, [("2", 10, 25), ("Body 2", 100, 115)]),
            _make_page(3, [("3", 10, 25), ("Body 3", 100, 115)]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=2)
        result = detect_repeating_headers_footers(doc, config)
        assert "#" in result.header_keys

    def test_threshold_greater_than_page_count(self):
        """When threshold exceeds page count, nothing can be detected."""
        doc = _make_doc(
            _make_page(1, [("Header", 10, 25), ("Body 1", 100, 115)]),
            _make_page(2, [("Header", 10, 25), ("Body 2", 100, 115)]),
            _make_page(3, [("Header", 10, 25), ("Body 3", 100, 115)]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=5)
        result = detect_repeating_headers_footers(doc, config)
        assert result.header_keys == frozenset()
        assert result.has_detections is False

    def test_line_outside_scan_region_not_counted(self):
        """Text at y > header_scan_height is not counted as a header candidate."""
        doc = _make_doc(
            _make_page(1, [("ACME Corp", 60, 75), ("Body 1", 100, 115)]),
            _make_page(2, [("ACME Corp", 60, 75), ("Body 2", 100, 115)]),
            _make_page(3, [("ACME Corp", 60, 75), ("Body 3", 100, 115)]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=2)
        result = detect_repeating_headers_footers(doc, config)
        # y_top=60 >= header_scan_height=50, so not in header region
        assert "ACME Corp" not in result.header_keys


# ---------------------------------------------------------------------------
# Stripping tests
# ---------------------------------------------------------------------------


class TestStripDetectedHeadersFooters:
    """Tests for strip_detected_headers_footers."""

    def test_strips_detected_headers_preserves_body(self):
        """Detected header lines are removed; body lines remain."""
        doc = _make_doc(
            _make_page(1, [("ACME Corp", 10, 25), ("Body line 1", 100, 115), ("Body line 2", 200, 215)]),
            _make_page(2, [("ACME Corp", 10, 25), ("Body line 3", 100, 115)]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=2)
        detection = detect_repeating_headers_footers(doc, config)
        result = strip_detected_headers_footers(doc, detection, config)

        # All body lines preserved
        all_texts = flatten_document_text(result)
        assert "Body line 1" in all_texts
        assert "Body line 2" in all_texts
        assert "Body line 3" in all_texts
        # Header removed
        assert "ACME Corp" not in all_texts

    def test_body_text_matching_header_not_stripped(self):
        """Same text in body region is preserved even if it matches a header key."""
        doc = _make_doc(
            _make_page(1, [("ACME Corp", 10, 25), ("ACME Corp", 400, 415)]),
            _make_page(2, [("ACME Corp", 10, 25), ("Other body", 400, 415)]),
            _make_page(3, [("ACME Corp", 10, 25), ("More body", 400, 415)]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=2)
        detection = detect_repeating_headers_footers(doc, config)
        result = strip_detected_headers_footers(doc, detection, config)

        # Page 1: header "ACME Corp" at y=10 removed, body "ACME Corp" at y=400 preserved
        page1_texts = []
        for block in result.pages[0].blocks:
            for line in block.lines:
                page1_texts.append(line.text)
        assert "ACME Corp" in page1_texts  # The body-region instance survives

    def test_strips_page_number_variants(self):
        """Different page-number variants sharing the same key are all stripped."""
        doc = _make_doc(
            _make_page(1, [("Page 1 of 5", 10, 25), ("Body A", 100, 115)]),
            _make_page(2, [("Page 2 of 5", 10, 25), ("Body B", 100, 115)]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=2)
        detection = detect_repeating_headers_footers(doc, config)
        result = strip_detected_headers_footers(doc, detection, config)

        all_texts = flatten_document_text(result)
        assert "Page 1 of 5" not in all_texts
        assert "Page 2 of 5" not in all_texts
        assert "Body A" in all_texts
        assert "Body B" in all_texts

    def test_re_indexing_after_strip(self):
        """After stripping, remaining lines have contiguous indices starting at 0."""
        doc = _make_doc(
            _make_page(1, [
                ("Header", 10, 25),
                ("Line A", 100, 115),
                ("Line B", 200, 215),
                ("Line C", 300, 315),
            ]),
            _make_page(2, [
                ("Header", 10, 25),
                ("Line D", 100, 115),
            ]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=2)
        detection = detect_repeating_headers_footers(doc, config)
        result = strip_detected_headers_footers(doc, detection, config)

        # Page 1 should have lines indexed 0, 1, 2
        page1_indices = [
            line.index for block in result.pages[0].blocks for line in block.lines
        ]
        assert page1_indices == [0, 1, 2]

        # Page 2 should have line indexed 0
        page2_indices = [
            line.index for block in result.pages[1].blocks for line in block.lines
        ]
        assert page2_indices == [0]

    def test_empty_blocks_removed_after_strip(self):
        """A block whose only line is a header gets removed entirely."""
        # Create a page with two blocks: one with only a header, one with body
        header_line = TextLine(index=0, text="Header", bbox=(0.0, 10.0, 612.0, 25.0))
        body_line = TextLine(index=1, text="Body text", bbox=(0.0, 100.0, 612.0, 115.0))
        header_block = TextBlock(index=0, bbox=(0, 0, 612, 30), lines=[header_line])
        body_block = TextBlock(index=1, bbox=(0, 90, 612, 120), lines=[body_line])
        page1 = PageStructure(page_number=1, width=612, height=792, blocks=[header_block, body_block])

        header_line2 = TextLine(index=0, text="Header", bbox=(0.0, 10.0, 612.0, 25.0))
        body_line2 = TextLine(index=1, text="More text", bbox=(0.0, 100.0, 612.0, 115.0))
        header_block2 = TextBlock(index=0, bbox=(0, 0, 612, 30), lines=[header_line2])
        body_block2 = TextBlock(index=1, bbox=(0, 90, 612, 120), lines=[body_line2])
        page2 = PageStructure(page_number=2, width=612, height=792, blocks=[header_block2, body_block2])

        doc = _make_doc(page1, page2)
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=2)
        detection = detect_repeating_headers_footers(doc, config)
        result = strip_detected_headers_footers(doc, detection, config)

        # Each page should have only 1 block (body_block), header_block removed
        for page in result.pages:
            assert len(page.blocks) == 1
            assert page.blocks[0].lines[0].text != "Header"

    def test_strips_footer_preserves_header_region(self):
        """Footer stripping does not affect header-region text."""
        doc = _make_doc(
            _make_page(1, [("Title", 10, 25), ("Body", 400, 415), ("Footer", 760, 775)]),
            _make_page(2, [("Title", 10, 25), ("Body 2", 400, 415), ("Footer", 760, 775)]),
            _make_page(3, [("Title", 10, 25), ("Body 3", 400, 415), ("Footer", 760, 775)]),
        )
        # Only detect footer, not header
        config = HeaderFooterConfig(header_scan_height=0, footer_scan_height=50, repeat_threshold=2)
        detection = detect_repeating_headers_footers(doc, config)
        result = strip_detected_headers_footers(doc, detection, config)

        all_texts = flatten_document_text(result)
        # Footer should be removed
        assert "Footer" not in all_texts
        # Header-region text preserved (not scanned as header since header_scan_height=0)
        assert all_texts.count("Title") == 3

    def test_no_detections_returns_original_structure(self):
        """When detection has no results, strip returns the original structure."""
        doc = _make_doc(
            _make_page(1, [("Unique A", 10, 25), ("Body", 100, 115)]),
            _make_page(2, [("Unique B", 10, 25), ("Body 2", 100, 115)]),
        )
        detection = DetectionResult(header_keys=frozenset(), footer_keys=frozenset())
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=2)
        result = strip_detected_headers_footers(doc, detection, config)
        assert result is doc  # Identity check: same object returned


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------


class TestFilterHeadersFooters:
    """Tests for the filter_headers_footers convenience function."""

    def test_filter_headers_footers_end_to_end(self):
        """filter_headers_footers produces same result as detect + strip."""
        doc = _make_doc(
            _make_page(1, [("ACME Corp", 10, 25), ("Body 1", 100, 115), ("Page 1", 760, 775)]),
            _make_page(2, [("ACME Corp", 10, 25), ("Body 2", 100, 115), ("Page 2", 760, 775)]),
            _make_page(3, [("ACME Corp", 10, 25), ("Body 3", 100, 115), ("Page 3", 760, 775)]),
        )
        config = HeaderFooterConfig(
            header_scan_height=50, footer_scan_height=50, repeat_threshold=2
        )

        # Manual two-step
        detection = detect_repeating_headers_footers(doc, config)
        expected = strip_detected_headers_footers(doc, detection, config)

        # Convenience one-step
        actual = filter_headers_footers(doc, config)

        # Compare text content
        assert flatten_document_text(actual) == flatten_document_text(expected)

    def test_filter_disabled_returns_same_object(self):
        """When config.enabled is False, the exact same object is returned."""
        doc = _make_doc(
            _make_page(1, [("Header", 10, 25), ("Body", 100, 115)]),
            _make_page(2, [("Header", 10, 25), ("Body 2", 100, 115)]),
        )
        config = HeaderFooterConfig(header_scan_height=0, footer_scan_height=0)
        result = filter_headers_footers(doc, config)
        assert result is doc


# ---------------------------------------------------------------------------
# Key scenario: page without header content preserved
# ---------------------------------------------------------------------------


class TestPageWithoutHeaderContentPreserved:
    """The key scenario: a page that lacks the repeating header must not
    have its body text incorrectly removed."""

    def test_page_without_header_content_preserved(self):
        """Page 2 has 'HEADER' but page 3 starts with different body text at
        the same y-position. That body text must NOT be removed."""
        doc = _make_doc(
            _make_page(1, [
                ("HEADER", 10, 25),
                ("Body page 1", 100, 115),
            ]),
            _make_page(2, [
                ("HEADER", 10, 25),
                ("Body page 2", 100, 115),
            ]),
            _make_page(3, [
                # No header line -- body text starts at y=10, same as header
                ("Important content", 10, 25),
                ("Body page 3", 100, 115),
            ]),
        )
        config = HeaderFooterConfig(header_scan_height=50, repeat_threshold=2)
        detection = detect_repeating_headers_footers(doc, config)

        # "HEADER" detected as header key
        assert "HEADER" in detection.header_keys

        result = strip_detected_headers_footers(doc, detection, config)
        all_texts = flatten_document_text(result)

        # "HEADER" removed from pages 1 and 2
        assert "HEADER" not in all_texts
        # "Important content" on page 3 preserved (different key)
        assert "Important content" in all_texts
        # All body text preserved
        assert "Body page 1" in all_texts
        assert "Body page 2" in all_texts
        assert "Body page 3" in all_texts

    def test_page_without_footer_content_preserved(self):
        """Symmetric case: a page missing the footer has its body text at
        the bottom preserved."""
        doc = _make_doc(
            _make_page(1, [
                ("Body 1", 100, 115),
                ("FOOTER", 760, 775),
            ]),
            _make_page(2, [
                ("Body 2", 100, 115),
                ("FOOTER", 760, 775),
            ]),
            _make_page(3, [
                ("Body 3", 100, 115),
                # Different text in footer region
                ("Final remarks", 760, 775),
            ]),
        )
        config = HeaderFooterConfig(footer_scan_height=50, repeat_threshold=2)
        detection = detect_repeating_headers_footers(doc, config)

        assert "FOOTER" in detection.footer_keys

        result = strip_detected_headers_footers(doc, detection, config)
        all_texts = flatten_document_text(result)

        assert "FOOTER" not in all_texts
        assert "Final remarks" in all_texts
        assert "Body 1" in all_texts
        assert "Body 2" in all_texts
        assert "Body 3" in all_texts
