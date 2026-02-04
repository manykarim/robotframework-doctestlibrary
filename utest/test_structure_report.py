"""Comprehensive unit tests for DocTest.StructureReportBuilder (ADR-003).

Tests cover:
  - Passing results returning empty strings
  - Single difference types (missing, extra, mismatch, geometry)
  - Hunk grouping (adjacent, separated, merge boundary)
  - Context lines with/without reference_texts
  - Summary statistics
  - Document-level and word-level differences
  - Text truncation
  - HTML escaping (XSS safety)
  - Large results with hunk collapse
  - Metadata rendering
  - Plain-text report structure
  - Internal helpers (_classify_diff_type, _group_into_hunks, _escape, _truncate)
"""

import pytest

from DocTest.PdfStructureComparator import (
    DocumentTextDifference,
    DocumentWordDifference,
    LineDifference,
    StructureComparisonResult,
)
from DocTest.StructureReportBuilder import (
    MAX_HUNKS_BEFORE_COLLAPSE,
    MAX_TEXT_DISPLAY_LENGTH,
    ReportMetadata,
    ReportSummary,
    _classify_diff_type,
    _compute_summary,
    _escape,
    _group_into_hunks,
    _truncate,
    build_structure_report,
    build_structure_report_plain_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_passing_result():
    """Return a StructureComparisonResult with passed=True."""
    return StructureComparisonResult()


def _make_result_with_page_diffs(diffs, page=1):
    """Return a failing StructureComparisonResult with the given LineDifferences."""
    result = StructureComparisonResult()
    for d in diffs:
        result.add_difference(d)
    return result


def _make_line_diff(diff_type, *, page=1, ref_text=None, cand_text=None,
                    deltas=None, reference_index=None, candidate_index=None,
                    message=None):
    """Convenience factory for LineDifference."""
    if message is None:
        message = f"Synthetic {diff_type}"
    return LineDifference(
        page=page,
        diff_type=diff_type,
        message=message,
        ref_text=ref_text,
        cand_text=cand_text,
        deltas=deltas,
        reference_index=reference_index,
        candidate_index=candidate_index,
    )


# ===========================================================================
# 1 & 2 - Passing result returns empty string
# ===========================================================================


class TestPassingResult:

    def test_html_report_empty_for_passing_result(self):
        result = _make_passing_result()
        assert result.passed is True
        html = build_structure_report(result)
        assert html == ""

    def test_plain_report_empty_for_passing_result(self):
        result = _make_passing_result()
        plain = build_structure_report_plain_text(result)
        assert plain == ""


# ===========================================================================
# 3-6 - Single differences
# ===========================================================================


class TestSingleDifferences:

    def test_html_report_single_missing_line(self):
        diff = _make_line_diff(
            "missing_line",
            ref_text="vanished line",
            reference_index=0,
        )
        result = _make_result_with_page_diffs([diff])
        html = build_structure_report(result)

        assert "#fdd" in html, "Missing line should use red background #fdd"
        assert "<b>-</b>" in html, "Missing line should display '-' symbol"
        assert "vanished line" in html

    def test_html_report_single_extra_line(self):
        diff = _make_line_diff(
            "extra_line",
            cand_text="new line appeared",
            candidate_index=0,
        )
        result = _make_result_with_page_diffs([diff])
        html = build_structure_report(result)

        assert "#dfd" in html, "Extra line should use green background #dfd"
        assert "<b>+</b>" in html, "Extra line should display '+' symbol"
        assert "new line appeared" in html

    def test_html_report_single_text_mismatch(self):
        diff = _make_line_diff(
            "text_mismatch",
            ref_text="foo",
            cand_text="bar",
            reference_index=0,
            candidate_index=0,
        )
        result = _make_result_with_page_diffs([diff])
        html = build_structure_report(result)

        assert "#ffd" in html, "Text mismatch should use yellow background #ffd"
        assert "ref:" in html, "Text mismatch should show 'ref:' label"
        assert "cand:" in html, "Text mismatch should show 'cand:' label"
        assert "foo" in html
        assert "bar" in html

    def test_html_report_single_geometry_mismatch(self):
        diff = _make_line_diff(
            "geometry_mismatch",
            ref_text="shifted text",
            deltas={"left": 5.0},
            reference_index=0,
        )
        result = _make_result_with_page_diffs([diff])
        html = build_structure_report(result)

        assert "#eee" in html, "Geometry mismatch should use grey background #eee"
        # The delta symbol U+0394
        assert "\u0394" in html or "&#916;" in html or "&Delta;" in html, \
            "Geometry mismatch should display delta symbol"


# ===========================================================================
# 7-9 - Grouping / Hunks
# ===========================================================================


class TestHunkGrouping:

    def test_adjacent_diffs_grouped_into_one_hunk(self):
        """5 consecutive LineDifferences at indices 10-14 produce 1 hunk."""
        diffs = [
            _make_line_diff("missing_line", ref_text=f"line {i}",
                            reference_index=i)
            for i in range(10, 15)
        ]
        result = _make_result_with_page_diffs(diffs)
        html = build_structure_report(result)

        assert "Hunk 1" in html
        assert "Hunk 2" not in html

    def test_separated_diffs_produce_separate_hunks(self):
        """Diffs at indices 5 and 50 produce two separate hunks."""
        diff_a = _make_line_diff("missing_line", ref_text="early",
                                 reference_index=5)
        diff_b = _make_line_diff("extra_line", cand_text="late",
                                 candidate_index=50)
        result = _make_result_with_page_diffs([diff_a, diff_b])
        html = build_structure_report(result)

        assert "Hunk 1" in html
        assert "Hunk 2" in html

    def test_gap_at_merge_boundary(self):
        """context_lines=3: merge_threshold = 2*3+1 = 7.

        Diffs at index 10 and 17 (gap=7) -> merged into 1 hunk.
        Diffs at index 10 and 18 (gap=8) -> 2 separate hunks.
        """
        # Gap = 7 => 1 hunk
        d1 = _make_line_diff("missing_line", ref_text="a", reference_index=10)
        d2 = _make_line_diff("missing_line", ref_text="b", reference_index=17)
        result_merged = _make_result_with_page_diffs([d1, d2])
        html_merged = build_structure_report(result_merged, context_lines=3)
        assert "Hunk 1" in html_merged
        assert "Hunk 2" not in html_merged

        # Gap = 8 => 2 hunks
        d3 = _make_line_diff("missing_line", ref_text="a", reference_index=10)
        d4 = _make_line_diff("missing_line", ref_text="b", reference_index=18)
        result_split = _make_result_with_page_diffs([d3, d4])
        html_split = build_structure_report(result_split, context_lines=3)
        assert "Hunk 1" in html_split
        assert "Hunk 2" in html_split


# ===========================================================================
# 10-11 - Context
# ===========================================================================


class TestContext:

    def test_context_shown_when_texts_provided(self):
        """When reference_texts is provided, context words appear in HTML."""
        ref_texts = [f"word_{i}" for i in range(20)]
        diff = _make_line_diff("missing_line", ref_text="word_10",
                               reference_index=10)
        result = _make_result_with_page_diffs([diff])
        html = build_structure_report(result, reference_texts=ref_texts,
                                      context_lines=3)

        # Context before should include words near index 10
        assert "word_7" in html or "word_8" in html or "word_9" in html, \
            "Context before the diff should be visible"
        # Context after
        assert "word_11" in html or "word_12" in html or "word_13" in html, \
            "Context after the diff should be visible"

    def test_no_context_when_texts_not_provided(self):
        """Without reference_texts, no context divs with '...' appear."""
        diff = _make_line_diff("missing_line", ref_text="gone",
                               reference_index=10)
        result = _make_result_with_page_diffs([diff])
        html = build_structure_report(result, reference_texts=None)

        # The "..." context wrapper should not appear
        # (the only "..." might come from truncation, but there should be
        #  no context div with the pattern ...word...)
        assert "color:#888;\">..." not in html


# ===========================================================================
# 12-13 - Summary statistics
# ===========================================================================


class TestSummaryStatistics:

    def test_summary_counts_correct(self):
        """Mix of diff types yields correct ReportSummary counts."""
        result = StructureComparisonResult()
        result.add_difference(_make_line_diff("missing_line", ref_text="a",
                                              reference_index=0))
        result.add_difference(_make_line_diff("missing_line", ref_text="b",
                                              reference_index=1))
        result.add_difference(_make_line_diff("extra_line", cand_text="c",
                                              candidate_index=2))
        result.add_difference(_make_line_diff("text_mismatch", ref_text="d",
                                              cand_text="e",
                                              reference_index=3))
        result.add_difference(_make_line_diff("geometry_mismatch",
                                              ref_text="f",
                                              deltas={"left": 1.0},
                                              reference_index=4))

        summary = _compute_summary(result)

        assert summary.missing_count == 2
        assert summary.extra_count == 1
        assert summary.mismatch_count == 1
        assert summary.geometry_count == 1
        assert summary.other_count == 0
        assert summary.total_differences == 5

    def test_summary_includes_word_diffs(self):
        """Word differences are counted in summary statistics."""
        result = StructureComparisonResult()
        result.add_word_difference(DocumentWordDifference(
            diff_type="missing_words",
            message="words gone",
            ref_words=["hello"],
            ref_start_index=0,
            ref_end_index=1,
        ))
        result.add_word_difference(DocumentWordDifference(
            diff_type="extra_words",
            message="words added",
            cand_words=["world"],
            cand_start_index=0,
            cand_end_index=1,
        ))
        result.add_word_difference(DocumentWordDifference(
            diff_type="word_mismatch",
            message="words changed",
            ref_words=["old"],
            cand_words=["new"],
            ref_start_index=5,
            ref_end_index=6,
            cand_start_index=5,
            cand_end_index=6,
        ))

        summary = _compute_summary(result)

        assert summary.missing_count == 1
        assert summary.extra_count == 1
        assert summary.mismatch_count == 1
        assert summary.total_differences == 3


# ===========================================================================
# 14-15 - Document-level and word-level
# ===========================================================================


class TestDocumentAndWordLevel:

    def test_document_level_diffs_in_report(self):
        """DocumentTextDifference items produce 'Document (text-only)' section."""
        result = StructureComparisonResult()
        result.add_document_difference(DocumentTextDifference(
            diff_type="missing_text",
            message="Text missing: hello",
            ref_text="hello",
            ref_index=0,
        ))

        html = build_structure_report(result)
        assert "Document (text-only)" in html

    def test_word_level_diffs_in_report(self):
        """DocumentWordDifference items produce 'Document (word-level)' section."""
        result = StructureComparisonResult()
        result.add_word_difference(DocumentWordDifference(
            diff_type="word_mismatch",
            message="Word changed",
            ref_words=["alpha"],
            cand_words=["beta"],
            ref_start_index=0,
            ref_end_index=1,
            cand_start_index=0,
            cand_end_index=1,
        ))

        html = build_structure_report(result)
        assert "Document (word-level)" in html


# ===========================================================================
# 16 - Truncation
# ===========================================================================


class TestTruncation:

    def test_long_text_truncated(self):
        """Diff with 500-char ref_text is truncated in HTML output."""
        long_text = "x" * 500
        diff = _make_line_diff("missing_line", ref_text=long_text,
                               reference_index=0)
        result = _make_result_with_page_diffs([diff])
        html = build_structure_report(result)

        # The full 500-char text should NOT appear in the report
        assert long_text not in html
        # Instead the truncated version with "..." should
        assert "..." in html
        # The output should contain at most MAX_TEXT_DISPLAY_LENGTH chars
        # of the original text (minus 3 for "...")
        truncated = long_text[:MAX_TEXT_DISPLAY_LENGTH - 3] + "..."
        assert _escape(truncated) in html


# ===========================================================================
# 17 - HTML safety
# ===========================================================================


class TestHTMLSafety:

    def test_html_special_chars_escaped(self):
        """XSS payload in diff text is escaped, not rendered raw."""
        xss = "<script>alert('xss')</script>"
        diff = _make_line_diff("missing_line", ref_text=xss,
                               reference_index=0)
        result = _make_result_with_page_diffs([diff])
        html = build_structure_report(result)

        assert "<script>" not in html, "Raw <script> tag must not appear"
        assert "&lt;script&gt;" in html, "Escaped script tag should appear"


# ===========================================================================
# 18 - Large results
# ===========================================================================


class TestLargeResults:

    def test_many_hunks_collapsed(self):
        """60+ differences at widely separated indices trigger collapse notice."""
        # Create diffs at indices 0, 1000, 2000, ..., separated enough to be
        # distinct hunks.  We need > MAX_HUNKS_BEFORE_COLLAPSE hunks.
        count = MAX_HUNKS_BEFORE_COLLAPSE + 10
        diffs = [
            _make_line_diff("missing_line", ref_text=f"line_{i}",
                            reference_index=i * 1000)
            for i in range(count)
        ]
        result = _make_result_with_page_diffs(diffs)
        html = build_structure_report(result)

        assert "more hunk" in html.lower(), \
            "Report should mention collapsed hunks when count exceeds limit"


# ===========================================================================
# 19-21 - Metadata
# ===========================================================================


class TestMetadata:

    def test_metadata_in_html_report(self):
        """Provided ReportMetadata appears in the HTML report."""
        meta = ReportMetadata(
            reference_name="ref.pdf",
            candidate_name="cand.pdf",
            comparison_mode="structure",
        )
        diff = _make_line_diff("missing_line", ref_text="a",
                               reference_index=0)
        result = _make_result_with_page_diffs([diff])
        html = build_structure_report(result, metadata=meta)

        assert "ref.pdf" in html
        assert "cand.pdf" in html
        assert "structure" in html

    def test_metadata_with_exclusions(self):
        """ReportMetadata with exclusions lists them in the report."""
        meta = ReportMetadata(
            reference_name="r.pdf",
            candidate_name="c.pdf",
            comparison_mode="full",
            exclusions_applied=["header", "footer"],
        )
        diff = _make_line_diff("extra_line", cand_text="x",
                               candidate_index=0)
        result = _make_result_with_page_diffs([diff])
        html = build_structure_report(result, metadata=meta)

        assert "header" in html
        assert "footer" in html
        assert "Exclusions" in html

    def test_no_metadata_still_works(self):
        """metadata=None does not cause errors and still generates a report."""
        diff = _make_line_diff("missing_line", ref_text="a",
                               reference_index=0)
        result = _make_result_with_page_diffs([diff])
        html = build_structure_report(result, metadata=None)

        assert len(html) > 0
        assert "PDF Structure Comparison Report" in html


# ===========================================================================
# 22-23 - Plain text report
# ===========================================================================


class TestPlainTextReport:

    def test_plain_text_has_structure(self):
        """Plain text report contains section delimiters."""
        diff = _make_line_diff("missing_line", ref_text="gone",
                               reference_index=0)
        result = _make_result_with_page_diffs([diff])
        plain = build_structure_report_plain_text(result)

        assert "=" * 60 in plain, "Plain text should have '===' section headers"
        assert "-" * 60 in plain, "Plain text should have '---' separators"
        assert "PDF Structure Comparison Report" in plain

    def test_plain_text_has_diff_symbols(self):
        """Plain text report uses -, +, ~ symbols for diff categories."""
        result = StructureComparisonResult()
        result.add_difference(_make_line_diff(
            "missing_line", ref_text="removed", reference_index=0))
        result.add_difference(_make_line_diff(
            "extra_line", cand_text="added", candidate_index=1))
        result.add_difference(_make_line_diff(
            "text_mismatch", ref_text="old", cand_text="new",
            reference_index=2, candidate_index=2))

        plain = build_structure_report_plain_text(result)

        # Check for the diff marker symbols
        assert '  - "removed"' in plain, "Missing line should use '-' symbol"
        assert '  + "added"' in plain, "Extra line should use '+' symbol"
        assert '  ~ ref:' in plain, "Mismatch should use '~' symbol"


# ===========================================================================
# 24 - _classify_diff_type helper
# ===========================================================================


class TestClassifyDiffType:

    @pytest.mark.parametrize("diff_type,expected", [
        ("missing_line", "missing"),
        ("missing_text", "missing"),
        ("missing_page", "missing"),
        ("missing_words", "missing"),
        ("extra_line", "extra"),
        ("extra_text", "extra"),
        ("extra_page", "extra"),
        ("extra_words", "extra"),
        ("text_mismatch", "mismatch"),
        ("word_mismatch", "mismatch"),
        ("geometry_mismatch", "geometry"),
        ("block_count_mismatch", "other"),
        ("unknown_type", "other"),
    ])
    def test_classify_diff_types(self, diff_type, expected):
        assert _classify_diff_type(diff_type) == expected


# ===========================================================================
# 25-26 - _group_into_hunks helper
# ===========================================================================


class TestGroupIntoHunks:

    def test_group_into_hunks_empty(self):
        """Empty list of differences produces empty hunk list."""
        hunks = _group_into_hunks([], context_lines=3)
        assert hunks == []

    def test_group_into_hunks_single(self):
        """Single difference produces exactly one hunk."""
        diff = _make_line_diff("missing_line", ref_text="solo",
                               reference_index=5)
        hunks = _group_into_hunks([diff], context_lines=3)

        assert len(hunks) == 1
        assert hunks[0]["start_index"] == 5
        assert hunks[0]["end_index"] == 5
        assert len(hunks[0]["differences"]) == 1

    def test_group_into_hunks_with_source_texts(self):
        """Hunks include context_before and context_after when source_texts given."""
        texts = [f"line_{i}" for i in range(20)]
        diff = _make_line_diff("missing_line", ref_text="line_10",
                               reference_index=10)
        hunks = _group_into_hunks([diff], context_lines=2, source_texts=texts)

        assert len(hunks) == 1
        # context_before: texts[8:10] = ["line_8", "line_9"]
        assert hunks[0]["context_before"] == ["line_8", "line_9"]
        # context_after: texts[11:13] = ["line_11", "line_12"]
        assert hunks[0]["context_after"] == ["line_11", "line_12"]

    def test_group_into_hunks_multiple_merged(self):
        """Adjacent diffs within merge_threshold form a single hunk."""
        diffs = [
            _make_line_diff("missing_line", ref_text="a", reference_index=10),
            _make_line_diff("missing_line", ref_text="b", reference_index=11),
            _make_line_diff("missing_line", ref_text="c", reference_index=12),
        ]
        hunks = _group_into_hunks(diffs, context_lines=3)
        assert len(hunks) == 1

    def test_group_into_hunks_multiple_separated(self):
        """Diffs far apart form separate hunks."""
        diffs = [
            _make_line_diff("missing_line", ref_text="a", reference_index=0),
            _make_line_diff("missing_line", ref_text="b", reference_index=100),
        ]
        hunks = _group_into_hunks(diffs, context_lines=3)
        assert len(hunks) == 2
        assert hunks[0]["start_index"] == 0
        assert hunks[1]["start_index"] == 100


# ===========================================================================
# Additional edge cases for internal helpers
# ===========================================================================


class TestEscapeHelper:

    def test_escape_ampersand(self):
        assert _escape("a & b") == "a &amp; b"

    def test_escape_angle_brackets(self):
        assert _escape("<div>") == "&lt;div&gt;"

    def test_escape_quotes(self):
        assert _escape('"hello"') == "&quot;hello&quot;"

    def test_escape_non_string(self):
        """_escape should handle non-string input via str() conversion."""
        assert _escape(42) == "42"


class TestTruncateHelper:

    def test_truncate_short_text_unchanged(self):
        text = "short"
        assert _truncate(text) == text

    def test_truncate_exact_boundary(self):
        text = "x" * MAX_TEXT_DISPLAY_LENGTH
        assert _truncate(text) == text
        assert "..." not in _truncate(text)

    def test_truncate_one_over_boundary(self):
        text = "x" * (MAX_TEXT_DISPLAY_LENGTH + 1)
        truncated = _truncate(text)
        assert truncated.endswith("...")
        assert len(truncated) == MAX_TEXT_DISPLAY_LENGTH

    def test_truncate_custom_max_length(self):
        text = "abcdefghij"  # 10 chars
        truncated = _truncate(text, max_length=7)
        assert truncated == "abcd..."
        assert len(truncated) == 7


# ===========================================================================
# Comprehensive integration-like tests
# ===========================================================================


class TestReportIntegration:

    def test_report_with_all_difference_types(self):
        """Build a report combining page, document, and word differences."""
        result = StructureComparisonResult()

        # Page-level
        result.add_difference(_make_line_diff(
            "missing_line", ref_text="page_missing", reference_index=0))
        result.add_difference(_make_line_diff(
            "extra_line", cand_text="page_extra", candidate_index=1))
        result.add_difference(_make_line_diff(
            "text_mismatch", ref_text="old", cand_text="new",
            reference_index=2, candidate_index=2))

        # Document-level
        result.add_document_difference(DocumentTextDifference(
            diff_type="missing_text",
            message="doc text missing",
            ref_text="doc_line",
            ref_index=0,
        ))

        # Word-level
        result.add_word_difference(DocumentWordDifference(
            diff_type="word_mismatch",
            message="word changed",
            ref_words=["alpha"],
            cand_words=["beta"],
            ref_start_index=0,
            ref_end_index=1,
            cand_start_index=0,
            cand_end_index=1,
        ))

        meta = ReportMetadata(
            reference_name="ref.pdf",
            candidate_name="cand.pdf",
            comparison_mode="full",
            page_count_ref=3,
            page_count_cand=3,
        )

        html = build_structure_report(result, metadata=meta)
        plain = build_structure_report_plain_text(result, metadata=meta)

        # HTML assertions
        assert "PDF Structure Comparison Report" in html
        assert "Page 1" in html
        assert "Document (text-only)" in html
        assert "Document (word-level)" in html
        assert "ref.pdf" in html
        assert "5" in html  # total differences

        # Plain text assertions
        assert "PDF Structure Comparison Report" in plain
        assert "ref.pdf" in plain
        assert "5 difference(s)" in plain

    def test_multi_page_diffs(self):
        """Differences on multiple pages appear under separate page headers."""
        result = StructureComparisonResult()
        result.add_difference(_make_line_diff(
            "missing_line", page=1, ref_text="p1", reference_index=0))
        result.add_difference(_make_line_diff(
            "extra_line", page=3, cand_text="p3", candidate_index=0))

        html = build_structure_report(result)

        assert "Page 1" in html
        assert "Page 3" in html

    def test_summary_line_entries_in_report(self):
        """result.summary entries appear in the HTML report."""
        result = StructureComparisonResult()
        result.add_difference(_make_line_diff(
            "missing_line", ref_text="x", reference_index=0))
        result.extend_summary("Page count mismatch: ref=2, cand=3")

        html = build_structure_report(result)
        assert "Page count mismatch" in html

        plain = build_structure_report_plain_text(result)
        assert "Page count mismatch" in plain
