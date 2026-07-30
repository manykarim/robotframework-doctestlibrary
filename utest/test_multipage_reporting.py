"""Multi-page comparisons must report every differing page (issue #98).

All pages were always compared, but the final reporting loop raised on the
first difference, so only one message was ever logged — and it did not name
its page. Users concluded the comparison had stopped at page 1.

No committed fixture pair differs on more than one page, so the fixtures are
built here with PyMuPDF.
"""

from unittest.mock import patch

import fitz
import pytest

from DocTest.VisualTest import VisualTest

FAILURE_MESSAGE = "The compared images are different."


def _make_pdf(path, page_texts):
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page(width=400, height=400)
        page.insert_text((50, 100), text, fontsize=24)
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def three_page_pair(tmp_path):
    """3-page documents differing on pages 1 and 3, identical on page 2."""
    reference = _make_pdf(
        tmp_path / "ref.pdf", ["PAGE ONE ORIGINAL", "PAGE TWO SAME", "PAGE THREE ORIGINAL"]
    )
    candidate = _make_pdf(
        tmp_path / "cand.pdf", ["PAGE ONE CHANGED!!", "PAGE TWO SAME", "PAGE THREE CHANGED!!"]
    )
    return reference, candidate


def _compare_and_capture(reference, candidate, tester=None, **kwargs):
    """Run a comparison, returning (raised_message_or_None, logged_lines)."""
    tester = tester or VisualTest(take_screenshots=False)
    with patch("DocTest.VisualTest.robot_logger") as logger:
        raised = None
        try:
            tester.compare_images(reference, candidate, **kwargs)
        except AssertionError as error:
            raised = str(error)
        lines = [str(call[0][0]) for call in logger.info.call_args_list if call[0]]
    return raised, lines


def test_every_differing_page_is_reported(three_page_pair):
    reference, candidate = three_page_pair

    raised, lines = _compare_and_capture(reference, candidate)

    assert raised == FAILURE_MESSAGE
    diff_lines = [line for line in lines if "Visual differences detected. SSIM" in line]
    assert len(diff_lines) == 2, f"expected one line per differing page, got {diff_lines}"
    assert any(line.startswith("Page 1: ") for line in diff_lines)
    assert any(line.startswith("Page 3: ") for line in diff_lines)
    # page 2 is identical and must not be reported
    assert not any(line.startswith("Page 2: ") for line in diff_lines)


def _summaries(lines):
    return [line for line in lines if line.startswith("Differences detected on")]


def test_summary_line_names_affected_pages(three_page_pair):
    reference, candidate = three_page_pair

    _, lines = _compare_and_capture(reference, candidate)

    summaries = _summaries(lines)
    assert len(summaries) == 1, f"expected exactly one summary line, got {summaries}"
    assert "2 of 3 page(s): 1, 3" in summaries[0]


def test_summary_precedes_the_per_page_details(three_page_pair):
    """A leading summary keeps a large multi-page failure scannable."""
    reference, candidate = three_page_pair

    _, lines = _compare_and_capture(reference, candidate)

    summary_index = next(
        i for i, line in enumerate(lines) if line.startswith("Differences detected on")
    )
    first_detail = next(i for i, line in enumerate(lines) if line.startswith("Page 1: "))
    assert summary_index < first_detail


def test_identical_documents_report_nothing_and_pass(tmp_path):
    reference = _make_pdf(tmp_path / "a.pdf", ["ONE", "TWO", "THREE"])
    candidate = _make_pdf(tmp_path / "b.pdf", ["ONE", "TWO", "THREE"])

    raised, lines = _compare_and_capture(reference, candidate)

    assert raised is None
    assert not [line for line in lines if "differences detected" in line]
    assert any("comparison passed" in line for line in lines)


def test_single_page_failure_message_unchanged(tmp_path):
    """The exception text is asserted verbatim by 39+ tests across the repo."""
    reference = _make_pdf(tmp_path / "one_ref.pdf", ["ORIGINAL"])
    candidate = _make_pdf(tmp_path / "one_cand.pdf", ["CHANGED!!"])

    raised, lines = _compare_and_capture(reference, candidate)

    assert raised == FAILURE_MESSAGE
    assert any(line.startswith("Page 1: ") for line in lines)


def test_single_page_comparison_emits_no_summary(tmp_path):
    """The summary exists to show later pages were compared — pointless for one page."""
    reference = _make_pdf(tmp_path / "s_ref.pdf", ["ORIGINAL"])
    candidate = _make_pdf(tmp_path / "s_cand.pdf", ["CHANGED!!"])

    _, lines = _compare_and_capture(reference, candidate)

    assert _summaries(lines) == []


def test_reference_run_clears_differences_and_still_passes(tmp_path):
    """The `if detected_differences:` guard must let cleared lists through.

    A reference run promotes the candidate and empties the list *after* it was
    populated — the path that would break if the raise were moved back up.
    """
    reference = _make_pdf(tmp_path / "r_ref.pdf", ["ONE", "TWO", "THREE"])
    candidate = _make_pdf(tmp_path / "r_cand.pdf", ["ONE!!", "TWO", "THREE!!"])
    tester = VisualTest(take_screenshots=False)
    tester.set_reference_run(True)

    raised, lines = _compare_and_capture(reference, candidate, tester=tester)

    assert raised is None, "a reference run must not fail"
    assert any("comparison passed" in line for line in lines)
    assert _summaries(lines) == []


def test_barcode_messages_are_not_double_labelled(tmp_path):
    """Barcode messages already name their page; they must not be prefixed."""
    tester = VisualTest(take_screenshots=False)
    barcode_diff = {
        "message": "The barcodes on page 2 differ: reference ['A'], candidate ['B']",
        "page": 2,
        "type": "barcode",
    }

    formatted = tester._format_difference_message(barcode_diff)

    assert formatted == barcode_diff["message"]
    assert formatted.count("page 2") == 1
    assert not formatted.startswith("Page 2:")
    # ...but it still contributes its page to the summary
    assert tester._difference_page_number(barcode_diff) == 2
