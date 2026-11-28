"""Unit tests for PrintJobTests module."""

import os
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import pytest

from DocTest import PrintJobParsing
from DocTest.PrintJobParsing import PclXlError
from DocTest.PrintJobTests import (
    PrintJob,
    check_print_job_property,
    chop,
    compare_print_jobs,
    get_pcl_print_job,
    get_postscript_print_job,
)

TESTDATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "testdata"
)
INVOICE_PS = os.path.join(TESTDATA_DIR, "invoice.ps")
INVOICE_NO_LOGO_PS = os.path.join(TESTDATA_DIR, "invoice_no_logo.ps")
INVOICE_PCL = os.path.join(TESTDATA_DIR, "invoice.pcl")
INVOICE_PCLXL = os.path.join(TESTDATA_DIR, "invoice_diff_format.pcl")
FIXTURES = Path(__file__).parent / "testdata" / "printjobs"


def fx(name):
    return str(FIXTURES / name)


def prop(print_job, name):
    return next(p["value"] for p in print_job.properties if p["property"] == name)


class TestPrintJobFromPrintJobTests:
    """Test cases for the PrintJob class in PrintJobTests module."""

    def test_printjob_initialization(self):
        """Test PrintJob initialization from PrintJobTests."""
        properties = [{"property": "test", "value": "value"}]
        job = PrintJob("pcl", properties)

        assert job.jobtype == "pcl"
        assert job.properties == properties


class TestChopFunction:
    """Test cases for chop utility function."""

    def test_chop_suffix(self):
        """Test chopping suffix from text."""
        result = chop("2h", suffix="(h|H)$")
        assert result == "2"

        result = chop("3X", suffix="(x|X)$")
        assert result == "3"

    def test_chop_prefix(self):
        """Test chopping prefix from text."""
        result = chop("%%Title: Document", prefix="%%")
        assert result == "Title: Document"

    def test_chop_both_prefix_and_suffix(self):
        """Test chopping both prefix and suffix."""
        result = chop("%%Title: Document%%", prefix="%%", suffix="%%$")
        assert result == "Title: Document"

    def test_chop_no_match(self):
        """Test chop when no match found."""
        result = chop("test", suffix="(h|H)$")
        assert result == "test"

        result = chop("test", prefix="%%")
        assert result == "test"

    def test_chop_none_values(self):
        """Test chop with None values."""
        result = chop("test", prefix=None, suffix=None)
        assert result == "test"


class TestGetPclPrintJob:
    """Test cases for get_pcl_print_job function."""

    def test_pcl5_invoice_properties(self):
        job = get_pcl_print_job(INVOICE_PCL)
        assert job.jobtype == "pcl"
        assert [p["property"] for p in job.properties] == ["pcl_commands", "page_count", "language"]
        assert prop(job, "page_count") == 1
        assert prop(job, "language") == ["PCL5"]
        commands = prop(job, "pcl_commands")
        for expected in ({"page": "1", "property": "page_orientation", "value": "0"},
                         {"page": "1", "property": "paper_format", "value": "26"},
                         {"page": "1", "property": "copies", "value": "1"},
                         {"page": "1", "property": "paper_source", "value": "7"}):
            assert expected in commands

    def test_tray_value_above_six_is_reported(self):
        commands = prop(get_pcl_print_job(fx("pcl5_tray7.pcl")), "pcl_commands")
        assert {"page": "1", "property": "paper_source", "value": "7"} in commands

    def test_pclxl_invoice_properties(self):
        job = get_pcl_print_job(INVOICE_PCLXL)
        assert prop(job, "language") == ["PCLXL"]
        assert prop(job, "page_count") == 1
        assert prop(job, "pcl_commands") == [
            {"page": "1", "property": "paper_format", "value": "2"},
            {"page": "1", "property": "paper_source", "value": "1"},
            {"page": "1", "property": "page_orientation", "value": "0"},
            {"page": "1", "property": "duplex", "value": "simplex"},
        ]

    def test_file_is_read_as_bytes(self):
        with patch("builtins.open", mock.mock_open(read_data=b"\x1bE\x1b&l26AText\x0c")) as mock_file:
            job = get_pcl_print_job("test.pcl")
        mock_file.assert_called_once_with("test.pcl", "rb")
        assert prop(job, "page_count") == 1

    @patch("DocTest.PrintJobTests.is_url")
    @patch("DocTest.PrintJobTests.download_file_from_url")
    def test_url_is_downloaded_before_parsing(self, mock_download, mock_is_url):
        mock_is_url.return_value = True
        mock_download.return_value = INVOICE_PCL
        job = get_pcl_print_job("http://example.com/test.pcl")
        mock_download.assert_called_once_with("http://example.com/test.pcl")
        assert prop(job, "page_count") == 1


class TestGetPostscriptPrintJob:
    """Test cases for get_postscript_print_job function using real fixture files."""

    def test_get_postscript_print_job_invoice(self):
        """Test parsing a real PostScript fixture (invoice.ps)."""
        result = get_postscript_print_job(INVOICE_PS)

        assert isinstance(result, PrintJob)
        assert result.jobtype == "postscript"
        assert [p["property"] for p in result.properties] == [
            "pjl_commands", "header", "features", "pages", "trailer", "page_count"]
        assert prop(result, "page_count") == 1
        assert {"property": "BoundingBox", "value": "0 0 596 842"} in prop(result, "header")

    def test_postscript_print_job_pages_parsed(self):
        """Test that page metadata is correctly extracted from the fixture."""
        pages = prop(get_postscript_print_job(INVOICE_PS), "pages")
        assert pages[0] == {"page": "1", "property": "PageBoundingBox", "value": "0 0 595 841"}

    def test_compare_postscript_print_jobs(self):
        """Both fixtures share the same page structure, so comparison should pass."""
        compare_print_jobs("ps", INVOICE_PS, INVOICE_NO_LOGO_PS)

    @patch("DocTest.PrintJobTests.is_url")
    @patch("DocTest.PrintJobTests.download_file_from_url")
    def test_get_postscript_print_job_url(self, mock_download, mock_is_url):
        """Test that URL files are downloaded before parsing."""
        mock_is_url.return_value = True
        mock_download.return_value = INVOICE_PS

        result = get_postscript_print_job("http://example.com/test.ps")

        mock_download.assert_called_once_with("http://example.com/test.ps")
        assert isinstance(result, PrintJob)
        assert result.jobtype == "postscript"

    def test_header_property_check_passes(self):
        job = get_postscript_print_job(fx("ps_01_minimal_dsc.ps"))
        check_print_job_property(job, "header", {"property": "Title", "value": "invoice"})

    @pytest.mark.skipif(PrintJobParsing.find_ghostscript() is None, reason="Ghostscript not installed")
    def test_non_dsc_job_is_counted_by_ghostscript(self):
        assert prop(get_postscript_print_job(fx("ps_07_no_dsc_two_pages.ps")), "page_count") == 2

    def test_non_dsc_job_without_ghostscript_is_an_error(self):
        with mock.patch.object(PrintJobParsing, "find_ghostscript", return_value=None):
            with pytest.raises(AssertionError, match="Ghostscript is required"):
                get_postscript_print_job(fx("ps_07_no_dsc_two_pages.ps"))


def assert_differs(kind, reference, candidate, *, names=(), ignore=None):
    with patch("DocTest.PrintJobTests.robot_logger") as log:
        with pytest.raises(AssertionError, match="The compared print jobs are different."):
            compare_print_jobs(kind, reference, candidate, ignore)
    logged = "\n".join(str(c.args[0]) for c in log.info.call_args_list)
    for name in names:
        assert name in logged.split("Candidate File:")[-1], f"{name} not reported as a difference"


class TestComparePrintJobs:
    """False-pass regressions and comparison semantics."""

    def test_postscript_page_count_difference(self):
        assert_differs("ps", fx("ps_01_minimal_dsc.ps"), fx("ps_09_two_pages.ps"), names=["page_count"])

    def test_pcl_page_count_difference(self):
        assert_differs("pcl", fx("pcl5_one_page.pcl"), fx("pcl5_two_pages.pcl"), names=["page_count"])

    def test_bounding_box_difference(self):
        assert_differs("ps", fx("ps_a4.ps"), fx("ps_letter.ps"), names=["BoundingBox"])

    def test_requested_feature_difference(self):
        assert_differs("ps", fx("ps_duplex_notumble.ps"), fx("ps_duplex_tumble.ps"), names=["*Duplex"])

    def test_pclxl_attribute_difference(self):
        assert_differs("pcl", INVOICE_PCLXL, fx("pclxl_letter.pcl"), names=["paper_format"])

    def test_scrambled_pclxl_is_not_equal(self):
        with pytest.raises(PclXlError):
            compare_print_jobs("pcl", INVOICE_PCLXL, fx("pclxl_scrambled.pcl"))

    def test_language_difference(self):
        assert_differs("pcl", INVOICE_PCL, INVOICE_PCLXL, names=["language"])

    def test_run_specific_fields_are_ignored_by_default(self):
        compare_print_jobs("ps", fx("ps_run_a.ps"), fx("ps_run_b.ps"))

    def test_ignore_properties_extends_defaults(self):
        assert_differs("ps", fx("ps_creator_a.ps"), fx("ps_creator_b.ps"), names=["Creator"])
        compare_print_jobs("ps", fx("ps_creator_a.ps"), fx("ps_creator_b.ps"), "Creator")
        compare_print_jobs("ps", fx("ps_creator_a.ps"), fx("ps_creator_b.ps"), ["creator"])

    def test_ignore_properties_accepts_groups(self):
        compare_print_jobs("ps", fx("ps_a4.ps"), fx("ps_letter.ps"), "header")
        assert_differs("ps", fx("ps_a4.ps"), fx("ps_letter.ps"), ignore="trailer")

    def test_pjl_key_value_pairs_are_filtered_not_whole_lines(self):
        from DocTest.PrintJobTests import _filter_pjl, _ignored_names
        ignored = _ignored_names(None)
        assert _filter_pjl('@PJL LJOBINFO USERID="a" HOSTID="b" COPIES="2"', ignored) == '@PJL LJOBINFO COPIES="2"'
        assert _filter_pjl('@PJL SET USERNAME="alice"', ignored) is None
        assert _filter_pjl('@PJL SET DUPLEX=ON', ignored) == "@PJL SET DUPLEX=ON"

    def test_page_count_property_check(self):
        job = get_pcl_print_job(fx("pcl5_two_pages.pcl"))
        check_print_job_property(job, "page_count", "2")
        with pytest.raises(AssertionError, match="check failed"):
            check_print_job_property(job, "page_count", "3")
