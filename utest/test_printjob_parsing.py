"""Unit tests for the byte-level print-job readers (DocTest.PrintJobParsing)."""

import time
from pathlib import Path
from unittest import mock

import pytest

from DocTest import PrintJobParsing as parsing
from DocTest.PrintJobParsing import (
    PclCommand,
    PclXlError,
    count_postscript_pages,
    iter_pcl5,
    iter_pclxl,
    read_pcl_pages,
    read_postscript,
    split_jobs,
)

FIXTURES = Path(__file__).parent / "testdata" / "printjobs"
SAMPLES = Path(__file__).parents[1] / "testdata"
E = b"\x1b"


def fixture(name):
    return (FIXTURES / name).read_bytes()


def commands(data):
    return [token[1] for token in iter_pcl5(data) if token[0] == "cmd"]


def page_settings(result):
    return [(c["page"], c["property"], c["value"]) for c in result.commands]


# --- PCL 5 tokenizer -------------------------------------------------------

def test_combined_sequence_yields_every_parameter():
    cmds = commands(E + b"&l1o26a1S")
    assert [(c.key, c.value) for c in cmds] == [("&lO", "1"), ("&lA", "26"), ("&lS", "1")]


def test_sequence_without_group_character():
    cmds = commands(E + b"(10U" + E + b"(s1p12v0s3b4148T")
    assert cmds[0] == PclCommand("(", "", "U", "10")
    assert [c.key for c in cmds[1:]] == ["(sP", "(sV", "(sS", "(sB", "(sT"]


@pytest.mark.parametrize("parameter,group,terminator", sorted(parsing.PCL5_PAYLOAD_COMMANDS))
def test_payload_is_skipped_for_every_binary_command(parameter, group, terminator):
    payload = b"\x1b&l1O\x0c\x0c"   # looks like a command and two form feeds
    data = (E + f"{parameter}{group}{len(payload)}{terminator}".encode() + payload
            + E + b"&l2A")
    tokens = list(iter_pcl5(data))
    cmds = [t[1] for t in tokens if t[0] == "cmd"]
    assert [(c.key, c.payload) for c in cmds] == [
        (f"{parameter}{group}{terminator}", len(payload)), ("&lA", 0)]
    assert not any(t[0] == "ff" for t in tokens)


def test_two_character_commands_and_form_feeds():
    tokens = list(iter_pcl5(E + b"Etext\x0c" + E + b"E"))
    assert tokens[0] == ("esc2", "E")
    assert tokens[1][0] == "text" and tokens[2] == ("ff",)
    assert tokens[3] == ("esc2", "E")


def test_hpgl_block_is_skipped_to_pcl_reentry():
    data = E + b"%1BIN;SP1;PD100,100;" + E + b"%0A" + E + b"&l26A"
    tokens = list(iter_pcl5(data))
    assert tokens[0][0] == "hpgl"
    assert [t[1].key for t in tokens if t[0] == "cmd"] == ["%A", "&lA"]


# --- PCL 5 page model --------------------------------------------------------

def test_form_feed_inside_soft_font_data_is_not_a_page():
    result = read_pcl_pages(fixture("pcl5_softfont_ff.pcl"))
    assert result.page_count == 2
    assert ("2", "page_orientation", "1") in page_settings(result)


def test_page_counts_of_fixtures():
    assert read_pcl_pages(fixture("pcl5_one_page.pcl")).page_count == 1
    assert read_pcl_pages(fixture("pcl5_two_pages.pcl")).page_count == 2
    assert read_pcl_pages(fixture("pcl5_no_formfeed.pcl")).page_count == 1
    assert read_pcl_pages(fixture("empty.pcl")).page_count == 0


@pytest.mark.parametrize("data,pages", [
    (E + b"E\x0c" + E + b"E", 1),                      # form feed ejects a blank page
    (E + b"E\x0c\x0c" + E + b"E", 2),
    (E + b"E" + E + b"&l0H" + E + b"E", 0),          # eject without marks: nothing
    (b"A" + E + b"&l0H" + E + b"&l0H", 1),
    (E + b"E" + E + b"&l26A" + E + b"E", 0),         # setup only
    (E + b"E   \r\n" + E + b"E", 0),                  # whitespace is not a mark
    (b"A" + E + b"E" + E + b"E", 1),
    (E + b"E" + E + b"*c100a100b0P" + E + b"E", 1),  # rectangle fill marks
    (E + b"E" + E + b"*r1A" + E + b"*b1W\xff" + E + b"*rB" + E + b"E", 1),
    (E + b"E" + E + b"*p100X" + E + b"*p200Y\x0c" + E + b"E", 1),
])
def test_page_eject_rules_match_ghostpcl(data, pages):
    """Expected counts were measured with GhostPCL 10.08.0."""
    assert read_pcl_pages(data).page_count == pages


def test_command_values_are_not_range_restricted():
    assert ("1", "paper_source", "7") in page_settings(read_pcl_pages(fixture("pcl5_tray7.pcl")))
    assert ("1", "paper_source", "7") in page_settings(read_pcl_pages((SAMPLES / "invoice.pcl").read_bytes()))


def test_overlay_macro_is_flagged():
    data = E + b"&f1y4X" + b"text\x0c"
    assert read_pcl_pages(data).overlay_macro is True


def test_tokenizer_throughput():
    """Large print jobs must stay cheap to read.

    Measured locally on 20 MB jobs: 21 MB/s for soft-font-heavy PCL 5,
    53 MB/s for raster PCL 5, 117 MB/s for PCL XL. The bound is lenient so
    slow shared CI runners do not flake."""
    chunk = b"".join((FIXTURES / name).read_bytes() for name in (
        "pcl5_softfont_ff.pcl", "pcl5_two_pages.pcl")) + (SAMPLES / "invoice_letterformat.pcl").read_bytes()
    data = chunk * max(1, (20 * 1024 * 1024) // len(chunk))
    start = time.perf_counter()
    read_pcl_pages(data)
    rate = len(data) / (time.perf_counter() - start) / 1024 / 1024
    assert rate > 5, f"PCL reader too slow: {rate:.1f} MB/s"


# --- job segmentation ----------------------------------------------------------

def test_split_jobs_reads_pjl_and_language():
    segments = split_jobs((SAMPLES / "invoice.pcl").read_bytes())
    languages = [s.language for s in segments if s.language]
    assert languages == ["PCL5"]
    assert any(line.startswith("@PJL SET RESOLUTION") for s in segments for line in s.pjl)


def test_language_of_unwrapped_data_is_sniffed():
    assert split_jobs(E + b"Etext")[0].language == "PCL5"
    assert split_jobs(b"%!PS\nshowpage\n")[0].language == "POSTSCRIPT"


# --- PCL XL ----------------------------------------------------------------------

def test_pclxl_invoice_page_attributes():
    result = read_pcl_pages((SAMPLES / "invoice_diff_format.pcl").read_bytes())
    assert result.languages == ["PCLXL"]
    assert result.page_count == 1
    assert page_settings(result) == [
        ("1", "paper_format", "2"), ("1", "paper_source", "1"),
        ("1", "page_orientation", "0"), ("1", "duplex", "simplex")]


def test_pclxl_modified_media_is_reported():
    result = read_pcl_pages(fixture("pclxl_letter.pcl"))
    assert ("1", "paper_format", "0") in page_settings(result)


def test_pclxl_scrambled_stream_is_an_error():
    with pytest.raises(PclXlError):
        read_pcl_pages(fixture("pclxl_scrambled.pcl"))


def test_pclxl_ascii_binding_is_rejected():
    with pytest.raises(PclXlError, match="ASCII"):
        list(iter_pclxl(b"` HP-PCL XL;2;0\nBeginSession\n"))


def test_pclxl_big_endian_binding():
    stream = b"( HP-PCL XL;2;0\n" + b"\xc1\x00\x02\xf8\x25" + b"\x43" + b"\xc1\x00\x03\xf8\x31" + b"\x44"
    result = read_pcl_pages(E + b"%-12345X@PJL ENTER LANGUAGE=PCLXL\r\n" + stream)
    assert page_settings(result) == [("1", "paper_format", "2"), ("1", "copies", "3")]


# --- PostScript DSC ------------------------------------------------------------------

@pytest.mark.parametrize("name,pages", [
    ("ps_01_minimal_dsc.ps", 1), ("ps_02_setup_features.ps", 1), ("ps_03_pagesetup_feature.ps", 1),
    ("ps_04_no_prolog.ps", 1), ("ps_05_crlf.ps", 1), ("ps_05b_cr.ps", 1), ("ps_06_pjl_wrapped.ps", 1),
    ("ps_08_pages_atend.ps", 1), ("ps_09_two_pages.ps", 2), ("ps_10_epsf.ps", 1), ("ps_11_ps_adobe_2.ps", 1),
    ("ps_12_no_final_newline.ps", 1), ("ps_13_nested_document.ps", 1), ("ps_14_binary_data.ps", 1),
])
def test_postscript_shapes_are_read(name, pages):
    job = read_postscript(fixture(name))
    assert job.page_count == pages
    assert job.header and job.header[0]["property"] == "Version"


def test_postscript_features_by_scope():
    assert read_postscript(fixture("ps_02_setup_features.ps")).features == [
        {"property": "*PageSize", "value": "A4"}, {"property": "*Duplex", "value": "DuplexNoTumble"}]
    assert read_postscript(fixture("ps_03_pagesetup_feature.ps")).pages == [
        {"page": "1", "property": "*InputSlot", "value": "Tray2"}]


def test_postscript_atend_is_resolved_from_trailer():
    header = read_postscript(fixture("ps_08_pages_atend.ps")).header
    assert {"property": "Pages", "value": "1"} in header


def test_postscript_pjl_wrapper_is_collected():
    job = read_postscript(fixture("ps_06_pjl_wrapped.ps"))
    assert "@PJL SET DUPLEX=ON" in job.pjl_commands
    assert "@PJL EOJ" in job.pjl_commands


def test_postscript_without_dsc_pages_has_no_page_count():
    assert read_postscript(fixture("ps_07_no_dsc_two_pages.ps")).page_count is None


@pytest.mark.skipif(parsing.find_ghostscript() is None, reason="Ghostscript not installed")
def test_ghostscript_counts_pages_of_non_dsc_job():
    assert count_postscript_pages(str(FIXTURES / "ps_07_no_dsc_two_pages.ps")) == 2


def test_missing_ghostscript_is_a_clear_error():
    with mock.patch.object(parsing, "find_ghostscript", return_value=None):
        with pytest.raises(AssertionError, match="no DSC page structure.*Ghostscript is required"):
            count_postscript_pages(str(FIXTURES / "ps_07_no_dsc_two_pages.ps"))


def test_committed_fixtures_match_their_generator(tmp_path):
    """Fixtures are byte-exact inputs (line endings included); a checkout or
    an edit that changes their bytes would silently weaken the tests."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_fixtures", FIXTURES / "make_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(tmp_path)
    generated = sorted(p.name for p in tmp_path.iterdir())
    assert generated
    for name in generated:
        assert (tmp_path / name).read_bytes() == (FIXTURES / name).read_bytes(), name
