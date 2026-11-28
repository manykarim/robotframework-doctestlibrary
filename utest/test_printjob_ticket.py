"""Unit tests for normalized print-job tickets (DocTest.PrintJobTicket)."""

import json
from pathlib import Path
from unittest import mock

import pytest

from DocTest import PrintJobParsing
from DocTest.PrintJobParsing import PageDeviceCapture, capture_postscript_page_device, iter_pcl5, iter_pclxl
from DocTest.PrintJobTests import (
    compare_print_job_tickets,
    get_pcl_print_job,
    get_print_job_ticket,
    print_job_ticket_should_contain,
)

FIXTURES = Path(__file__).parent / "testdata" / "printjobs"
SAMPLES = Path(__file__).parents[1] / "testdata"
HAS_GS = PrintJobParsing.find_ghostscript() is not None
needs_gs = pytest.mark.skipif(not HAS_GS, reason="Ghostscript not installed")
E = b"\x1b"


def fx(name):
    return str(FIXTURES / name)


def ticket(path, **kwargs):
    return get_print_job_ticket(str(path), **kwargs)


def page(path, number=1, **kwargs):
    return ticket(path, **kwargs)["pages"][number - 1]


def without_gs():
    return mock.patch.object(PrintJobParsing, "find_ghostscript", return_value=None)


# --- 3.x reader extensions --------------------------------------------------------------

def test_payload_offset_points_at_payload():
    data = E + b"&n11WdLetterhead" + E + b"&l2A"
    command = next(t[1] for t in iter_pcl5(data) if t[0] == "cmd")
    assert data[command.payload_offset:command.payload_offset + command.payload] == b"dLetterhead"


def test_payload_offset_does_not_change_existing_output():
    assert get_pcl_print_job(str(SAMPLES / "invoice.pcl")).properties[0]["value"] == [
        {"page": "1", "property": "page_orientation", "value": "0"},
        {"page": "1", "property": "paper_source", "value": "7"},
        {"page": "1", "property": "paper_format", "value": "26"},
        {"page": "1", "property": "copies", "value": "1"},
    ]


@pytest.mark.parametrize("binding,stream", [
    ("little", b") HP-PCL XL;2;0\n" + b"\xc0\x02\xf8\x24" + b"\x43"),
    ("big", b"( HP-PCL XL;2;0\n" + b"\xc0\x02\xf8\x24" + b"\x43"),
])
def test_pclxl_media_destination_is_decoded(binding, stream):
    operator, attributes = next(iter_pclxl(stream))
    assert operator == 0x43 and attributes[0x24] == 2


@needs_gs
def test_page_device_capture_per_page():
    capture = capture_postscript_page_device(fx("ticket_ps_mid_job_changes.ps"))
    assert capture.error is None and capture.page_count == 3
    assert capture.requests[0] == (1, {"NumCopies": "2", "PageSize": "[595 842]", "MediaPosition": "1",
                                       "Tumble": "false", "Duplex": "true"})
    assert capture.requests[1] == (3, {"PageSize": "[612 792]", "Duplex": "false"})


def test_page_device_capture_without_ghostscript():
    with without_gs():
        capture = capture_postscript_page_device(fx("ticket_ps_mid_job_changes.ps"))
    assert capture.requests == [] and "not found" in capture.error


def test_page_device_capture_when_ghostscript_fails():
    failing = mock.Mock(return_value=mock.Mock(returncode=1, stdout=b"", stderr=b"Unrecoverable error"))
    with mock.patch.object(PrintJobParsing, "find_ghostscript", return_value="gs"), \
         mock.patch.object(PrintJobParsing.subprocess, "run", failing):
        capture = capture_postscript_page_device(fx("ticket_ps_mid_job_changes.ps"))
    assert "exited with 1" in capture.error


# --- 4.x ticket model and effective values ------------------------------------------------------

def test_ticket_dictionary_shape():
    t = ticket(SAMPLES / "invoice.pcl")
    assert set(t) == {"languages", "page_count", "settings_source", "job", "job_sources", "pjl", "pages"}
    entry = t["pages"][0]
    assert set(entry["sources"]) == {"paper_size", "orientation", "duplex", "copies",
                                     "input_tray", "output_bin", "media_type"}
    assert entry["raw"]["paper_size"] == "&lA26"
    assert json.loads(json.dumps(t)) == t


def test_pcl5_invoice_ticket():
    t = ticket(SAMPLES / "invoice.pcl")
    assert t["languages"] == ["PCL5"] and t["page_count"] == 1
    entry = t["pages"][0]
    assert (entry["paper_size"], entry["orientation"], entry["input_tray"], entry["copies"]) == \
        ("A4", "portrait", "auto", "1")
    assert entry["sources"]["paper_size"] == "pcl5"


def test_pjl_variables_and_resolution():
    t = ticket(fx("ticket_run_a.pcl"))
    assert t["job"]["resolution"] == "600" and t["pjl"]["RESOLUTION"] == "600"


def test_pjl_set_lasts_until_reset_default_applies_after_reset():
    """PJL TRM ch. 6: DEFAULT writes the user defaults, which reach the current
    environment only at the next PJL reset (here: a UEL outside JOB..EOJ)."""
    data = (E + b"%-12345X@PJL SET PAPER=LETTER\r\n@PJL DEFAULT DUPLEX=ON\r\n@PJL ENTER LANGUAGE=PCL\r\n"
            + E + b"EPage one\x0c" + E + b"E"
            + E + b"%-12345X@PJL ENTER LANGUAGE=PCL\r\n" + E + b"EPage two\x0c" + E + b"E")
    from DocTest.PrintJobTicket import read_ticket
    pages = read_ticket("job.pcl", data).as_dict()["pages"]
    assert pages[0]["paper_size"] == "Letter" and pages[1]["paper_size"] == "unspecified"
    assert (pages[0]["duplex"], pages[1]["duplex"]) == ("simplex", "long_edge")


def test_pjl_set_survives_a_uel_inside_job():
    data = (E + b"%-12345X@PJL JOB\r\n@PJL SET PAPER=LETTER\r\n@PJL ENTER LANGUAGE=PCL\r\n"
            + E + b"EPage one\x0c" + E + b"E"
            + E + b"%-12345X@PJL ENTER LANGUAGE=PCL\r\n" + E + b"EPage two\x0c" + E + b"E"
            + E + b"%-12345X@PJL EOJ\r\n" + E + b"%-12345X@PJL ENTER LANGUAGE=PCL\r\n" + E + b"EPage three\x0c")
    from DocTest.PrintJobTicket import read_ticket
    pages = read_ticket("job.pcl", data).as_dict()["pages"]
    assert [p["paper_size"] for p in pages] == ["Letter", "Letter", "unspecified"]


def test_pcl_page_size_overrides_pjl_paper():
    entry = page(fx("ticket_pjl_letter_pcl_a4.pcl"))
    assert entry["paper_size"] == "A4" and entry["sources"]["paper_size"] == "pcl5"


def test_settings_carry_forward_across_pages():
    t = ticket(fx("ticket_duplex_once.pcl"))
    assert [p["duplex"] for p in t["pages"]] == ["long_edge", "long_edge"]


def test_reset_restores_pjl_defaults():
    pages = ticket(fx("ticket_pjl_letter_reset.pcl"))["pages"]
    assert pages[0]["paper_size"] == "A4"
    assert pages[1]["paper_size"] == "Letter" and pages[1]["sources"]["paper_size"] == "pjl"


def test_unspecified_paper_size():
    entry = page(fx("pcl5_no_pagesize.pcl"))
    assert entry["paper_size"] == "unspecified"
    assert entry["sources"]["copies"] == "default" and entry["copies"] == "1"


def test_pcl5_media_name_and_bin():
    assert page(fx("ticket_media_name.pcl"))["media_type"] == "letterhead"
    assert page(fx("ticket_bin2.pcl"))["output_bin"] == "face_up"


def test_pjl_short_edge_duplex():
    assert page(fx("ticket_pjl_duplex_short.pcl"))["duplex"] == "short_edge"


def test_pclxl_invoice_ticket():
    entry = page(SAMPLES / "invoice_diff_format.pcl")
    assert (entry["paper_size"], entry["orientation"], entry["input_tray"], entry["duplex"]) == \
        ("A4", "portrait", "auto", "simplex")


def test_pclxl_duplex_and_copies():
    entry = page(fx("ticket_pclxl_duplex_copies.pcl"))
    assert entry["duplex"] == "long_edge" and entry["copies"] == "2"


@needs_gs
def test_postscript_page_device_overrides_dsc():
    t = ticket(fx("ticket_ps_dsc_letter_code_a4.ps"))
    assert t["settings_source"] == "dsc+page-device"
    assert t["pages"][0]["paper_size"] == "A4"
    assert t["pages"][0]["sources"]["paper_size"] == "ps-page-device"


def test_postscript_without_ghostscript_uses_dsc():
    with without_gs():
        t = ticket(fx("ticket_ps_dsc_letter_code_a4.ps"))
    assert t["pages"][0]["paper_size"] == "Letter"
    assert t["pages"][0]["sources"]["paper_size"] == "ps-dsc"
    assert t["settings_source"].startswith("dsc (page-device requests unavailable")


@needs_gs
def test_postscript_requests_per_page():
    pages = ticket(fx("ticket_ps_mid_job_changes.ps"))["pages"]
    assert [(p["paper_size"], p["duplex"], p["copies"]) for p in pages] == [
        ("A4", "long_edge", "2"), ("A4", "long_edge", "2"), ("Letter", "simplex", "2")]


@needs_gs
def test_postscript_custom_size_and_duplex():
    assert page(fx("ticket_ps_custom_size.ps"))["paper_size"] == "212x282mm"
    assert page(fx("ticket_ps_duplex_code.ps"))["duplex"] == "long_edge"


def test_postscript_dsc_features():
    with without_gs():
        assert page(fx("ticket_ps_inputslot_tray2.ps"))["input_tray"] == "ps:Tray2"
        assert page(fx("ps_duplex_notumble.ps"))["duplex"] == "long_edge"
        assert page(fx("ps_duplex_tumble.ps"))["duplex"] == "short_edge"


@needs_gs
def test_ps2write_output_leaves_paper_to_the_printer():
    """ps2write requests no page size by default; the bounding box is no media request."""
    assert page(SAMPLES / "invoice.ps")["paper_size"] == "unspecified"


# --- 5.1 mapping ---------------------------------------------------------------------------------

MAPPING = {"input_tray": {"pcl5:4": "drawer2", "ps:Tray2": "drawer2"}}


def test_mapping_translates_values_and_keeps_raw():
    entry = page(fx("ticket_tray4.pcl"), mapping=MAPPING)
    assert entry["input_tray"] == "drawer2" and entry["raw"]["input_tray"] == "&lH4"
    with without_gs():
        assert page(fx("ticket_ps_inputslot_tray2.ps"), mapping=MAPPING)["input_tray"] == "drawer2"


def test_mapping_from_json_file(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(MAPPING), encoding="utf-8")
    assert page(fx("ticket_tray4.pcl"), mapping=str(path))["input_tray"] == "drawer2"


def test_unknown_mapping_section_is_an_error():
    with pytest.raises(ValueError, match="Unknown mapping section.*input_trays"):
        ticket(fx("ticket_tray4.pcl"), mapping={"input_trays": {"pcl5:4": "x"}})


# --- 6.x comparison and keywords -----------------------------------------------------------------------

def test_tray_difference_message():
    with pytest.raises(AssertionError) as error:
        compare_print_job_tickets(str(SAMPLES / "invoice.pcl"), str(SAMPLES / "invoice_difftray.pcl"))
    message = str(error.value)
    assert message.startswith("The compared print job tickets are different:")
    assert "page 1: input_tray auto → pcl5:4" in message
    assert "duplex" not in message


def test_cross_format_tickets_are_equal_ignoring_job_values():
    compare_print_job_tickets(str(SAMPLES / "invoice.pcl"), str(SAMPLES / "invoice_diff_format.pcl"),
                              ignore_properties="pjl,resolution,color_mode")


def test_unspecified_differs_from_a_concrete_value():
    with pytest.raises(AssertionError, match="page 1: paper_size unspecified → A4"):
        compare_print_job_tickets(fx("pcl5_no_pagesize.pcl"), fx("pcl5_one_page.pcl"))


def test_run_specific_pjl_values_are_ignored():
    compare_print_job_tickets(fx("ticket_run_a.pcl"), fx("ticket_run_b.pcl"))


def test_ignoring_a_setting():
    compare_print_job_tickets(str(SAMPLES / "invoice.pcl"), str(SAMPLES / "invoice_difftray.pcl"),
                              ignore_properties="input_tray")


def test_mapped_trays_compare_across_languages():
    with without_gs():
        compare_print_job_tickets(fx("ticket_tray4.pcl"), fx("ticket_ps_inputslot_tray2.ps"),
                                  ignore_properties="pjl,orientation,duplex,paper_size", mapping=MAPPING)


def test_page_count_and_missing_pages():
    with pytest.raises(AssertionError) as error:
        compare_print_job_tickets(fx("pcl5_one_page.pcl"), fx("pcl5_two_pages.pcl"))
    assert "job: page_count 1 → 2" in str(error.value)
    assert "page 2: missing in reference" in str(error.value)


def test_message_is_capped_at_twenty_differences():
    from DocTest.PrintJobTicket import read_ticket
    many = b"".join(E + b"&l%dHPage\x0c" % (i % 3 + 4) for i in range(30))
    reference = read_ticket("a.pcl", many).as_dict()
    candidate = read_ticket("b.pcl", many.replace(b"&l4H", b"&l7H").replace(b"&l5H", b"&l7H")
                            .replace(b"&l6H", b"&l7H")).as_dict()
    with pytest.raises(AssertionError) as error:
        compare_print_job_tickets(reference, candidate)
    lines = str(error.value).splitlines()
    assert len(lines) == 22 and lines[-1] == "... and 10 more"


def test_should_contain_names_failing_pages():
    with pytest.raises(AssertionError, match="page 2: duplex is simplex, expected long_edge"):
        print_job_ticket_should_contain(fx("ticket_duplex_page1_only.pcl"), duplex="long_edge")


def test_should_contain_single_page_and_job_settings():
    print_job_ticket_should_contain(fx("ticket_duplex_page1_only.pcl"), page=1, paper_size="a4", duplex="long_edge")
    print_job_ticket_should_contain(fx("ticket_run_a.pcl"), resolution="600", page_count="1")


def test_should_contain_rejects_unknown_settings():
    with pytest.raises(ValueError, match="Unknown ticket setting"):
        print_job_ticket_should_contain(fx("ticket_run_a.pcl"), duplx="simplex")
    with pytest.raises(ValueError, match="at least one"):
        print_job_ticket_should_contain(fx("ticket_run_a.pcl"))


@needs_gs
def test_page_device_capture_cost_on_a_large_job(tmp_path):
    """Measured locally (Ghostscript 10.08.0): a 5.1 MB, 500-page job is
    captured in 0.17 s and fully ticketed in 0.20 s, because the nullpage
    device interprets without rasterizing. The bound is lenient for CI."""
    import time
    pages = 500
    hexdata = "0f" * 5000 + "\n"
    body = "".join(f"%%Page: {n} {n}\n/Helvetica findfont 12 scalefont setfont 72 720 moveto (Page {n}) show\n"
                   f"<{hexdata}> pop\nshowpage\n" for n in range(1, pages + 1))
    path = tmp_path / "large.ps"
    path.write_text(f"%!PS-Adobe-3.0\n%%Pages: {pages}\n%%EndComments\n"
                    "<< /PageSize [595 842] /Duplex true >> setpagedevice\n" + body + "%%EOF\n")
    assert path.stat().st_size >= 5_000_000
    start = time.perf_counter()
    result = ticket(path)
    elapsed = time.perf_counter() - start
    assert result["page_count"] == pages and result["pages"][-1]["paper_size"] == "A4"
    assert elapsed < 30, f"ticket of a 5 MB PostScript job took {elapsed:.1f}s"
