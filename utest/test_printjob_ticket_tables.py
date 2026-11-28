"""One test per normalization-table entry of DocTest.PrintJobTicket.

Expected values come from the primary documents listed in the module
docstring (PCL 5 TRM / Comparison Guide / Implementor's Guide, PCL XL
Feature Reference 2.0/2.1/3.0, PJL Technical Reference, PLRM 3, PPD 4.3).
"""

import pytest

from DocTest import PrintJobParsing
from DocTest import PrintJobTicket as T
from DocTest.PrintJobTicket import paper_from_points, read_ticket

E = b"\x1b"
UEL = E + b"%-12345X"
HAS_GS = PrintJobParsing.find_ghostscript() is not None


def pcl5_page(*commands):
    return read_ticket("t.pcl", E + b"E" + b"".join(E + c for c in commands) + b"Page\x0c" + E + b"E").as_dict()["pages"][0]


def pjl_page(*lines):
    data = UEL + b"".join(b"@PJL " + line + b"\r\n" for line in lines) + b"@PJL ENTER LANGUAGE=PCL\r\n" + b"Page\x0c"
    return read_ticket("t.pcl", data).as_dict()["pages"][0]


def xl_page(**attributes):
    """A one-page little-endian PCL XL stream with ubyte BeginPage attributes."""
    ids = {"size": 0x25, "source": 0x26, "orientation": 0x28, "destination": 0x24, "duplex": 0x35, "simplex": 0x34}
    begin = b"".join(b"\xc0" + bytes([value]) + b"\xf8" + bytes([ids[name]]) for name, value in attributes.items())
    stream = b") HP-PCL XL;2;0\n" + begin + b"\x43" + b"\x44"
    return read_ticket("t.pcl", UEL + b"@PJL ENTER LANGUAGE=PCLXL\r\n" + stream).as_dict()["pages"][0]


def xl_media_type(name):
    stream = (b") HP-PCL XL;2;0\n" + b"\xc8\xc0" + bytes([len(name)]) + name + b"\xf8\x27" + b"\x43\x44")
    return read_ticket("t.pcl", UEL + b"@PJL ENTER LANGUAGE=PCLXL\r\n" + stream).as_dict()["pages"][0]


# --- paper sizes ----------------------------------------------------------------------------

@pytest.mark.parametrize("code,name", sorted(T.PCL5_PAPER.items(), key=lambda kv: int(kv[0])))
def test_pcl5_paper_codes(code, name):
    assert pcl5_page(b"&l" + code.encode() + b"A")["paper_size"] == name


@pytest.mark.parametrize("value,name", sorted(T.PCLXL_PAPER.items()))
def test_pclxl_media_sizes(value, name):
    assert xl_page(size=value)["paper_size"] == name


@pytest.mark.parametrize("pjl,name", [
    ("LETTER", "Letter"), ("LEGAL", "Legal"), ("A5", "A5"), ("A4", "A4"), ("EXECUTIVE", "Executive"),
    ("LEDGER", "Ledger"), ("A3", "A3"), ("COM10", "Com10"), ("C5", "C5"), ("DL", "DL"),
    ("MONARCH", "Monarch"), ("B5", "B5"), ("JISB4", "JIS B4"), ("JISB5", "JIS B5"),
    ("JPOST", "Hagaki"), ("JPOSTD", "Oufuku-Hagaki"), ("CUSTOM", "pjl:CUSTOM"),
])
def test_pjl_paper_values(pjl, name):
    assert pjl_page(b"SET PAPER=" + pjl.encode())["paper_size"] == name


@pytest.mark.parametrize("name,size", sorted(T.PAPER_DIMENSIONS.items()))
def test_postscript_dimensions_in_both_orientations(name, size):
    width, height = size
    assert paper_from_points(width, height) == name
    assert paper_from_points(height, width) == name
    assert paper_from_points(width + 1.5, height - 1.5) == name


def test_unknown_postscript_size_in_millimetres():
    assert paper_from_points(600, 800) == "212x282mm"


def test_named_sizes_are_unambiguous():
    names = list(T.PAPER_DIMENSIONS)
    for a in names:
        for b in names:
            if a < b:
                wa, ha = T.PAPER_DIMENSIONS[a]
                assert paper_from_points(wa, ha) == a, f"{a} collides with {b}"


# --- orientation / duplex ------------------------------------------------------------------

@pytest.mark.parametrize("code,name", [("0", "portrait"), ("1", "landscape"),
                                       ("2", "reverse_portrait"), ("3", "reverse_landscape")])
def test_pcl5_orientation(code, name):
    assert pcl5_page(b"&l" + code.encode() + b"O")["orientation"] == name


@pytest.mark.parametrize("value,name", sorted(T.PCLXL_ORIENTATIONS.items()))
def test_pclxl_orientation(value, name):
    assert xl_page(orientation=value)["orientation"] == name


@pytest.mark.parametrize("code,name", sorted(T.PCL5_DUPLEX.items()))
def test_pcl5_duplex_is_physical(code, name):
    """TRM 4-5: binding is defined on the physical page, independent of orientation."""
    assert pcl5_page(b"&l" + code.encode() + b"S")["duplex"] == name
    assert pcl5_page(b"&l1O", b"&l" + code.encode() + b"S")["duplex"] == name


@pytest.mark.parametrize("orientation,mode,name", [
    (0, 1, "long_edge"), (0, 0, "short_edge"),       # portrait: vertical = long edge
    (1, 1, "short_edge"), (1, 0, "long_edge"),       # landscape: vertical = short edge
    (3, 1, "short_edge"), (2, 1, "long_edge"),
])
def test_pclxl_duplex_is_relative_to_orientation(orientation, mode, name):
    assert xl_page(orientation=orientation, duplex=mode)["duplex"] == name


def test_pclxl_simplex():
    assert xl_page(simplex=0)["duplex"] == "simplex"


@pytest.mark.parametrize("lines,name", [
    ((b"SET DUPLEX=OFF",), "simplex"), ((b"SET DUPLEX=ON",), "long_edge"),
    ((b"SET DUPLEX=ON", b"SET BINDING=LONGEDGE"), "long_edge"),
    ((b"SET DUPLEX=ON", b"SET BINDING=SHORTEDGE"), "short_edge"),
])
def test_pjl_duplex(lines, name):
    assert pjl_page(*lines)["duplex"] == name


@pytest.mark.parametrize("value,name", [(b"PORTRAIT", "portrait"), (b"LANDSCAPE", "landscape")])
def test_pjl_orientation(value, name):
    assert pjl_page(b"SET ORIENTATION=" + value)["orientation"] == name


# --- trays, bins, media ------------------------------------------------------------------------

@pytest.mark.parametrize("code,name", [("2", "manual"), ("3", "manual_envelope"), ("6", "envelope"),
                                       ("7", "auto"), ("1", "pcl5:1"), ("4", "pcl5:4"), ("5", "pcl5:5"),
                                       ("8", "pcl5:8"), ("20", "pcl5:20")])
def test_pcl5_trays(code, name):
    assert pcl5_page(b"&l" + code.encode() + b"H")["input_tray"] == name


def test_pcl5_eject_is_not_a_tray():
    assert pcl5_page(b"&l0H")["input_tray"] == "unspecified"


@pytest.mark.parametrize("value,name", sorted(T.PCLXL_TRAYS.items()) + [(8, "pclxl:8")])
def test_pclxl_trays(value, name):
    assert xl_page(source=value)["input_tray"] == name


@pytest.mark.parametrize("code,name", sorted(T.PCL5_BINS.items()) + [("3", "pcl5:3")])
def test_pcl5_bins(code, name):
    assert pcl5_page(b"&l" + code.encode() + b"G")["output_bin"] == name


@pytest.mark.parametrize("value,name", sorted(T.PCLXL_BINS.items()) + [(5, "pclxl:5")])
def test_pclxl_bins(value, name):
    assert xl_page(destination=value)["output_bin"] == name


@pytest.mark.parametrize("lines,setting,name", [
    ((b"SET MANUALFEED=ON",), "input_tray", "manual"),
    ((b"SET MEDIASOURCE=TRAY2",), "input_tray", "pjl:TRAY2"),
    ((b"SET OUTBIN=UPPER",), "output_bin", "face_down"),
    ((b"SET OUTBIN=LOWER",), "output_bin", "face_up"),
    ((b"SET OUTBIN=OPTIONALOUTPUTBIN1",), "output_bin", "pjl:OPTIONALOUTPUTBIN1"),
    ((b"SET MEDIATYPE=PAPER",), "media_type", "plain"),
    ((b"SET MEDIATYPE=TRANSPARENCY",), "media_type", "transparency"),
])
def test_pjl_tray_bin_media(lines, setting, name):
    assert pjl_page(*lines)[setting] == name


@pytest.mark.parametrize("code,name", sorted(T.PCL5_MEDIA.items()))
def test_pcl5_media_numbers(code, name):
    assert pcl5_page(b"&l" + code.encode() + b"M")["media_type"] == name


@pytest.mark.parametrize("name,expected", [(b"Plain", "plain"), (b"Bond", "bond"),
                                           (b"Transparency", "transparency"), (b"Letterhead", "letterhead")])
def test_pcl5_media_by_name(name, expected):
    payload = b"d" + name                     # operation 100 ('d') + name; # = len(name) + 1
    assert pcl5_page(b"&n" + str(len(payload)).encode() + b"W" + payload)["media_type"] == expected


@pytest.mark.parametrize("name,expected", [(b"", "plain"), (b"Glossy  Paper", "glossy paper")])
def test_pclxl_media_type_strings(name, expected):
    assert xl_media_type(name)["media_type"] == expected


# --- PostScript -------------------------------------------------------------------------------------

def ps_job(setup=b"", header=b""):
    return (b"%!PS-Adobe-3.0\n" + header + b"%%Pages: 1\n%%EndComments\n" + setup
            + b"%%Page: 1 1\nshowpage\n%%EOF\n")


def ps_page(tmp_path, setup=b"", header=b""):
    path = tmp_path / "job.ps"
    data = ps_job(setup, header)
    path.write_bytes(data)
    return read_ticket(str(path), data).as_dict()["pages"][0]


def feature(key, option):
    return b"%%BeginFeature: *" + key + b" " + option + b"\n%%EndFeature\n"


@pytest.mark.parametrize("option,name", [(b"None", "simplex"), (b"DuplexNoTumble", "long_edge"),
                                         (b"DuplexTumble", "short_edge")])
def test_dsc_duplex_options(tmp_path, option, name):
    assert ps_page(tmp_path, b"%%BeginSetup\n" + feature(b"Duplex", option) + b"%%EndSetup\n")["duplex"] == name


def test_dsc_manual_feed(tmp_path):
    assert ps_page(tmp_path, b"%%BeginSetup\n" + feature(b"ManualFeed", b"True") + b"%%EndSetup\n")["input_tray"] == "manual"


@pytest.mark.skipif(not HAS_GS, reason="Ghostscript not installed")
@pytest.mark.parametrize("size,tumble,duplex,orientation", [
    (b"[595 842]", b"false", "long_edge", "portrait"),
    (b"[595 842]", b"true", "short_edge", "portrait"),
    (b"[842 595]", b"false", "short_edge", "landscape"),    # Tumble is relative to the page shape
    (b"[842 595]", b"true", "long_edge", "landscape"),
])
def test_postscript_tumble_is_relative_to_page_size(tmp_path, size, tumble, duplex, orientation):
    setup = b"<< /PageSize " + size + b" /Duplex true /Tumble " + tumble + b" >> setpagedevice\n"
    entry = ps_page(tmp_path, setup)
    assert (entry["paper_size"], entry["duplex"], entry["orientation"]) == ("A4", duplex, orientation)


@pytest.mark.skipif(not HAS_GS, reason="Ghostscript not installed")
def test_dsc_orientation_wins_over_page_size(tmp_path):
    setup = b"<< /PageSize [595 842] >> setpagedevice\n"
    assert ps_page(tmp_path, setup, header=b"%%Orientation: Landscape\n")["orientation"] == "landscape"


@pytest.mark.skipif(not HAS_GS, reason="Ghostscript not installed")
def test_postscript_orientation_key_is_not_used(tmp_path):
    """PLRM 3 Table 6.3: Orientation rotates roll-fed media; it is not portrait/landscape."""
    entry = ps_page(tmp_path, b"<< /Orientation 1 >> setpagedevice\n")
    assert entry["orientation"] == "unspecified"

