"""Regenerate the print-job fixtures in this directory.

The fixtures are small, synthetic and free of personal data. They cover the
false-pass cases of the print-job keywords (see the OpenSpec change
``fix-print-job-false-passes``). Run from the repository root:

    python utest/testdata/printjobs/make_fixtures.py
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
E = b"\x1b"
UEL = E + b"%-12345X"


def pcl5(*pages, setup=b"&l26a0O"):
    """A PCL 5 job: reset, page setup, one form-feed terminated text per page."""
    body = E + b"E" + E + setup
    for page in pages:
        body += page + b"\x0c"
    return body + E + b"E"


def ps(pages=1, header=b"", setup=b"", page_setup=b"", trailer=b"", first=b"%!PS-Adobe-3.0"):
    lines = [first, b"%%Title: invoice", b"%%Creator: fixture generator",
             b"%%Pages: " + str(pages).encode(), b"%%BoundingBox: 0 0 595 842"]
    out = b"\n".join(lines) + b"\n" + header + b"%%EndComments\n%%BeginProlog\n/x {} def\n%%EndProlog\n"
    out += setup
    for n in range(1, pages + 1):
        out += b"%%Page: " + b"%d %d" % (n, n) + b"\n" + page_setup
        out += b"/Helvetica findfont 12 scalefont setfont 72 720 moveto (Page %d) show showpage\n" % n
    return out + b"%%Trailer\n" + trailer + b"%%EOF\n"


def feature(key, option, code=b"% feature code"):
    return b"%%BeginFeature: *" + key + b" " + option + b"\n" + code + b"\n%%EndFeature\n"


OUT = HERE


def write(name, data):
    (OUT / name).write_bytes(data)


def main(out_dir=HERE):
    global OUT
    OUT = Path(out_dir)
    # --- PCL 5 -------------------------------------------------------------
    write("pcl5_one_page.pcl", pcl5(b"Page one"))
    write("pcl5_two_pages.pcl", pcl5(b"Page one", b"Page two"))
    # bitmap soft-font character whose data contains form-feed bytes,
    # then landscape on page 2
    glyph = bytes([4, 0, 14, 1, 0, 0, 0, 0, 0, 8, 0, 8, 0, 8]) + bytes(
        [0x0C, 0x3C, 0x0C, 0x0C, 0x7E, 0x0C, 0x0C, 0x00])
    softfont = (E + b"E" + E + b"&l26a0O" + E + b"*c1D" + E + b"*c65E"
                + E + b"(s%dW" % len(glyph) + glyph
                + b"Page one\x0c" + E + b"&l1O" + b"Page two\x0c" + E + b"E")
    write("pcl5_softfont_ff.pcl", softfont)
    write("pcl5_tray7.pcl", pcl5(b"Page one", setup=b"&l26a7h0O"))
    write("pcl5_no_pagesize.pcl", E + b"E" + b"Hello, no page size set\x0c" + E + b"E")
    write("pcl5_letter.pcl", pcl5(b"Letter page", setup=b"&l2a0O"))
    write("pcl5_no_formfeed.pcl", E + b"E" + E + b"&l26A" + b"Text without trailing form feed" + E + b"E")
    write("empty.pcl", b"")

    # --- PCL XL -------------------------------------------------------------
    xl = (REPO_ROOT / "testdata" / "invoice_diff_format.pcl").read_bytes()
    header_end = xl.index(b"\n", xl.index(b") HP-PCL XL")) + 1
    write("pclxl_scrambled.pcl", xl[:header_end] + bytes(b ^ 0x5A for b in xl[header_end:]))
    # BeginPage MediaSize A4 (enum 2) -> Letter (enum 0)
    media_a4 = b"\xc0\x02\xf8\x25"
    assert xl.count(media_a4) == 1, "fixture source changed"
    write("pclxl_letter.pcl", xl.replace(media_a4, b"\xc0\x00\xf8\x25"))

    # --- PostScript shape variants -------------------------------------------
    write("ps_01_minimal_dsc.ps", ps())
    write("ps_02_setup_features.ps", ps(setup=b"%%BeginSetup\n" + feature(b"PageSize", b"A4")
                                         + feature(b"Duplex", b"DuplexNoTumble") + b"%%EndSetup\n"))
    write("ps_03_pagesetup_feature.ps", ps(page_setup=b"%%BeginPageSetup\n"
                                            + feature(b"InputSlot", b"Tray2") + b"%%EndPageSetup\n"))
    write("ps_04_no_prolog.ps", b"%!PS-Adobe-3.0\n%%Pages: 1\n%%EndComments\n%%Page: 1 1\n"
          b"/Helvetica findfont 12 scalefont setfont 72 720 moveto (Page 1) show showpage\n%%Trailer\n%%EOF\n")
    write("ps_05_crlf.ps", ps().replace(b"\n", b"\r\n"))
    write("ps_05b_cr.ps", ps().replace(b"\n", b"\r"))
    write("ps_06_pjl_wrapped.ps", UEL + b"@PJL JOB NAME=\"invoice\"\r\n@PJL SET DUPLEX=ON\r\n"
          b"@PJL SET COPIES=2\r\n@PJL ENTER LANGUAGE=POSTSCRIPT\r\n" + ps()
          + UEL + b"@PJL EOJ\r\n" + UEL)
    write("ps_07_no_dsc_two_pages.ps", b"%!PS\n" + b"".join(
        b"/Helvetica findfont 12 scalefont setfont 72 720 moveto (Page %d) show showpage\n" % n
        for n in (1, 2)))
    write("ps_08_pages_atend.ps", ps(header=b"").replace(b"%%Pages: 1", b"%%Pages: (atend)")
          .replace(b"%%Trailer\n", b"%%Trailer\n%%Pages: 1\n"))
    write("ps_09_two_pages.ps", ps(pages=2))
    write("ps_10_epsf.ps", ps(first=b"%!PS-Adobe-3.0 EPSF-3.0"))
    write("ps_11_ps_adobe_2.ps", ps(first=b"%!PS-Adobe-2.0"))
    write("ps_12_no_final_newline.ps", ps().rstrip(b"\n"))
    nested = (b"%%BeginDocument: logo.eps\n%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\n"
              b"%%Pages: 1\n%%EndComments\n%%Page: 1 1\n0 0 moveto 10 10 lineto stroke\n%%EOF\n"
              b"%%EndDocument\n")
    write("ps_13_nested_document.ps", ps(page_setup=nested))
    binary = b"%%BeginBinary: 12\n%%Page: 9 9\x0c\x00\n%%EndBinary\n"
    write("ps_14_binary_data.ps", ps(page_setup=binary))

    # --- PostScript comparison pairs ------------------------------------------
    write("ps_a4.ps", ps())
    write("ps_letter.ps", ps().replace(b"%%BoundingBox: 0 0 595 842", b"%%BoundingBox: 0 0 612 792"))
    write("ps_duplex_notumble.ps", ps(setup=b"%%BeginSetup\n" + feature(b"Duplex", b"DuplexNoTumble") + b"%%EndSetup\n"))
    write("ps_duplex_tumble.ps", ps(setup=b"%%BeginSetup\n" + feature(b"Duplex", b"DuplexTumble") + b"%%EndSetup\n"))
    def run(job, user):
        return (UEL + b'@PJL JOB NAME="' + job + b'"\r\n@PJL SET USERNAME="' + user + b'"\r\n'
                b'@PJL SET JOBNAME="' + job + b'"\r\n@PJL SET DUPLEX=ON\r\n@PJL ENTER LANGUAGE=POSTSCRIPT\r\n')
    write("ps_run_a.ps", run(b"job-1", b"alice") + ps(header=b"%%CreationDate: D:20260101120000\n"))
    write("ps_run_b.ps", run(b"job-2", b"bob")
          + ps(header=b"%%CreationDate: D:20261127093000\n").replace(b"%%Title: invoice", b"%%Title: invoice (2)"))
    write("ps_creator_a.ps", ps())
    write("ps_creator_b.ps", ps().replace(b"%%Creator: fixture generator", b"%%Creator: fixture generator 2.0"))


if __name__ == "__main__":
    main()
