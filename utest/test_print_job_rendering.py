"""Rendering PCL/PostScript for visual comparison must never pass silently."""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

from DocTest.DocumentRepresentation import DocumentRepresentation
from DocTest.VisualTest import VisualTest

FIXTURES = Path(__file__).parent / "testdata" / "printjobs"
PCL_COMMANDS = ("pcl6", "gpcl6win64", "gpcl6win32", "gpcl6")
HAS_GHOSTPCL = any(shutil.which(name) for name in PCL_COMMANDS)
real_which = shutil.which


def fake_which(name):
    return r"C:\fake\pcl6.exe" if name == "pcl6" else real_which(name)


def fake_renderer(pages=None, returncode=0, stderr=b"", delay=0.0):
    """A stand-in for subprocess.run that writes ``pages`` blank PNGs.

    ``pages=None`` writes as many pages as the input's form feeds."""
    def run(args, **kwargs):
        output = next(a for a in args if a.startswith("-sOutputFile=")).split("=", 1)[1]
        count = pages if pages is not None else Path(args[-1]).read_bytes().count(b"\x0c")
        for index in range(1, count + 1):
            cv2.imwrite(output.replace("%d", str(index)), np.full((20, 10, 3), 255, np.uint8))
        time.sleep(delay)
        return subprocess.CompletedProcess(args, returncode, b"", stderr)
    return run


def load_with(renderer, path=FIXTURES / "pcl5_one_page.pcl", **kwargs):
    with mock.patch("shutil.which", fake_which), \
         mock.patch("DocTest.DocumentRepresentation.subprocess.run", renderer):
        return DocumentRepresentation(str(path), dpi=72, **kwargs)


# --- 6.1 renderer failures ---------------------------------------------------------

def test_non_zero_exit_is_an_error():
    with pytest.raises(AssertionError, match="failed with exit code 1"):
        load_with(fake_renderer(pages=1, returncode=1, stderr=b"boom"))


def test_renderer_that_cannot_start_is_an_error():
    with pytest.raises(AssertionError, match=r"3221225781 \(0xC0000135\)"):
        load_with(fake_renderer(pages=0, returncode=3221225781))
    with pytest.raises(AssertionError, match="could not be started"):
        load_with(mock.Mock(side_effect=OSError("missing DLL")))


def test_interpreter_error_marker_is_an_error():
    stderr = b"Warning interpreter exited with error code -1\nFlushing to end of job\n"
    with pytest.raises(AssertionError, match="interpreter error"):
        load_with(fake_renderer(pages=1, stderr=stderr))


def test_render_without_pages_is_an_error():
    with pytest.raises(AssertionError, match="produced no pages"):
        load_with(fake_renderer(pages=0))


@pytest.mark.skipif(not HAS_GHOSTPCL, reason="GhostPCL not installed")
def test_compare_images_on_empty_pcl_fails():
    with pytest.raises(AssertionError, match="produced no pages"):
        VisualTest(take_screenshots=False).compare_images(
            str(FIXTURES / "empty.pcl"), str(FIXTURES / "empty.pcl"))


# --- 6.2 page-count cross-check ---------------------------------------------------------

def test_extra_rendered_page_is_an_error():
    with pytest.raises(AssertionError, match=r"produced 2 page\(s\).*contains 1 page\(s\)"):
        load_with(fake_renderer(pages=2))


def test_matching_page_count_loads(tmp_path):
    doc = load_with(fake_renderer(), path=FIXTURES / "pcl5_two_pages.pcl")
    assert len(doc.pages) == 2


def test_envelope_pages_are_accounted_for(tmp_path):
    job = tmp_path / "envelope.pcl"
    job.write_bytes(b"\x1bEenvelope\x0cletter\x0cenvelope\x0c\x1bE")
    doc = load_with(fake_renderer(), path=job, ignore_printfactory_envelope=True)
    assert [p.page_number for p in doc.pages] == [2]


def test_overlay_macro_skips_the_cross_check(tmp_path):
    job = tmp_path / "overlay.pcl"
    job.write_bytes(b"\x1bE\x1b&f1y4Xpage\x0c\x1bE")
    assert len(load_with(fake_renderer(pages=2), path=job).pages) == 2


# --- 6.3 paper default ------------------------------------------------------------------

def test_paper_default_is_passed_as_one_pjl_argument():
    seen = {}

    def run(args, **kwargs):
        seen["args"] = args
        return fake_renderer(pages=1)(args, **kwargs)

    load_with(run, printing_papersize="a4")
    assert "-J@PJL DEFAULT PAPER=A4" in seen["args"]
    assert not any(arg.startswith('-J"') for arg in seen["args"])


@pytest.mark.skipif(not HAS_GHOSTPCL, reason="GhostPCL not installed")
@pytest.mark.parametrize("fixture,papersize,size", [
    ("pcl5_no_pagesize.pcl", "a4", (842, 595)),       # default A4
    ("pcl5_no_pagesize.pcl", "letter", (792, 612)),
    ("pcl5_letter.pcl", "a4", (792, 612)),            # the job's own size wins
])
def test_rendered_paper_size(fixture, papersize, size):
    doc = DocumentRepresentation(str(FIXTURES / fixture), dpi=72, printing_papersize=papersize)
    assert doc.pages[0].image.shape[:2] == size


# --- 6.4 isolated temporary output ---------------------------------------------------------

def test_concurrent_renders_of_same_named_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one = tmp_path / "a" / "job.pcl"
    two = tmp_path / "b" / "job.pcl"
    shutil.copy(FIXTURES / "pcl5_one_page.pcl", one)
    shutil.copy(FIXTURES / "pcl5_two_pages.pcl", two)
    results, errors = {}, []

    def render(path):
        try:
            results[path] = len(DocumentRepresentation(str(path), dpi=72).pages)
        except Exception as exc:  # collected for the assertion below
            errors.append(exc)

    with mock.patch("shutil.which", fake_which), \
         mock.patch("DocTest.DocumentRepresentation.subprocess.run", fake_renderer(delay=0.2)):
        threads = [threading.Thread(target=render, args=(p,)) for p in (one, two)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert not errors
    assert results == {one: 1, two: 2}


def test_no_temporary_directory_remains_after_failure(tmp_path):
    with mock.patch("DocTest.DocumentRepresentation.tempfile.gettempdir", return_value=str(tmp_path)):
        with pytest.raises(AssertionError):
            load_with(fake_renderer(pages=1, returncode=1))
    assert os.listdir(tmp_path) == []
