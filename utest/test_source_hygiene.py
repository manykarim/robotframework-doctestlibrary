"""Shipped code must not trigger interpreter-level warnings on new Pythons
(invalid escape sequences become errors in a future release)."""

import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted(
    path
    for package in ("DocTest", "doctest_dashboard")
    for path in (REPO_ROOT / package).rglob("*.py")
)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_module_compiles_without_syntax_warnings(path):
    source = path.read_text(encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        compile(source, str(path), "exec")


def test_print_job_modules_import_without_syntax_warnings():
    """The print-job readers compile their regular expressions at import
    time; a non-raw literal surfaces there, not in a plain compile."""
    result = subprocess.run(
        [sys.executable, "-W", "error::SyntaxWarning", "-c",
         "import DocTest.PrintJobTests, DocTest.PrintJobParsing"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
    )
    assert result.returncode == 0, result.stderr


def test_no_tracked_paths_differ_only_by_case():
    """Windows and macOS file systems are case-insensitive: two tracked paths
    such as docs/AI.html and docs/Ai.html share one file on disk there, so a
    checkout is permanently dirty and regenerated docs land in the wrong entry."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, text=True, errors="replace"
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout")
    by_folded_name = {}
    for path in filter(None, result.stdout.split("\0")):
        by_folded_name.setdefault(path.casefold(), []).append(path)
    collisions = [paths for paths in by_folded_name.values() if len(paths) > 1]
    assert collisions == []


def test_pdf_backend_is_imported_by_its_current_name():
    result = subprocess.run(
        [sys.executable, "-W", "always", "-c", "import DocTest.VisualTest, DocTest.PdfTest"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
    )
    assert result.returncode == 0, result.stderr
    assert "fitz" not in result.stderr
