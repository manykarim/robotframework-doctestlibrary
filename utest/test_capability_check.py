"""Tests for DocTest.CapabilityCheck module."""

import importlib
import sys

import pytest

from DocTest.CapabilityCheck import (
    CAPABILITY_REGISTRY,
    PYTHON_PACKAGE_REGISTRY,
    CapabilityCheck,
    check_all_capabilities,
    check_binary,
    check_python_package,
    format_capability_report,
)


# ---------------------------------------------------------------------------
# check_binary
# ---------------------------------------------------------------------------


class TestCheckBinary:
    def test_returns_path_when_binary_exists(self, monkeypatch):
        monkeypatch.setattr("DocTest.CapabilityCheck.shutil.which", lambda name: f"/usr/bin/{name}")
        assert check_binary("tesseract") == "/usr/bin/tesseract"

    def test_returns_none_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr("DocTest.CapabilityCheck.shutil.which", lambda name: None)
        assert check_binary("tesseract") is None

    def test_returns_none_on_exception(self, monkeypatch):
        def _raise(name):
            raise OSError("boom")

        monkeypatch.setattr("DocTest.CapabilityCheck.shutil.which", _raise)
        assert check_binary("tesseract") is None


# ---------------------------------------------------------------------------
# check_python_package
# ---------------------------------------------------------------------------


class TestCheckPythonPackage:
    def test_returns_true_for_available_package(self):
        # 'os' is always available
        assert check_python_package("os") is True

    def test_returns_false_for_missing_package(self):
        assert check_python_package("nonexistent_pkg_xyz_999") is False


# ---------------------------------------------------------------------------
# check_all_capabilities – all binaries missing, all packages missing
# ---------------------------------------------------------------------------


class TestCheckAllCapabilitiesAllMissing:
    def test_all_binaries_missing(self, monkeypatch):
        monkeypatch.setattr("DocTest.CapabilityCheck.shutil.which", lambda name: None)
        # Make all Python packages appear missing
        monkeypatch.setattr(
            "DocTest.CapabilityCheck.check_python_package", lambda name: False
        )

        caps = check_all_capabilities()

        for name in CAPABILITY_REGISTRY:
            assert caps[name]["available"] is False
            assert caps[name]["path"] is None
            assert "install_hint" in caps[name]

        for name in PYTHON_PACKAGE_REGISTRY:
            assert caps[name]["available"] is False
            assert "install_hint" in caps[name]

    def test_all_binaries_present(self, monkeypatch):
        monkeypatch.setattr(
            "DocTest.CapabilityCheck.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        monkeypatch.setattr(
            "DocTest.CapabilityCheck._get_version", lambda path: "1.0.0"
        )
        monkeypatch.setattr(
            "DocTest.CapabilityCheck.check_python_package", lambda name: True
        )

        caps = check_all_capabilities()

        for name in CAPABILITY_REGISTRY:
            assert caps[name]["available"] is True
            assert caps[name]["path"] is not None

        for name in PYTHON_PACKAGE_REGISTRY:
            assert caps[name]["available"] is True


# ---------------------------------------------------------------------------
# check_all_capabilities – fallback binary
# ---------------------------------------------------------------------------


class TestCheckAllCapabilitiesFallback:
    def test_imagemagick_uses_fallback(self, monkeypatch):
        """When 'magick' is missing but 'convert' is found, use the fallback."""

        def _which(name):
            if name == "convert":
                return "/usr/bin/convert"
            return None

        monkeypatch.setattr("DocTest.CapabilityCheck.shutil.which", _which)
        monkeypatch.setattr(
            "DocTest.CapabilityCheck._get_version", lambda path: None
        )
        monkeypatch.setattr(
            "DocTest.CapabilityCheck.check_python_package", lambda name: False
        )

        caps = check_all_capabilities()

        assert caps["imagemagick"]["available"] is True
        assert caps["imagemagick"]["path"] == "/usr/bin/convert"
        assert "fallback" in caps["imagemagick"].get("note", "").lower()

    def test_ghostscript_uses_fallback(self, monkeypatch):
        """When 'gs' is missing but 'gswin64c' is found, use the fallback."""

        def _which(name):
            if name == "gswin64c":
                return "C:\\Program Files\\gs\\bin\\gswin64c.exe"
            return None

        monkeypatch.setattr("DocTest.CapabilityCheck.shutil.which", _which)
        monkeypatch.setattr(
            "DocTest.CapabilityCheck._get_version", lambda path: None
        )
        monkeypatch.setattr(
            "DocTest.CapabilityCheck.check_python_package", lambda name: False
        )

        caps = check_all_capabilities()

        assert caps["ghostscript"]["available"] is True
        assert "gswin64c" in caps["ghostscript"]["path"]


# ---------------------------------------------------------------------------
# format_capability_report
# ---------------------------------------------------------------------------


class TestFormatCapabilityReport:
    def test_report_contains_header(self):
        caps = {
            "tesseract": {
                "available": True,
                "feature": "OCR text extraction",
                "path": "/usr/bin/tesseract",
                "version": "5.3.0",
            },
        }
        report = format_capability_report(caps)
        assert "DocTest Environment Capability Report" in report
        assert "tesseract" in report
        assert "OK" in report

    def test_report_shows_missing_with_hint(self):
        caps = {
            "ghostpcl": {
                "available": False,
                "feature": "PCL document rendering",
                "path": None,
                "install_hint": "Install GhostPCL",
            },
        }
        report = format_capability_report(caps)
        assert "MISSING" in report
        assert "Install GhostPCL" in report

    def test_report_summary_line(self):
        caps = {
            "a": {"available": True, "feature": "A"},
            "b": {"available": False, "feature": "B"},
            "c": {"available": True, "feature": "C"},
        }
        report = format_capability_report(caps)
        assert "2/3 capabilities available" in report

    def test_report_shows_version_and_note(self):
        caps = {
            "imagemagick": {
                "available": True,
                "feature": "Image format conversion",
                "path": "/usr/bin/convert",
                "version": "7.1.0",
                "note": "Using fallback 'convert' command",
            },
        }
        report = format_capability_report(caps)
        assert "7.1.0" in report
        assert "fallback" in report


# ---------------------------------------------------------------------------
# CapabilityCheck Robot keyword
# ---------------------------------------------------------------------------


class TestCapabilityCheckKeyword:
    def test_keyword_returns_dict(self, monkeypatch):
        monkeypatch.setattr("DocTest.CapabilityCheck.shutil.which", lambda name: None)
        monkeypatch.setattr(
            "DocTest.CapabilityCheck.check_python_package", lambda name: False
        )

        lib = CapabilityCheck()
        result = lib.check_doctest_environment()

        assert isinstance(result, dict)
        # Must contain all registered capabilities
        for name in CAPABILITY_REGISTRY:
            assert name in result
        for name in PYTHON_PACKAGE_REGISTRY:
            assert name in result

    def test_keyword_logs_report(self, monkeypatch, caplog):
        monkeypatch.setattr("DocTest.CapabilityCheck.shutil.which", lambda name: None)
        monkeypatch.setattr(
            "DocTest.CapabilityCheck.check_python_package", lambda name: False
        )

        lib = CapabilityCheck()
        with caplog.at_level("INFO", logger="DocTest.CapabilityCheck"):
            lib.check_doctest_environment()

        assert "DocTest Environment Capability Report" in caplog.text


# ---------------------------------------------------------------------------
# _get_version
# ---------------------------------------------------------------------------


class TestGetVersion:
    def test_returns_version_string(self, monkeypatch):
        from unittest.mock import MagicMock
        from DocTest.CapabilityCheck import _get_version

        mock_result = MagicMock()
        mock_result.stdout = "tesseract 5.3.0\nlinesep info"
        mock_result.stderr = ""

        monkeypatch.setattr(
            "DocTest.CapabilityCheck.subprocess.run",
            lambda *args, **kwargs: mock_result,
        )
        version = _get_version("/usr/bin/tesseract")
        assert version == "tesseract 5.3.0"

    def test_returns_none_on_failure(self, monkeypatch):
        from DocTest.CapabilityCheck import _get_version

        def _raise(*args, **kwargs):
            raise FileNotFoundError("not found")

        monkeypatch.setattr(
            "DocTest.CapabilityCheck.subprocess.run", _raise
        )
        version = _get_version("/nonexistent/binary")
        assert version is None

    def test_falls_back_to_stderr(self, monkeypatch):
        from unittest.mock import MagicMock
        from DocTest.CapabilityCheck import _get_version

        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "gs 10.0.0"

        monkeypatch.setattr(
            "DocTest.CapabilityCheck.subprocess.run",
            lambda *args, **kwargs: mock_result,
        )
        version = _get_version("/usr/bin/gs")
        assert version == "gs 10.0.0"
