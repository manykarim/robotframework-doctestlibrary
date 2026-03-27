"""Preflight capability checker for DocTest library.

Detects whether optional external binaries (Tesseract, Ghostscript, GhostPCL,
ImageMagick) and Python packages (PyMuPDF, pyzbar, openai) are available before
tests run.  This avoids cryptic runtime errors when a dependency is missing.

Usage from Robot Framework::

    Library    DocTest.CapabilityCheck

    *** Test Cases ***
    Verify Environment
        ${caps}=    Check DocTest Environment
        Log    ${caps}

Usage from Python::

    from DocTest.CapabilityCheck import check_all_capabilities, format_capability_report
    caps = check_all_capabilities()
    print(format_capability_report(caps))
"""

import importlib
import logging
import shutil
import subprocess
from typing import Dict, List, Optional

from robot.api.deco import keyword, library

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capability registry
# ---------------------------------------------------------------------------

CAPABILITY_REGISTRY: Dict[str, Dict[str, str]] = {
    "tesseract": {
        "binary": "tesseract",
        "feature": "OCR text extraction",
        "install_hint": "apt install tesseract-ocr (Linux) / brew install tesseract (macOS)",
    },
    "ghostscript": {
        "binary": "gs",
        "fallback_binaries": "gswin64c,gswin32c,ghostscript",
        "feature": "PDF/PostScript rendering",
        "install_hint": "apt install ghostscript (Linux) / brew install ghostscript (macOS)",
    },
    "ghostpcl": {
        "binary": "pcl6",
        "fallback_binaries": "gpcl6win64,gpcl6win32,gpcl6",
        "feature": "PCL document rendering",
        "install_hint": (
            "Install GhostPCL from https://ghostscript.com/releases/gpcldnld.html"
        ),
    },
    "imagemagick": {
        "binary": "magick",
        "fallback_binaries": "convert",
        "feature": "Image format conversion",
        "install_hint": (
            "apt install imagemagick (Linux) / brew install imagemagick (macOS)"
        ),
    },
}

PYTHON_PACKAGE_REGISTRY: Dict[str, Dict[str, str]] = {
    "pymupdf": {
        "import_name": "fitz",
        "feature": "PDF rendering and text extraction",
        "install_hint": "pip install PyMuPDF",
    },
    "pyzbar": {
        "import_name": "pyzbar",
        "feature": "Barcode detection and decoding",
        "install_hint": "pip install pyzbar",
    },
    "openai": {
        "import_name": "openai",
        "feature": "LLM-assisted image comparison",
        "install_hint": 'pip install "robotframework-doctestlibrary[ai]"',
    },
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def check_binary(name: str) -> Optional[str]:
    """Return the absolute path of *name* if it is on ``$PATH``, else ``None``.

    Uses :func:`shutil.which` internally.
    """
    try:
        return shutil.which(name)
    except Exception:
        return None


def check_python_package(name: str) -> bool:
    """Return ``True`` if the Python package *name* can be imported."""
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _get_version(binary_path: str) -> Optional[str]:
    """Try to obtain a version string from *binary_path*.

    Attempts ``--version`` first, then ``-version``.  Returns ``None`` on
    any failure so the caller can proceed without version info.
    """
    for flag in ("--version", "-version"):
        try:
            result = subprocess.run(
                [binary_path, flag],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = (result.stdout or result.stderr or "").strip()
            if output:
                # Return the first non-empty line.
                return output.splitlines()[0]
        except Exception:
            continue
    return None


def check_all_capabilities() -> Dict[str, Dict]:
    """Check every registered capability and return a structured report.

    Returns a ``dict`` keyed by capability name.  Each value is a ``dict``
    containing at least ``"available"`` (bool) and ``"feature"`` (str).
    """
    results: Dict[str, Dict] = {}

    # -- Binary capabilities --
    for name, info in CAPABILITY_REGISTRY.items():
        entry: Dict = {
            "available": False,
            "feature": info["feature"],
            "path": None,
        }

        # Try the primary binary first, then any fallbacks.
        binaries_to_try: List[str] = [info["binary"]]
        fallback_str = info.get("fallback_binaries", "")
        if fallback_str:
            binaries_to_try.extend(
                b.strip() for b in fallback_str.split(",") if b.strip()
            )

        used_fallback = False
        for idx, binary_name in enumerate(binaries_to_try):
            path = check_binary(binary_name)
            if path is not None:
                entry["available"] = True
                entry["path"] = path
                version = _get_version(path)
                if version:
                    entry["version"] = version
                if idx > 0:
                    used_fallback = True
                    entry["note"] = f"Using fallback '{binary_name}' command"
                break

        if not entry["available"]:
            entry["install_hint"] = info["install_hint"]

        results[name] = entry

    # -- Python package capabilities --
    for name, info in PYTHON_PACKAGE_REGISTRY.items():
        import_name = info["import_name"]
        available = check_python_package(import_name)
        entry = {
            "available": available,
            "feature": info["feature"],
            "package": import_name,
        }
        if not available:
            entry["install_hint"] = info["install_hint"]
        results[name] = entry

    return results


def format_capability_report(capabilities: Dict[str, Dict]) -> str:
    """Format *capabilities* into a human-readable report string."""
    lines: List[str] = []
    lines.append("DocTest Environment Capability Report")
    lines.append("=" * 42)

    available_items: List[str] = []
    missing_items: List[str] = []

    for name, info in capabilities.items():
        status = "OK" if info["available"] else "MISSING"
        feature = info.get("feature", "")
        line = f"  [{status:>7}] {name:<15} - {feature}"
        if info["available"]:
            path = info.get("path")
            version = info.get("version")
            note = info.get("note")
            details: List[str] = []
            if path:
                details.append(f"path: {path}")
            if version:
                details.append(f"version: {version}")
            if note:
                details.append(note)
            if details:
                line += f" ({', '.join(details)})"
            available_items.append(line)
        else:
            hint = info.get("install_hint", "")
            if hint:
                line += f"\n{'':>28}Install: {hint}"
            missing_items.append(line)

    if available_items:
        lines.append("")
        lines.append("Available:")
        lines.extend(available_items)

    if missing_items:
        lines.append("")
        lines.append("Missing:")
        lines.extend(missing_items)

    lines.append("")
    total = len(capabilities)
    ok_count = sum(1 for v in capabilities.values() if v["available"])
    lines.append(f"Summary: {ok_count}/{total} capabilities available")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Robot Framework library
# ---------------------------------------------------------------------------


@library(scope="GLOBAL")
class CapabilityCheck:
    """Preflight capability checker for the DocTest library.

    Provides a keyword to detect whether optional external binaries and
    Python packages required by various DocTest features are installed.
    """

    ROBOT_LIBRARY_VERSION = "1.0"

    @keyword("Check DocTest Environment")
    def check_doctest_environment(self) -> Dict[str, Dict]:
        """Check and report available capabilities for DocTest library.

        Returns a structured ``dict`` keyed by capability name and logs a
        human-readable report.

        Example:
        | ${caps}= | Check DocTest Environment |
        | Log       | ${caps}                   |
        """
        capabilities = check_all_capabilities()
        report = format_capability_report(capabilities)
        logger.info(report)
        return capabilities
