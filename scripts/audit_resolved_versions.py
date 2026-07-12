#!/usr/bin/env python
"""Assert that the marker-controlled dependencies resolved to versions that
match the poetry-era selection intent for the *running* interpreter.

Run inside each supported environment, e.g.::

    uv sync --python 3.11 --all-extras
    uv run python scripts/audit_resolved_versions.py

Exit code 0 = the environment matches the expectation table.
"""

import sys
from importlib.metadata import PackageNotFoundError, version

from packaging.specifiers import SpecifierSet
from packaging.version import Version

PY = sys.version_info[:2]

# package -> expected specifier for this interpreter ("" = any version)
EXPECTATIONS = {
    (3, 10): {"numpy": ">=1.26", "pymupdf": ">=1.26.0", "pydantic-ai-slim": ">=1",
              "scipy": "", "scikit-image": "", "deepdiff": ""},
    (3, 11): {"numpy": ">=1.26", "pymupdf": ">=1.26.0", "pydantic-ai-slim": ">=1",
              "scipy": "", "scikit-image": "", "deepdiff": ""},
    (3, 12): {"numpy": ">=1.26", "pymupdf": ">=1.26.0",
              "scipy": ">=1.11", "scikit-image": ">=0.22.0", "deepdiff": ">=6.0",
              "pydantic-ai-slim": ">=1"},
    (3, 13): {"numpy": ">=2.1.0", "pymupdf": ">=1.26.0",
              "scipy": ">=1.11", "scikit-image": ">=0.25.0", "deepdiff": ">=6.0",
              "pydantic-ai-slim": ">=1"},
}

# packages that must exist on this interpreter (beyond the table)
PRESENCE = ["opencv-python-headless", "robotframework", "pytesseract", "wand"]
PRESENCE_312_PLUS = ["setuptools"]
EXTRA_PRESENCE = ["fastapi", "uvicorn", "python-multipart"]


def main() -> int:
    if PY not in EXPECTATIONS:
        print(f"unsupported interpreter for audit: {sys.version}")
        return 2
    failures = []
    for package, specifier in EXPECTATIONS[PY].items():
        try:
            resolved = version(package)
        except PackageNotFoundError:
            failures.append(f"{package}: not installed")
            continue
        if specifier and Version(resolved) not in SpecifierSet(specifier):
            failures.append(f"{package}: {resolved} violates '{specifier}'")
        else:
            print(f"  ok {package}=={resolved}"
                  + (f"  (expected {specifier})" if specifier else ""))
    presence = PRESENCE + (PRESENCE_312_PLUS if PY >= (3, 12) else [])
    for package in presence + EXTRA_PRESENCE:
        try:
            version(package)
        except PackageNotFoundError:
            failures.append(f"{package}: missing")
    if failures:
        print(f"AUDIT FAILED on {sys.version.split()[0]}:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"AUDIT OK on Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
