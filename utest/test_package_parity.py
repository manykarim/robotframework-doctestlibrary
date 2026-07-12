"""Run the wheel-parity gate whenever built artifacts are present."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_wheel_parity_against_poetry_baseline():
    wheels = sorted(ROOT.glob("dist/robotframework_doctestlibrary-*.whl"))
    if not wheels:
        pytest.skip("no built wheel in dist/ — run `uv build` first")
    sdists = sorted(ROOT.glob("dist/robotframework_doctestlibrary-*.tar.gz"))
    command = [sys.executable, str(ROOT / "scripts" / "compare_wheel_contents.py"),
               str(wheels[-1])]
    if sdists:
        command += ["--sdist", str(sdists[-1])]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_extras_present_and_base_dependencies_pure():
    """Dashboard/AI/web stay behind extras; the base install must never pull
    them. WebVisualTest itself ships in the base package dependency-free."""
    from importlib.metadata import metadata

    meta = metadata("robotframework-doctestlibrary")
    extras = set(meta.get_all("Provides-Extra") or [])
    assert {"ai", "dashboard", "all", "browser", "selenium"} <= extras

    base_requirements = [
        requirement for requirement in (meta.get_all("Requires-Dist") or [])
        if "extra ==" not in requirement
    ]
    forbidden_prefixes = (
        "fastapi", "uvicorn", "python-multipart", "pydantic-ai",
        "robotframework-browser", "robotframework-seleniumlibrary",
    )
    offending = [
        requirement for requirement in base_requirements
        if requirement.lower().startswith(forbidden_prefixes)
    ]
    assert not offending, f"extra-only packages leaked into base deps: {offending}"
