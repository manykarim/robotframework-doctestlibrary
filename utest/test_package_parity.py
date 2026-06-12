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
