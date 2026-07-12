"""Shared e2e helpers: suite template and run factory."""

import shutil
import sys
from pathlib import Path

E2E_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(E2E_DIR.parent / "utest" / "dashboard"))

from helpers import CAND_IMAGE, REF_IMAGE, run_robot_suite  # noqa: E402, F401

SUITE_TEMPLATE = """
*** Settings ***
Library    DocTest.VisualTest    result_json=true    take_screenshots=false

*** Test Cases ***
Comparison To Review
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images    {reference}    {candidate}
"""


def make_image_run(base_dir: Path):
    """Reference/candidate copies + a real robot run, all inside base_dir."""
    artifacts = base_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    reference = artifacts / "reference.png"
    candidate = artifacts / "candidate.png"
    shutil.copyfile(REF_IMAGE, reference)
    shutil.copyfile(CAND_IMAGE, candidate)
    suite = SUITE_TEMPLATE.format(reference=reference, candidate=candidate)
    output_xml = run_robot_suite(suite, base_dir / "run")
    return output_xml, reference, candidate
