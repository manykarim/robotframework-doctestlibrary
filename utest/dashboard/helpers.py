"""Shared test helpers: repo paths and real robot-run execution."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UTESTDATA = REPO_ROOT / "utest" / "testdata"

REF_IMAGE = UTESTDATA / "birthday_1080.png"
CAND_IMAGE = UTESTDATA / "birthday_1080_date_id.png"
PDF_REF = UTESTDATA / "sample_1_page.pdf"
PDF_CAND = UTESTDATA / "sample_1_page_moved.pdf"
PDF_MASK = UTESTDATA / "pdf_area_mask.json"


def run_robot_suite(suite_text: str, out_dir: Path) -> Path:
    """Execute a real Robot Framework run; returns the output.xml path."""
    import robot

    out_dir.mkdir(parents=True, exist_ok=True)
    suite = out_dir / "suite.robot"
    suite.write_text(suite_text, encoding="utf-8")
    with open(os.devnull, "w") as devnull:
        rc = robot.run(
            str(suite),
            outputdir=str(out_dir),
            output="output.xml",
            log=None,
            report=None,
            loglevel="INFO",
            stdout=devnull,
        )
    assert rc == 0, f"robot run failed with rc={rc}"
    return out_dir / "output.xml"
