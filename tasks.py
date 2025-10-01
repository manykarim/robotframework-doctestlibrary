import importlib
import os
import pathlib
import subprocess
import sys

from invoke import task

import DocTest
from DocTest import __version__ as VERSION
import inspect

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

ROOT = pathlib.Path(__file__).parent.resolve().as_posix()
utests_completed_process = None
atests_completed_process = None
RUST_MANIFEST = pathlib.Path(__file__).parent / "rust" / "ocrs_py" / "Cargo.toml"


def _ensure_ocrs_extension() -> None:
    """Compile the OCRS Rust extension if it is not ready."""

    if os.getenv("DOCTEST_SKIP_OCRS_BUILD") == "1":
        return

    from DocTest import ocrs_adapter

    if ocrs_adapter.ensure_ready():
        return

    cmd = [
        sys.executable,
        "-m",
        "maturin",
        "develop",
        "--release",
        "--manifest-path",
        RUST_MANIFEST.as_posix(),
    ]
    subprocess.run(cmd, check=True)

    sys.modules.pop("DocTest._ocrs", None)
    sys.modules.pop("_ocrs", None)
    importlib.reload(ocrs_adapter)

    if not ocrs_adapter.ensure_ready():
        raise RuntimeError(
            "OCRS extension is unavailable after build. Check the Rust toolchain "
            "and model download connectivity."
        )

@task
def utests(context):
    _ensure_ocrs_extension()
    cmd = [
        "coverage",
        "run",
        "--source=DocTest",
        "-p",
        "-m",
        "pytest",
        "--junitxml=results/pytest.xml",
        f"{ROOT}/utest",
    ]
    global utests_completed_process  
    utests_completed_process = subprocess.run(" ".join(cmd), shell=True, check=False)

@task
def atests(context):
    _ensure_ocrs_extension()
    cmd = [
        "coverage",
        "run",
        "--source=DocTest",
        "-p",
        "-m",
        "robot",
        "--loglevel=TRACE:DEBUG",
        "--listener RobotStackTracer",
        "-d results",
        "--prerebotmodifier utilities.xom.XUnitOut:results/xunit.xml",
        f"{ROOT}/atest/Compare.robot",
        f"{ROOT}/atest/Barcode.robot",
        f"{ROOT}/atest/PdfContent.robot",
        f"{ROOT}/atest/PrintJobs.robot",
        f"{ROOT}/atest/MovementDetection.robot",
    ]
    global atests_completed_process
    atests_completed_process = subprocess.run(" ".join(cmd), shell=True, check=False)

@task(utests, atests)
def tests(context):
    subprocess.run("coverage combine", shell=True, check=False)
    subprocess.run("coverage report", shell=True, check=False)
    subprocess.run("coverage html -d results/htmlcov", shell=True, check=False)
    if utests_completed_process.returncode != 0 or atests_completed_process.returncode != 0:
        raise Exception("Tests failed")

@task
def coverage_report(context):
    subprocess.run("coverage combine", shell=True, check=False)
    subprocess.run("coverage report", shell=True, check=False)
    subprocess.run("coverage html -d results/htmlcov", shell=True, check=False)

@task
def libdoc(context):
    source = f"{ROOT}/DocTest/VisualTest.py"
    target = f"{ROOT}/docs/VisualTest.html"
    cmd = [
        "python",
        "-m",
        "robot.libdoc",
        "-n VisualTest",
        f"-v {VERSION}",
        source,
        target,
    ]
    subprocess.run(" ".join(cmd), shell=True)
    source = f"{ROOT}/DocTest/PdfTest.py"
    target = f"{ROOT}/docs/PdfTest.html"
    cmd = [
        "python",
        "-m",
        "robot.libdoc",
        "-n PdfTest",
        f"-v {VERSION}",
        source,
        target,
    ]
    subprocess.run(" ".join(cmd), shell=True)
    source = f"{ROOT}/DocTest/PrintJobTests.py"
    target = f"{ROOT}/docs/PrintJobTest.html"
    cmd = [
        "python",
        "-m",
        "robot.libdoc",
        "-n PrintJobTest",
        f"-v {VERSION}",
        source,
        target,
    ]
    subprocess.run(" ".join(cmd), shell=True)

@task
def readme(context):
    with open(f"{ROOT}/README.md", "w", encoding="utf-8") as readme:
        doc_string = DocTest.__doc__
        readme.write(str(doc_string))
