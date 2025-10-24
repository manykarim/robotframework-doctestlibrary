import inspect
import pathlib
import subprocess

from invoke import task

import DocTest
from DocTest import __version__ as VERSION

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

ROOT = pathlib.Path(__file__).parent.resolve().as_posix()
utests_completed_process = None
atests_completed_process = None


@task
def utests(context):
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
        f"{ROOT}/atest/LLM.robot",
    ]
    global atests_completed_process
    atests_completed_process = subprocess.run(" ".join(cmd), shell=True, check=False)


@task(utests, atests)
def tests(context):
    subprocess.run("coverage combine", shell=True, check=False)
    subprocess.run("coverage report", shell=True, check=False)
    subprocess.run("coverage html -d results/htmlcov", shell=True, check=False)
    if (
        utests_completed_process.returncode != 0
        or atests_completed_process.returncode != 0
    ):
        raise Exception("Tests failed")


@task
def coverage_report(context):
    subprocess.run("coverage combine", shell=True, check=False)
    subprocess.run("coverage report", shell=True, check=False)
    subprocess.run("coverage html -d results/htmlcov", shell=True, check=False)


@task
def libdoc(context):
    libraries = [
        ("VisualTest", "DocTest/VisualTest.py"),
        ("PdfTest", "DocTest/PdfTest.py"),
        ("PrintJobTest", "DocTest/PrintJobTests.py"),
        ("Ai", "DocTest/Ai/__init__.py"),
    ]

    for name, source_path in libraries:
        source = f"{ROOT}/{source_path}"

        # Document without version in filename
        target = f"{ROOT}/docs/{name}.html"
        cmd = [
            "python",
            "-m",
            "robot.libdoc",
            "-n",
            name,
            "-v",
            VERSION,
            source,
            target,
        ]
        subprocess.run(" ".join(cmd), shell=True)

        # Document with version in filename
        target_versioned = f"{ROOT}/docs/{name}-{VERSION}.html"
        cmd = [
            "python",
            "-m",
            "robot.libdoc",
            "-n",
            name,
            "-v",
            VERSION,
            source,
            target_versioned,
        ]
        subprocess.run(" ".join(cmd), shell=True)


@task
def readme(context):
    doc_string = DocTest.__doc__ or ""
    with open(f"{ROOT}/README.md", "w", encoding="utf-8") as readme:
        readme.write(str(doc_string).strip() + "\n")
