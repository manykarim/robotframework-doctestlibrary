import pathlib
import subprocess
from invoke import task
import DocTest
from DocTest import __version__ as VERSION

ROOT = pathlib.Path(__file__).parent.resolve().as_posix()


@task
def utests(context):
    cmd = [
        "coverage",
        "run",
        "--source=DocTest",
        "-p",
        "-m",
        "pytest",
        f"{ROOT}/utest",
    ]
    subprocess.run(" ".join(cmd), shell=True, check=False)

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
        f"{ROOT}/atest/Compare.robot",
        f"{ROOT}/atest/PdfContent.robot",
    ]
    subprocess.run(" ".join(cmd), shell=True, check=False)

@task(utests, atests)
def tests(context):
    subprocess.run("coverage combine", shell=True, check=False)
    subprocess.run("coverage report", shell=True, check=False)
    subprocess.run("coverage html", shell=True, check=False)

@task
def libdoc(context):
    source = f"{ROOT}/DocTest/VisualTest.py"
    target = f"{ROOT}/docs/VisualTest.html"
    cmd = [
        "python",
        "-m",
        "robot.libdoc",
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
        source,
        target,
    ]
    subprocess.run(" ".join(cmd), shell=True)

@task
def readme(context):
    with open(f"{ROOT}/README.md", "w", encoding="utf-8") as readme:
        doc_string = DocTest.__doc__
        readme.write(str(doc_string))