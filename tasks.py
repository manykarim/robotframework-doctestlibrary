import pathlib
import subprocess
from invoke import task
import DocTest
from DocTest import __version__ as VERSION
import inspect

if not hasattr(inspect, 'getargspec'):
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
        f"{ROOT}/atest/Compare.robot",
        f"{ROOT}/atest/PdfContent.robot",
        f"{ROOT}/atest/PrintJobs.robot",
    ]
    global atests_completed_process
    atests_completed_process = subprocess.run(" ".join(cmd), shell=True, check=False)

@task(utests, atests)
def tests(context):
    subprocess.run("coverage combine", shell=True, check=False)
    subprocess.run("coverage report", shell=True, check=False)
    subprocess.run("coverage html", shell=True, check=False)
    if utests_completed_process.returncode != 0 or atests_completed_process.returncode != 0:
        raise Exception("Tests failed")

@task
def coverage_report(context):
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