import inspect
import pathlib
import subprocess

from invoke import task

import DocTest
from DocTest import __version__ as VERSION

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

ROOT = pathlib.Path(__file__).parent.resolve().as_posix()
SUPPORTED_PYTHONS = ["3.10", "3.11", "3.12", "3.13"]
utests_completed_process = None
atests_completed_process = None


@task
def utests(context):
    # pytest-cov instead of a coverage-run wrapper: xdist workers are traced
    # correctly, and [tool.coverage.run] parallel=true keeps the data files
    # mergeable with the coverage-wrapped Robot runs via `coverage combine`.
    cmd = [
        "uv",
        "run",
        "--",
        "pytest",
        "--junitxml=results/pytest.xml",
        "--cov=DocTest",
        "--cov=doctest_dashboard",
        "--cov-report=",
        "-n auto",
        "--timeout=300",
        f"{ROOT}/utest",
    ]
    global utests_completed_process
    utests_completed_process = subprocess.run(" ".join(cmd), shell=True, check=False)


@task
def atests(context):
    cmd = [
        "uv",
        "run",
        "--",
        "coverage",
        "run",
        "--source=DocTest,doctest_dashboard",
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
        f"{ROOT}/atest/ReferenceRun.robot",
        f"{ROOT}/atest/ResultJson.robot",
        f"{ROOT}/atest/LLM.robot",
    ]
    global atests_completed_process
    atests_completed_process = subprocess.run(" ".join(cmd), shell=True, check=False)


@task(utests, atests)
def tests(context):
    subprocess.run("uv run -- coverage combine", shell=True, check=False)
    # coverage report enforces [tool.coverage.report] fail_under
    report_process = subprocess.run("uv run -- coverage report", shell=True, check=False)
    subprocess.run("uv run -- coverage html -d results/htmlcov", shell=True, check=False)
    if (
        utests_completed_process.returncode != 0
        or atests_completed_process.returncode != 0
    ):
        raise Exception("Tests failed")
    if report_process.returncode != 0:
        raise Exception("Coverage below the configured fail_under floor")


@task
def coverage_report(context):
    subprocess.run("uv run -- coverage combine", shell=True, check=False)
    subprocess.run("uv run -- coverage report", shell=True, check=False)
    subprocess.run("uv run -- coverage html -d results/htmlcov", shell=True, check=False)


@task
def e2e(context):
    """Run the dashboard end-to-end journeys (requires a built frontend)."""
    subprocess.run(
        f"uv run -- pytest {ROOT}/e2e --browser chromium", shell=True, check=True
    )


@task
def multipython(context):
    """Validate every supported interpreter: sync with all extras, audit the
    resolved marker-controlled dependency versions, and import-smoke both
    packages. Restores the default environment afterwards."""
    results = {}
    for version in SUPPORTED_PYTHONS:
        print(f"\n=== Python {version} ===")
        steps = [
            f"uv sync --python {version} --all-extras -q",
            "uv run python scripts/audit_resolved_versions.py",
            (
                'uv run python -c "from DocTest.VisualTest import VisualTest; '
                "from doctest_dashboard.server.app import create_app; "
                "print('imports OK')\""
            ),
        ]
        outcome = "OK"
        for step in steps:
            if subprocess.run(step, shell=True, check=False).returncode != 0:
                outcome = "FAILED"
                break
        results[version] = outcome
    subprocess.run("uv sync --all-extras -q", shell=True, check=False)
    print("\n=== multipython summary ===")
    for version, outcome in results.items():
        print(f"  {version}: {outcome}")
    if any(outcome != "OK" for outcome in results.values()):
        raise Exception("multipython validation failed")


@task
def libdoc(context):
    libraries = [
        ("VisualTest", "DocTest/VisualTest.py"),
        ("PdfTest", "DocTest/PdfTest.py"),
        ("PrintJobTest", "DocTest/PrintJobTests.py"),
        ("WebVisualTest", "DocTest/WebVisualTest.py"),
        ("Ai", "DocTest/Ai/__init__.py"),
    ]

    for name, source_path in libraries:
        source = f"{ROOT}/{source_path}"

        # Document without version in filename
        target = f"{ROOT}/docs/{name}.html"
        cmd = [
            "uv",
            "run",
            "--",
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
            "uv",
            "run",
            "--",
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
