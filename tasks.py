import pathlib
import subprocess
from invoke import task
import DocTest

project_root = pathlib.Path(__file__).parent.resolve().as_posix()


@task
def tests(context):
    cmd = [
        "coverage",
        "run",
        "-m",
        "pytest",
        f"{project_root}/utest",
    ]
    subprocess.run(" ".join(cmd), shell=True)
    cmd = [
        "coverage",
        "run",
        "-m",
        "robot",
        f"--variable=root:{project_root}",
        f"--outputdir={project_root}/reports",
        f"--loglevel=TRACE:DEBUG",
        f"{project_root}/atest",
    ]
    subprocess.run(" ".join(cmd), shell=True)
    subprocess.run("coverage combine", shell=True)
    subprocess.run("coverage report", shell=True)
    subprocess.run("coverage html", shell=True)


@task
def lint(context):
    subprocess.run(f"mypy {project_root}", shell=True)
    subprocess.run(f"pylint {project_root}/src/OpenApiDriver", shell=True)


@task
def format_code(context):
    subprocess.run(f"python -m black {project_root}", shell=True)
    subprocess.run(f"python -m isort {project_root}", shell=True)
    subprocess.run(f"robotidy {project_root}", shell=True)


@task
def libdoc(context):
    source = f"{project_root}/DocTest/VisualTest.py"
    target = f"{project_root}/docs/VisualTest.html"
    cmd = [
        "python",
        "-m",
        "robot.libdoc",
        source,
        target,
    ]
    subprocess.run(" ".join(cmd), shell=True)
    source = f"{project_root}/DocTest/PdfTest.py"
    target = f"{project_root}/docs/PdfTest.html"
    cmd = [
        "python",
        "-m",
        "robot.libdoc",
        source,
        target,
    ]
    subprocess.run(" ".join(cmd), shell=True)
    source = f"{project_root}/DocTest/PrintJobTests.py"
    target = f"{project_root}/docs/PrintJobTest.html"
    cmd = [
        "python",
        "-m",
        "robot.libdoc",
        source,
        target,
    ]
    subprocess.run(" ".join(cmd), shell=True)


@task
def libspec(context):
    source = f"{project_root}/DocTest/VisualTest.py"
    target = f"{project_root}/docs/VisualTest.libspec"
    cmd = [
        "python",
        "-m",
        "robot.libdoc",
        source,
        target,
    ]
    subprocess.run(" ".join(cmd), shell=True)
    source = f"{project_root}/DocTest/PdfTest.py"
    target = f"{project_root}/docs/PdfTest.libspec"
    cmd = [
        "python",
        "-m",
        "robot.libdoc",
        source,
        target,
    ]
    subprocess.run(" ".join(cmd), shell=True)
    source = f"{project_root}/DocTest/PrintJobTests.py"
    target = f"{project_root}/docs/PrintJobTest.libspec"
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
    with open(f"{project_root}/README.md", "w", encoding="utf-8") as readme:
        doc_string = DocTest.__doc__
        print(doc_string)
        #readme.write(str(doc_string).replace("\\", "\\\\").replace("\\\\*", "\\*"))

@task(tests, libdoc, libspec)
def build(context):
    print("Creating Build")
    # subprocess.run("poetry build", shell=True)


# @task(post=[build])
# def bump_version(context, rule):
#     subprocess.run(f"poetry version {rule}", shell=True)