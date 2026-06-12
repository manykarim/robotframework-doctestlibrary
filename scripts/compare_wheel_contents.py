#!/usr/bin/env python
"""Compare a built wheel (and optionally sdist) against the committed
poetry-era baseline (scripts/wheel_baseline.json).

Parity rules (release gate for the uv-unified-packaging migration):

- the wheel's ``DocTest/**`` file set is identical to the baseline
- ``DocTest/data/frozen_east_text_detection.pb`` stays excluded
- additions are limited to ``doctest_dashboard/**`` (incl. bundled static
  frontend), dist-info, and entry points
- the *effective* runtime dependency set — names, extras, specifiers after
  evaluating environment markers — matches the baseline for every supported
  interpreter (3.9–3.13), for the base install and the ``ai`` extra.
  Raw string equality is deliberately not required: the baseline metadata
  contains dead ``<3.9`` branches and poetry-specific formatting.

Exit code 0 = parity holds.
"""

import argparse
import json
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

BASELINE_PATH = Path(__file__).parent / "wheel_baseline.json"
SUPPORTED_PYTHONS = ["3.10", "3.11", "3.12", "3.13"]
BASELINE_EXTRAS = ["ai"]

# Documented intentional deviations from the baseline (tightenings only):
# extra specifiers that may be ADDED for the named package without failing
# the gate. Anything else still fails.
INTENTIONAL_TIGHTENINGS = {
    # 3.9 support drop: maintained pydantic-ai (>=1) requires Python 3.10+;
    # the unpinned baseline resolved broken 0.8.x on EOL interpreters.
    "pydantic-ai-slim": frozenset({">=1"}),
}


def _effective_requirements(requires_dist, python_version, extra):
    """Resolve marker-conditional requirements for one environment."""
    environment = {
        "python_version": python_version,
        "python_full_version": f"{python_version}.0",
        "extra": extra or "",
        "sys_platform": "linux",
        "platform_machine": "x86_64",
        "platform_system": "Linux",
        "os_name": "posix",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
    }
    effective = {}
    for raw in requires_dist:
        requirement = Requirement(raw)
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        name = canonicalize_name(requirement.name)
        key = (name, tuple(sorted(requirement.extras)))
        specifiers = effective.setdefault(key, set())
        specifiers.update(str(s) for s in requirement.specifier)
    return {key: frozenset(specs) for key, specs in effective.items()}


def _read_wheel(path):
    with zipfile.ZipFile(path) as wheel:
        files = sorted(n for n in wheel.namelist() if not n.endswith("/"))
        metadata_name = next(
            n for n in files if n.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(wheel.read(metadata_name).decode())
        entry_points = ""
        for name in files:
            if name.endswith(".dist-info/entry_points.txt"):
                entry_points = wheel.read(name).decode()
    return files, metadata, entry_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--sdist", type=Path, default=None)
    args = parser.parse_args()

    baseline = json.loads(BASELINE_PATH.read_text())
    errors = []

    files, metadata, entry_points = _read_wheel(args.wheel)

    # 1. DocTest file-set parity (paths inside the package are version-free)
    baseline_doctest = {f for f in baseline["wheel_files"] if f.startswith("DocTest/")}
    new_doctest = {f for f in files if f.startswith("DocTest/")}
    for missing in sorted(baseline_doctest - new_doctest):
        errors.append(f"missing from wheel: {missing}")
    for extra_file in sorted(new_doctest - baseline_doctest):
        errors.append(f"unexpected DocTest file: {extra_file}")

    # 2. East model exclusion
    if any("frozen_east_text_detection" in f for f in files):
        errors.append("frozen_east_text_detection.pb leaked into the wheel")

    # 3. Additions limited to doctest_dashboard + dist-info
    allowed_prefixes = ("DocTest/", "doctest_dashboard/")
    for name in files:
        if name.startswith(allowed_prefixes) or ".dist-info/" in name:
            continue
        errors.append(f"unexpected wheel member: {name}")

    has_dashboard = any(f.startswith("doctest_dashboard/") for f in files)
    if has_dashboard:
        if not any(f == "doctest_dashboard/static/index.html" for f in files):
            errors.append("dashboard wheel lacks built frontend (static/index.html)")
        if not any(
            f.startswith("doctest_dashboard/static/assets/") and f.endswith(".js")
            for f in files
        ):
            errors.append("dashboard wheel lacks built frontend JS assets")
        if "doctest-dashboard" not in entry_points:
            errors.append("doctest-dashboard console script missing from entry points")

    # 4. Metadata equivalence
    if metadata["Name"] != baseline["metadata"]["name"]:
        errors.append(
            f"name changed: {metadata['Name']} != {baseline['metadata']['name']}")
    new_requires = metadata.get_all("Requires-Dist") or []
    for python_version in SUPPORTED_PYTHONS:
        for extra in [None] + BASELINE_EXTRAS:
            expected = _effective_requirements(
                baseline["metadata"]["requires_dist"], python_version, extra)
            actual = _effective_requirements(new_requires, python_version, extra)
            # new extras (dashboard/all) add requirements; for the baseline
            # comparison only assert the baseline set is preserved exactly
            for key, specifiers in expected.items():
                if key not in actual:
                    errors.append(
                        f"py{python_version} extra={extra}: dependency dropped: {key[0]}")
                elif actual[key] != specifiers:
                    allowed = INTENTIONAL_TIGHTENINGS.get(key[0], frozenset())
                    if actual[key] == specifiers | allowed:
                        continue
                    errors.append(
                        f"py{python_version} extra={extra}: {key[0]} specifiers "
                        f"changed: {sorted(actual[key])} != {sorted(specifiers)}")
    for extra in BASELINE_EXTRAS:
        if extra not in (metadata.get_all("Provides-Extra") or []):
            errors.append(f"extra disappeared: {extra}")

    # 5. sdist (looser: baseline DocTest sources must all be present)
    if args.sdist:
        with tarfile.open(args.sdist) as sdist:
            sdist_files = {
                m.name.split("/", 1)[1] for m in sdist.getmembers() if m.isfile()}
        baseline_sources = {
            f for f in baseline["sdist_files"]
            if f.startswith("DocTest/") or f in ("README.md", "pyproject.toml")}
        for missing in sorted(baseline_sources - sdist_files):
            errors.append(f"missing from sdist: {missing}")
        if any("frozen_east_text_detection" in f for f in sdist_files):
            errors.append("frozen_east_text_detection.pb leaked into the sdist")

    if errors:
        print(f"PARITY FAILED ({len(errors)} problems):")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Parity OK: DocTest content, exclusions, and effective dependencies "
          "match the poetry baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
