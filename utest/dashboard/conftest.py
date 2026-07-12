"""Fixtures executing real Robot Framework runs as ingestion test data.

No mocked engine anywhere: suites run the actual DocTest library against
the repository's test data, and the resulting genuine ``output.xml`` files
are what the dashboard ingests.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from doctest_dashboard.config import AppConfig
from doctest_dashboard.db import Database
from doctest_dashboard.server.app import create_app
from helpers import (  # noqa: F401  (re-exported for test modules)
    CAND_IMAGE,
    PDF_CAND,
    PDF_MASK,
    PDF_REF,
    REF_IMAGE,
    REPO_ROOT,
    UTESTDATA,
    run_robot_suite,
)

SIDECAR_SUITE = f"""
*** Settings ***
Library    DocTest.VisualTest    result_json=true    take_screenshots=false

*** Test Cases ***
Passing Comparison
    Compare Images    {REF_IMAGE}    {REF_IMAGE}

Failing Comparison Expected
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images    {REF_IMAGE}    {CAND_IMAGE}

Masked Failing Comparison
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images    {PDF_REF}    {PDF_CAND}    placeholder_file={PDF_MASK}
"""

LEGACY_SUITE = f"""
*** Settings ***
Library    DocTest.VisualTest    take_screenshots=true    screenshot_format=png

*** Test Cases ***
Legacy Failing Comparison
    Run Keyword And Expect Error    The compared images are different.
    ...    Compare Images    {REF_IMAGE}    {CAND_IMAGE}
"""


@pytest.fixture(scope="session")
def sidecar_output_xml(tmp_path_factory) -> Path:
    return run_robot_suite(SIDECAR_SUITE, tmp_path_factory.mktemp("sidecar_run"))


@pytest.fixture(scope="session")
def legacy_output_xml(tmp_path_factory) -> Path:
    return run_robot_suite(LEGACY_SUITE, tmp_path_factory.mktemp("legacy_run"))


@pytest.fixture
def config(tmp_path) -> AppConfig:
    return AppConfig(data_dir=tmp_path / "data")


@pytest.fixture
def database(config) -> Database:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(config.db_path)
    yield db
    db.close()


@pytest.fixture
def client(config, database) -> TestClient:
    app = create_app(config, database)
    return TestClient(app)
