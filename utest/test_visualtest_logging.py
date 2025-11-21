import logging

import pytest

from DocTest.VisualTest import VisualTest


@pytest.fixture
def visual_test_instance():
    # Use default settings; Robot Framework variables are unavailable during tests
    return VisualTest(show_diff=False, take_screenshots=False)


def test_verbose_movement_logging_suppressed(visual_test_instance, caplog):
    caplog.set_level(logging.WARNING, logger="DocTest.VisualTest")
    visual_test_instance.verbose_movement_logging = False

    visual_test_instance._log_verbose_warning("suppressed warning")

    assert all("suppressed warning" not in record.message for record in caplog.records)


def test_verbose_movement_logging_emitted(caplog):
    vt = VisualTest(show_diff=False, take_screenshots=False, verbose_movement_logging=True)
    caplog.set_level(logging.WARNING, logger="DocTest.VisualTest")

    vt._log_verbose_warning("emitted warning")

    assert any("emitted warning" in record.message for record in caplog.records)
