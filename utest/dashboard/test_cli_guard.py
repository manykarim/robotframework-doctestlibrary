"""The console script must degrade gracefully without the [dashboard] extra."""

import sys

from doctest_dashboard.cli import main


def test_missing_dashboard_deps_yield_friendly_error(monkeypatch, capsys):
    # None in sys.modules makes `import fastapi` raise ImportError
    monkeypatch.setitem(sys.modules, "fastapi", None)
    exit_code = main(["serve"])
    assert exit_code == 3
    err = capsys.readouterr().err
    assert "robotframework-doctestlibrary[dashboard]" in err
    assert "Traceback" not in err


def test_present_deps_pass_the_guard():
    from doctest_dashboard.cli import _require_dashboard_dependencies

    assert _require_dashboard_dependencies() is True
