"""Backwards-compatibility gate over the public keyword surface.

Fails when any baselined keyword or argument disappears or changes its
signature; additions never fail. Regenerate the baseline for intentional
changes: uv run python scripts/update_keyword_surface.py
"""

import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from update_keyword_surface import (  # noqa: E402
    BASELINE_PATH,
    build_surface,
    compare_surfaces,
)


def test_public_keyword_surface_is_backwards_compatible():
    baseline = json.loads(BASELINE_PATH.read_text())
    violations = compare_surfaces(baseline, build_surface())
    assert not violations, (
        "Public keyword surface broke backwards compatibility:\n- "
        + "\n- ".join(violations)
        + "\nIf intentional, regenerate: uv run python scripts/update_keyword_surface.py"
    )


def test_baseline_covers_all_shipped_libraries():
    baseline = json.loads(BASELINE_PATH.read_text())
    assert set(baseline) == {
        "DocTest.VisualTest",
        "DocTest.PdfTest",
        "DocTest.PrintJobTests",
        "DocTest.WebVisualTest",
    }
    assert "Compare Images" in baseline["DocTest.VisualTest"]
    assert "Compare Page To Baseline" in baseline["DocTest.WebVisualTest"]


# -- the comparison rules themselves ------------------------------------------

def _mini_surface():
    return {
        "Lib": {
            "Do Thing": [
                {"name": "path", "kind": "POSITIONAL_OR_NAMED", "required": True, "default": None},
                {"name": "mode", "kind": "POSITIONAL_OR_NAMED", "required": False, "default": "auto"},
            ]
        }
    }


def test_removed_keyword_is_a_violation():
    current = {"Lib": {}}
    assert any("removed" in v for v in compare_surfaces(_mini_surface(), current))


def test_removed_argument_is_a_violation():
    current = copy.deepcopy(_mini_surface())
    current["Lib"]["Do Thing"] = current["Lib"]["Do Thing"][:1]
    assert any("lost argument" in v for v in compare_surfaces(_mini_surface(), current))


def test_changed_default_is_a_violation():
    current = copy.deepcopy(_mini_surface())
    current["Lib"]["Do Thing"][1]["default"] = "manual"
    assert any("changed default" in v for v in compare_surfaces(_mini_surface(), current))


def test_additions_are_not_violations():
    current = copy.deepcopy(_mini_surface())
    current["Lib"]["Do Thing"].append(
        {"name": "extra", "kind": "POSITIONAL_OR_NAMED", "required": False, "default": "x"})
    current["Lib"]["New Keyword"] = []
    current["OtherLib"] = {}
    assert compare_surfaces(_mini_surface(), current) == []
