"""Keyword-signature compatibility snapshot (api-compatibility Phase 0).

Fails when a public keyword or argument is removed or renamed, or when an
argument's default changes. Additive changes pass. Signatures are
normalized to argument NAME and DEFAULT only — type-hint rendering varies
between Python versions, but names and defaults are the Robot-facing
contract. Refresh the baseline when adding keywords intentionally:

    uv run python -m utest.test_keyword_signatures
"""

import json
from pathlib import Path

import pytest

BASELINE_PATH = Path(__file__).parent / "keyword_signatures_baseline.json"
LIBRARIES = (
    "DocTest.VisualTest",
    "DocTest.PdfTest",
    "DocTest.PrintJobTests",
    "DocTest.WebVisualTest",
)


def _normalize_arg(arg):
    """'name: type = default' -> 'name=default'; type text is not contract."""
    text = str(arg)
    if "=" in text:
        head, default = text.split("=", 1)
        name = head.split(":", 1)[0].strip()
        return f"{name}={default.strip()}"
    return text.split(":", 1)[0].strip()


def _current_signatures(library):
    from robot.libdoc import LibraryDocumentation

    doc = LibraryDocumentation(library)
    return {kw.name: [_normalize_arg(a) for a in kw.args] for kw in doc.keywords}


@pytest.fixture(scope="module")
def baseline():
    with open(BASELINE_PATH, encoding="utf-8") as file:
        return json.load(file)


@pytest.mark.parametrize("library", LIBRARIES)
def test_keyword_signatures_are_backwards_compatible(library, baseline):
    current = _current_signatures(library)
    problems = []

    for keyword, expected_args in baseline[library].items():
        if keyword not in current:
            problems.append(f"keyword removed/renamed: {keyword}")
            continue
        actual_args = current[keyword]
        # Every baseline argument must still exist with the same spec
        # (name, kind, default) at a position >= its old position.
        for arg in expected_args:
            if arg not in actual_args:
                problems.append(
                    f"{keyword}: argument changed or removed: {arg!r} "
                    f"(current: {actual_args})"
                )

    assert not problems, (
        f"{library} broke keyword compatibility:\n  " + "\n  ".join(problems)
    )


if __name__ == "__main__":
    # Refresh the committed baseline from the current code.
    baseline_data = {library: _current_signatures(library) for library in LIBRARIES}
    BASELINE_PATH.write_text(
        json.dumps(baseline_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"baseline refreshed: {BASELINE_PATH}")
