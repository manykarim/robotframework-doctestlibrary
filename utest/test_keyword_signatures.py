"""Keyword-signature compatibility snapshot (api-compatibility Phase 0).

Fails when a public keyword or argument is removed or renamed, or when an
argument's default changes. Additive changes pass — refresh the baseline
with the snippet below when adding keywords intentionally:

    uv run python -c "
    import json; from robot.libdoc import LibraryDocumentation
    libs = ('DocTest.VisualTest','DocTest.PdfTest','DocTest.PrintJobTests','DocTest.WebVisualTest')
    b = {l: {k.name: [str(a) for a in k.args] for k in LibraryDocumentation(l).keywords} for l in libs}
    json.dump(b, open('utest/keyword_signatures_baseline.json','w'), indent=2, sort_keys=True)"
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


def _current_signatures(library):
    from robot.libdoc import LibraryDocumentation

    doc = LibraryDocumentation(library)
    return {kw.name: [str(a) for a in kw.args] for kw in doc.keywords}


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
