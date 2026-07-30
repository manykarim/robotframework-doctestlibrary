"""Library-level template detection method (issue #127).

`Image Should Contain Template` took a per-call `detection` argument with no
way to set it once at import. These tests pin the new `template_detection`
import argument / `Set Template Detection` keyword, the resolution order,
and that existing suites are unaffected.
"""

from unittest.mock import patch

import pytest

from DocTest.VisualTest import VisualTest


# --- library-level setting -------------------------------------------------

def test_unset_by_default():
    assert VisualTest().template_detection is None


@pytest.mark.parametrize(
    "given,expected",
    [
        ("sift", "sift"),
        ("orb", "orb"),
        ("template", "template"),
        ("SIFT", "sift"),
        ("  Sift  ", "sift"),
        ("classic", "template"),
    ],
)
def test_import_argument_is_normalized(given, expected):
    assert VisualTest(template_detection=given).template_detection == expected


@pytest.mark.parametrize("bad", ["bogus", "text"])
def test_invalid_import_argument_rejected(bad):
    with pytest.raises(ValueError, match="Unsupported template detection method"):
        VisualTest(template_detection=bad)
    # the message must list the supported values
    with pytest.raises(ValueError, match="orb, sift, template"):
        VisualTest(template_detection=bad)


def test_setter_keyword_sets_and_resets():
    tester = VisualTest()
    tester.set_template_detection("sift")
    assert tester.template_detection == "sift"
    tester.set_template_detection("template")
    assert tester.template_detection == "template"


def test_setter_keyword_rejects_invalid():
    with pytest.raises(ValueError, match="Unsupported template detection method"):
        VisualTest().set_template_detection("text")


def test_movement_detection_does_not_leak_into_template_detection():
    """movement_detection supports 'text', which is invalid here — keep them apart."""
    tester = VisualTest(movement_detection="text")
    assert tester.movement_detection == "text"
    assert tester.template_detection is None
    assert _resolved(tester) == "template"


def test_documented_classic_alias_accepted_for_movement_detection():
    """README documents movement_detection=classic; it must resolve to template."""
    assert VisualTest(movement_detection="classic").movement_detection == "template"


# --- resolution order inside the keyword ----------------------------------

class _StopBeforeIO(Exception):
    """Raised in place of loading images, once resolution has happened."""


def _resolved(tester, **kwargs):
    """Return the detection method the keyword resolves to.

    Resolution happens at the top of the keyword, so we spy on the normalizer
    and abort at the first image load — no fixtures or disk access needed.
    """
    seen = {}
    original = VisualTest._normalize_template_detection

    def spy(self, value):
        seen["detection"] = original(self, value)
        return seen["detection"]

    with patch.object(VisualTest, "_normalize_template_detection", spy), patch(
        "DocTest.VisualTest.DocumentRepresentation", side_effect=_StopBeforeIO
    ):
        with pytest.raises(_StopBeforeIO):
            tester.image_should_contain_template("img.png", "tpl.png", **kwargs)
    return seen["detection"]


def test_resolution_defaults_to_template_when_nothing_set():
    assert _resolved(VisualTest()) == "template"


def test_resolution_uses_library_setting():
    assert _resolved(VisualTest(template_detection="sift")) == "sift"


def test_resolution_uses_setter_value():
    tester = VisualTest()
    tester.set_template_detection("orb")
    assert _resolved(tester) == "orb"


def test_explicit_call_argument_wins_over_library_setting():
    tester = VisualTest(template_detection="sift")
    assert _resolved(tester, detection="template") == "template"


def test_empty_detection_argument_counts_as_not_provided():
    tester = VisualTest(template_detection="sift")
    assert _resolved(tester, detection="") == "sift"


def test_explicit_call_argument_is_normalized():
    assert _resolved(VisualTest(), detection="SIFT") == "sift"


def test_invalid_call_argument_raises_before_file_access():
    """Validation is up front, so a bad value fails without touching the disk."""
    with patch("DocTest.VisualTest.DocumentRepresentation") as doc:
        with pytest.raises(ValueError, match="Unsupported template detection method"):
            VisualTest().image_should_contain_template(
                "img.png", "tpl.png", detection="bogus"
            )
        doc.assert_not_called()


# --- subclass wiring ------------------------------------------------------

def test_webvisualtest_forwards_the_setting():
    """WebVisualTest passes **kwargs through, and unknown keys are swallowed
    silently — so a wiring regression would be invisible without this test."""
    from DocTest.WebVisualTest import WebVisualTest

    assert WebVisualTest(template_detection="sift").template_detection == "sift"
