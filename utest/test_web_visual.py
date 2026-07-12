"""Unit tests for WebVisualTest and the web capture adapters (no robot run)."""

import shutil

import cv2
import numpy as np
import pytest

from DocTest.WebCapture import (
    BrowserLibraryAdapter,
    ElementRect,
    SeleniumLibraryAdapter,
    detect_adapter,
)
from DocTest.WebVisualTest import WebVisualTest, sanitize_baseline_name


class FakeBuiltIn:
    """Records run_keyword calls; canned responses per keyword name."""

    def __init__(self, responses=None, libraries=()):
        self.calls = []
        self.responses = responses or {}
        self.libraries = set(libraries)

    def run_keyword(self, keyword, *args):
        self.calls.append((keyword, args))
        value = self.responses.get(keyword)
        return value(*args) if callable(value) else value

    def get_library_instance(self, name):
        if name not in self.libraries:
            raise RuntimeError(f"No library '{name}' found.")
        return object()


# -- adapters ---------------------------------------------------------------

def test_browser_page_capture_uses_css_scale_and_fullpage(tmp_path):
    fake = FakeBuiltIn({"Browser.Take Screenshot": str(tmp_path / "shot.png")})
    adapter = BrowserLibraryAdapter(fake)
    result = adapter.capture_page(tmp_path / "shot.png", full_page=True)
    keyword, args = fake.calls[0]
    assert keyword == "Browser.Take Screenshot"
    assert f"filename={tmp_path / 'shot'}" in args
    assert "fullPage=True" in args
    assert "scale=css" in args
    assert "disableAnimations=True" in args
    assert result.name == "shot.png"


def test_browser_element_capture_passes_selector(tmp_path):
    fake = FakeBuiltIn({"Browser.Take Screenshot": str(tmp_path / "el.png")})
    BrowserLibraryAdapter(fake).capture_element(tmp_path / "el.png", "id=clock")
    _, args = fake.calls[0]
    assert "selector=id=clock" in args


def test_browser_element_rects_and_dpr():
    fake = FakeBuiltIn({
        "Browser.Get Elements": ["element=1", "element=2"],
        "Browser.Get BoundingBox": {"x": 0, "y": 119.875, "width": 216, "height": 34},
        "Browser.Evaluate JavaScript": 2,
    })
    adapter = BrowserLibraryAdapter(fake)
    rects = adapter.element_rects("id=clock")
    assert len(rects) == 2
    assert rects[0].y == 119.875
    assert adapter.device_pixel_ratio() == 2.0
    assert adapter.capture_scale() == 1.0  # scale=css → CSS pixels


def test_selenium_rejects_full_page(tmp_path):
    adapter = SeleniumLibraryAdapter(FakeBuiltIn())
    with pytest.raises(RuntimeError, match="full-page"):
        adapter.capture_page(tmp_path / "x.png", full_page=True)


def test_selenium_viewport_and_element_capture(tmp_path):
    fake = FakeBuiltIn({
        "SeleniumLibrary.Capture Page Screenshot": str(tmp_path / "page.png"),
        "SeleniumLibrary.Capture Element Screenshot": str(tmp_path / "el.png"),
    })
    adapter = SeleniumLibraryAdapter(fake)
    adapter.capture_page(tmp_path / "page.png", full_page=False)
    adapter.capture_element(tmp_path / "el.png", "id=clock")
    assert fake.calls[0][1] == (f"filename={tmp_path / 'page.png'}",)
    # locator travels as a named argument so '=' inside it survives run_keyword
    assert fake.calls[1][1] == ("locator=id=clock", f"filename={tmp_path / 'el.png'}")


def test_selenium_capture_scale_is_dpr():
    fake = FakeBuiltIn({"SeleniumLibrary.Execute Javascript": 2})
    assert SeleniumLibraryAdapter(fake).capture_scale() == 2.0


def test_element_rect_mask_rounds_outward():
    mask = ElementRect(x=10.6, y=119.875, width=216, height=34).to_mask(scale=2.0)
    assert mask["x"] == 21          # floor(21.2)
    assert mask["y"] == 239         # floor(239.75)
    assert mask["x"] + mask["width"] >= 454   # ceil((10.6+216)*2)
    assert mask["y"] + mask["height"] >= 308  # ceil((119.875+34)*2)
    assert mask["type"] == "coordinates" and mask["unit"] == "px"


def test_detect_adapter_prefers_browser_and_honours_override():
    both = FakeBuiltIn(libraries=["Browser", "SeleniumLibrary"])
    assert isinstance(detect_adapter(builtin=both), BrowserLibraryAdapter)
    forced = detect_adapter("SeleniumLibrary", builtin=both)
    assert isinstance(forced, SeleniumLibraryAdapter)


def test_detect_adapter_without_web_library_names_both():
    with pytest.raises(RuntimeError, match="Browser.*SeleniumLibrary"):
        detect_adapter(builtin=FakeBuiltIn())


def test_detect_adapter_rejects_unknown_override():
    with pytest.raises(RuntimeError, match="Unsupported web_library"):
        detect_adapter("Cypress", builtin=FakeBuiltIn(libraries=["Browser"]))


# -- WebVisualTest baseline lifecycle ----------------------------------------

def _write_png(path, box=False):
    image = np.full((120, 200, 3), 255, dtype=np.uint8)
    if box:
        cv2.rectangle(image, (40, 40), (90, 80), (0, 0, 255), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path


class FakeAdapter:
    """Serves a scripted sequence of images (last one repeats)."""

    library_name = "Fake"

    def __init__(self, tmp_path, sequence, rects=None, scale=1.0, context=None):
        self.tmp = tmp_path
        self.sequence = list(sequence)
        self.captures = 0
        self.rects = rects or {}
        self.scale = scale
        self.rect_reads = 0
        self.context = context if context is not None else {"library": "Fake"}

    def element_rects(self, locator):
        self.rect_reads += 1
        return self.rects.get(locator, [])

    def capture_scale(self):
        return self.scale

    def describe(self):
        return dict(self.context)

    def _next(self, path):
        # stable capture takes two shots per attempt → one spec per PAIR of calls
        spec = self.sequence[min(self.captures // 2, len(self.sequence) - 1)]
        self.captures += 1
        source = self.tmp / f"src_{spec}.png"
        if not source.exists():
            _write_png(source, box=(spec == "diff"))
        shutil.copyfile(source, path)
        return path

    def capture_page(self, path, full_page=True):
        return self._next(path)

    def capture_element(self, path, locator):
        return self._next(path)


def _make_lib(tmp_path, adapter, retry_timeout="1s", retry_interval="50ms"):
    lib = WebVisualTest(
        baseline_directory=str(tmp_path / "baselines"),
        retry_timeout=retry_timeout,
        retry_interval=retry_interval,
    )
    lib._adapter = adapter
    lib._robot_variable = lambda variable, default: str(tmp_path)
    return lib


def test_first_run_creates_baseline(tmp_path):
    lib = _make_lib(tmp_path, FakeAdapter(tmp_path, ["same"]))
    lib.compare_page_to_baseline("home")
    assert (tmp_path / "baselines" / "home.png").exists()


def test_matching_page_passes(tmp_path):
    adapter = FakeAdapter(tmp_path, ["same", "same"])
    lib = _make_lib(tmp_path, adapter)
    lib.compare_page_to_baseline("home")   # creates baseline
    lib.compare_page_to_baseline("home")   # compares, must pass
    assert adapter.captures == 4  # two stable-capture shots per attempt


def test_retry_recovers_from_late_settling_page(tmp_path):
    adapter = FakeAdapter(tmp_path, ["same", "diff", "same"])
    lib = _make_lib(tmp_path, adapter)
    lib.compare_page_to_baseline("home")   # baseline = same
    lib.compare_page_to_baseline("home")   # first attempt diff → retry → same
    assert adapter.captures == 6


def test_persistent_difference_fails_after_retry_window(tmp_path):
    adapter = FakeAdapter(tmp_path, ["same", "diff"])
    lib = _make_lib(tmp_path, adapter, retry_timeout="300ms")
    lib.compare_page_to_baseline("home")
    with pytest.raises(AssertionError):
        lib.compare_page_to_baseline("home")
    assert adapter.captures > 4  # it really retried


def test_zero_retry_timeout_fails_immediately(tmp_path):
    adapter = FakeAdapter(tmp_path, ["same", "diff"])
    lib = _make_lib(tmp_path, adapter, retry_timeout="0")
    lib.compare_page_to_baseline("home")
    with pytest.raises(AssertionError):
        lib.compare_page_to_baseline("home")
    assert adapter.captures == 4


def test_reference_run_overwrites_baseline(tmp_path):
    adapter = FakeAdapter(tmp_path, ["same", "diff"])
    lib = _make_lib(tmp_path, adapter)
    lib.compare_page_to_baseline("home")
    lib.reference_run = True
    lib.compare_page_to_baseline("home")   # diff capture becomes new baseline
    baseline = cv2.imread(str(tmp_path / "baselines" / "home.png"))
    assert (baseline[60, 60] == [0, 0, 255]).all()


def test_element_keyword_uses_element_capture(tmp_path):
    adapter = FakeAdapter(tmp_path, ["same"])
    lib = _make_lib(tmp_path, adapter)
    lib.compare_element_to_baseline("id=header", "header")
    assert (tmp_path / "baselines" / "header.png").exists()


def test_baseline_names_cannot_escape_directory(tmp_path):
    lib = _make_lib(tmp_path, FakeAdapter(tmp_path, ["same"]))
    lib.compare_page_to_baseline("../../etc/evil")
    stored = list((tmp_path / "baselines").glob("*.png"))
    assert len(stored) == 1
    assert stored[0].parent == tmp_path / "baselines"
    assert ".." not in stored[0].name and "/" not in stored[0].name


# -- selector ignore masks ---------------------------------------------------
# The FakeAdapter diff image has a red box at (40,40)-(90,80) in a 200x120 page.

def test_ignore_elements_masks_dynamic_region(tmp_path):
    rects = {"id=clock": [ElementRect(x=38.4, y=38.6, width=53, height=42)]}
    adapter = FakeAdapter(tmp_path, ["same", "diff"], rects=rects)
    lib = _make_lib(tmp_path, adapter, retry_timeout="0")
    lib.compare_page_to_baseline("home")
    lib.compare_page_to_baseline("home", ignore_elements="id=clock")  # must pass


def test_ignore_elements_scaled_by_capture_scale(tmp_path):
    # rects in CSS px, screenshot in device px (scale 2): (19,19,28,23)*2 covers the box
    rects = {"id=clock": [ElementRect(x=19.2, y=19.3, width=28, height=23)]}
    adapter = FakeAdapter(tmp_path, ["same", "diff"], rects=rects, scale=2.0)
    lib = _make_lib(tmp_path, adapter, retry_timeout="0")
    lib.compare_page_to_baseline("home")
    lib.compare_page_to_baseline("home", ignore_elements="id=clock")


def test_ignore_elements_translated_to_element_space(tmp_path):
    # captured element starts at page (100, 200); dynamic child at page (140, 240)
    # → element-space (40, 40), exactly over the diff box origin
    rects = {
        "id=widget": [ElementRect(x=100, y=200, width=200, height=120)],
        "id=child": [ElementRect(x=140, y=240, width=52, height=41)],
    }
    adapter = FakeAdapter(tmp_path, ["same", "diff"], rects=rects)
    lib = _make_lib(tmp_path, adapter, retry_timeout="0")
    lib.compare_element_to_baseline("id=widget", "widget")
    lib.compare_element_to_baseline("id=widget", "widget", ignore_elements="id=child")


def test_ignore_elements_without_matches_is_skipped_not_fatal(tmp_path):
    adapter = FakeAdapter(tmp_path, ["same", "diff"], rects={})
    lib = _make_lib(tmp_path, adapter, retry_timeout="0")
    lib.compare_page_to_baseline("home")
    with pytest.raises(AssertionError):  # diff not masked, comparison still runs
        lib.compare_page_to_baseline("home", ignore_elements="id=gone")


def test_ignore_elements_merges_with_user_mask(tmp_path):
    rects = {"id=clock": [ElementRect(x=38, y=38, width=54, height=44)]}
    adapter = FakeAdapter(tmp_path, ["same", "diff"], rects=rects)
    lib = _make_lib(tmp_path, adapter, retry_timeout="0")
    lib.compare_page_to_baseline("home")
    user_mask = {"page": "all", "type": "coordinates", "unit": "px",
                 "x": 0, "y": 0, "width": 5, "height": 5}
    lib.compare_page_to_baseline("home", ignore_elements="id=clock", mask=user_mask)


def test_ignore_rects_reread_on_each_retry(tmp_path):
    rects = {"id=clock": []}
    adapter = FakeAdapter(tmp_path, ["same", "diff"], rects=rects)
    lib = _make_lib(tmp_path, adapter, retry_timeout="200ms", retry_interval="50ms")
    lib.compare_page_to_baseline("home")
    with pytest.raises(AssertionError):
        lib.compare_page_to_baseline("home", ignore_elements="id=clock")
    assert adapter.rect_reads >= 2  # one read per attempt


def test_semicolon_and_list_locator_forms(tmp_path):
    lib = _make_lib(tmp_path, FakeAdapter(tmp_path, ["same"]))
    assert lib._parse_ignore_elements("id=a; css=.b ;") == ["id=a", "css=.b"]
    assert lib._parse_ignore_elements(["id=a", "id=b"]) == ["id=a", "id=b"]
    assert lib._parse_ignore_elements(None) == []


# -- capture context & split baselines ---------------------------------------

def _sidecars(directory):
    results = directory / "doctest_results"
    if not results.exists():
        return []
    return [f for f in results.glob("*.json") if f.name != "run.json"]


def test_context_and_label_written_to_sidecar(tmp_path, monkeypatch):
    import json
    monkeypatch.chdir(tmp_path)
    context = {"library": "Fake", "browser": "chromium", "viewport": "1280x720",
               "device_pixel_ratio": 1, "url": "https://example.com"}
    adapter = FakeAdapter(tmp_path, ["same", "same"], context=context)
    lib = _make_lib(tmp_path, adapter)
    lib.result_json = True
    lib.compare_page_to_baseline("home")   # create (no sidecar)
    lib.compare_page_to_baseline("home")   # compare → sidecar
    sidecars = _sidecars(tmp_path)
    assert len(sidecars) == 1
    data = json.loads(sidecars[0].read_text())
    assert data["context"]["browser"] == "chromium"
    assert data["context"]["url"] == "https://example.com"
    assert data["name"] == "home"


def test_split_baselines_by_builds_config_paths(tmp_path):
    context = {"library": "Fake", "browser": "chromium", "viewport": "1280x720"}
    adapter = FakeAdapter(tmp_path, ["same"], context=context)
    lib = _make_lib(tmp_path, adapter)
    lib.split_baselines_by = ["browser", "viewport"]
    lib.compare_page_to_baseline("home")
    assert (tmp_path / "baselines" / "chromium" / "1280x720" / "home.png").exists()


def test_split_baselines_label_qualifies_identity(tmp_path, monkeypatch):
    import json
    monkeypatch.chdir(tmp_path)
    context = {"library": "Fake", "browser": "firefox", "viewport": "800x600"}
    adapter = FakeAdapter(tmp_path, ["same", "same"], context=context)
    lib = _make_lib(tmp_path, adapter)
    lib.result_json = True
    lib.split_baselines_by = ["browser"]
    lib.compare_page_to_baseline("home")
    lib.compare_page_to_baseline("home")
    data = json.loads(_sidecars(tmp_path)[0].read_text())
    assert data["name"] == "firefox/home"


def test_split_baselines_missing_context_value_is_deterministic(tmp_path):
    adapter = FakeAdapter(tmp_path, ["same"], context={"library": "Fake"})
    lib = _make_lib(tmp_path, adapter)
    lib.split_baselines_by = ["browser"]
    lib.compare_page_to_baseline("home")
    assert (tmp_path / "baselines" / "unknown-browser" / "home.png").exists()


def test_split_baselines_unknown_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="split_baselines_by supports"):
        WebVisualTest(
            baseline_directory=str(tmp_path), split_baselines_by="browser,os"
        )


@pytest.mark.parametrize("raw,expected", [
    ("home page", "home_page"),
    ("a/b\\c", "a_b_c"),
    ("..hidden..", "hidden"),
    ("Ümläut-page", "ml_ut-page"),
])
def test_sanitize_baseline_name(raw, expected):
    assert sanitize_baseline_name(raw) == expected


def test_sanitize_rejects_empty():
    with pytest.raises(ValueError):
        sanitize_baseline_name("/../")


# -- DOM-assisted comparison ---------------------------------------------------

DOM_A = '{"tag": "body", "children": ["Hello"]}'
DOM_A_REORDERED = '{"children": ["Hello"], "tag": "body"}'
DOM_B = '{"tag": "body", "children": ["Changed"]}'


class DomAdapter(FakeAdapter):
    def __init__(self, tmp_path, sequence, snapshots, **kw):
        super().__init__(tmp_path, sequence, **kw)
        self.snapshots = list(snapshots)
        self.snapshot_calls = 0
        self.snapshot_locators = []

    def dom_snapshot(self, locator=None):
        self.snapshot_locators.append(locator)
        snap = self.snapshots[min(self.snapshot_calls, len(self.snapshots) - 1)]
        self.snapshot_calls += 1
        return snap


def test_dom_baseline_stored_on_create(tmp_path):
    adapter = DomAdapter(tmp_path, ["same"], [DOM_A])
    lib = _make_lib(tmp_path, adapter)
    lib.dom_analysis = True
    lib.compare_page_to_baseline("home")
    assert (tmp_path / "baselines" / "home.dom.json").read_text() == DOM_A


def test_rendering_only_failure_accepted(tmp_path):
    adapter = DomAdapter(tmp_path, ["same", "diff"], [DOM_A, DOM_A_REORDERED])
    lib = _make_lib(tmp_path, adapter, retry_timeout="0")
    lib.dom_analysis = True
    lib.compare_page_to_baseline("home")
    # visual diff, DOM canonically identical → accepted only with the flag
    with pytest.raises(AssertionError):
        lib.compare_page_to_baseline("home")
    lib.compare_page_to_baseline("home", accept_rendering_only=True)


def test_semantic_change_never_auto_accepted(tmp_path):
    adapter = DomAdapter(tmp_path, ["same", "diff"], [DOM_A, DOM_B])
    lib = _make_lib(tmp_path, adapter, retry_timeout="0")
    lib.dom_analysis = True
    lib.compare_page_to_baseline("home")
    with pytest.raises(AssertionError):
        lib.compare_page_to_baseline("home", accept_rendering_only=True)


def test_missing_dom_baseline_never_auto_accepted(tmp_path):
    adapter = DomAdapter(tmp_path, ["same", "diff"], [DOM_A, DOM_A])
    lib = _make_lib(tmp_path, adapter, retry_timeout="0")
    lib.compare_page_to_baseline("home")  # dom_analysis off → no .dom.json stored
    lib.dom_analysis = True
    with pytest.raises(AssertionError):
        lib.compare_page_to_baseline("home", accept_rendering_only=True)


def test_dom_verdict_recorded_in_sidecar_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapter = DomAdapter(tmp_path, ["same", "same"], [DOM_A, DOM_B])
    lib = _make_lib(tmp_path, adapter)
    lib.result_json = True
    lib.dom_analysis = True
    lib.compare_page_to_baseline("home")
    lib.compare_page_to_baseline("home")  # visual pass, DOM changed
    import json as jsonlib
    sidecar = jsonlib.loads(_sidecars(tmp_path)[0].read_text())
    assert sidecar["context"]["dom_analysis"]["verdict"] == "changed"
    assert sidecar["context"]["dom_analysis"]["changes"]


def test_dom_baseline_refreshes_on_visual_pass(tmp_path):
    adapter = DomAdapter(tmp_path, ["same", "same"], [DOM_A, DOM_B])
    lib = _make_lib(tmp_path, adapter)
    lib.dom_analysis = True
    lib.compare_page_to_baseline("home")
    lib.compare_page_to_baseline("home")  # passes → snapshot refreshed to DOM_B
    assert (tmp_path / "baselines" / "home.dom.json").read_text() == DOM_B


def test_element_dom_snapshot_uses_locator(tmp_path):
    adapter = DomAdapter(tmp_path, ["same"], [DOM_A])
    lib = _make_lib(tmp_path, adapter)
    lib.dom_analysis = True
    lib.compare_element_to_baseline("id=widget", "widget")
    assert adapter.snapshot_locators == ["id=widget"]


def test_dom_snapshot_failure_does_not_break_comparison(tmp_path):
    class BrokenDomAdapter(FakeAdapter):
        def dom_snapshot(self, locator=None):
            raise RuntimeError("no js in this session")

    lib = _make_lib(tmp_path, BrokenDomAdapter(tmp_path, ["same", "same"]))
    lib.dom_analysis = True
    lib.compare_page_to_baseline("home")
    lib.compare_page_to_baseline("home")  # still compares normally


# -- optional AI review --------------------------------------------------------

def test_llm_passthrough_with_web_context(tmp_path, monkeypatch):
    pytest.importorskip("pydantic", reason="LLM support requires optional deps")
    import DocTest.VisualTest as visual_module
    from DocTest.llm.config import LLMSettings
    from DocTest.llm.types import LLMDecision, LLMDecisionLabel

    captured = {}

    def fake_load(overrides):
        return LLMSettings(
            enabled=True, visual_enabled=True, pdf_enabled=False,
            provider="openai", models=["fake"], vision_models=["fake"],
            api_key="k", base_url=None, temperature=0.0,
            max_output_tokens=None, request_timeout=5.0,
        )

    def fake_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
        captured["extra_messages"] = list(extra_messages)
        return LLMDecision(decision=LLMDecisionLabel.APPROVE, confidence=1.0, reason="noise")

    def fake_create(data, media_type):
        return {"media_type": media_type}

    monkeypatch.setattr(visual_module, "load_llm_settings", fake_load)
    monkeypatch.setattr(
        visual_module, "_load_visual_llm_runtime",
        lambda: (fake_assess, fake_create, LLMDecisionLabel),
    )
    monkeypatch.chdir(tmp_path)

    context = {"library": "Fake", "browser": "chromium", "viewport": "900x600"}
    adapter = DomAdapter(tmp_path, ["same", "diff"], [DOM_A, DOM_A], context=context)
    lib = _make_lib(tmp_path, adapter, retry_timeout="0")
    lib.dom_analysis = True
    lib.compare_page_to_baseline("home")
    # visual diff + LLM approval with override → keyword passes
    lib.compare_page_to_baseline("home", llm=True, llm_override=True)

    notes = "\n".join(captured["extra_messages"])
    assert "Capture context" in notes
    assert "chromium" in notes
    assert '"verdict": "identical"' in notes


def test_llm_rejection_keeps_failure(tmp_path, monkeypatch):
    pytest.importorskip("pydantic", reason="LLM support requires optional deps")
    import DocTest.VisualTest as visual_module
    from DocTest.llm.config import LLMSettings
    from DocTest.llm.types import LLMDecision, LLMDecisionLabel

    def fake_load(overrides):
        return LLMSettings(
            enabled=True, visual_enabled=True, pdf_enabled=False,
            provider="openai", models=["fake"], vision_models=["fake"],
            api_key="k", base_url=None, temperature=0.0,
            max_output_tokens=None, request_timeout=5.0,
        )

    def fake_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
        return LLMDecision(decision=LLMDecisionLabel.REJECT, confidence=1.0, reason="real bug")

    monkeypatch.setattr(visual_module, "load_llm_settings", fake_load)
    monkeypatch.setattr(
        visual_module, "_load_visual_llm_runtime",
        lambda: (fake_assess, lambda d, m: {"media_type": m}, LLMDecisionLabel),
    )
    monkeypatch.chdir(tmp_path)

    adapter = FakeAdapter(tmp_path, ["same", "diff"])
    lib = _make_lib(tmp_path, adapter, retry_timeout="0")
    lib.compare_page_to_baseline("home")
    with pytest.raises(AssertionError):
        lib.compare_page_to_baseline("home", llm=True, llm_override=True)


def test_half_settled_first_capture_never_poisons_baseline(tmp_path):
    """Discovered via the example gallery: the very first full-page capture can
    reflow the page (scrollbar), differing from every later capture."""

    class SettlingAdapter(FakeAdapter):
        def _next(self, path):
            # call-level sequence: first capture unstable, everything after settled
            spec = "diff" if self.captures == 0 else "same"
            self.captures += 1
            source = self.tmp / f"src_{spec}.png"
            if not source.exists():
                _write_png(source, box=(spec == "diff"))
            shutil.copyfile(source, path)
            return path

    adapter = SettlingAdapter(tmp_path, [])
    lib = _make_lib(tmp_path, adapter, retry_timeout="0")
    lib.compare_page_to_baseline("home")   # baseline must be the SETTLED capture
    lib.compare_page_to_baseline("home")   # same session, must pass


# -- documentation completeness -------------------------------------------------

def test_web_keywords_have_libdoc_documentation_with_examples():
    from robot.libdocpkg import LibraryDocumentation

    doc = LibraryDocumentation("DocTest.WebVisualTest")
    assert "baseline" in doc.doc.lower() and "Quickstart" in doc.doc
    own_keywords = {
        "Compare Page To Baseline",
        "Compare Element To Baseline",
        "Set Baseline Directory",
    }
    documented = {keyword.name: keyword.doc for keyword in doc.keywords}
    assert own_keywords <= set(documented)
    for name in own_keywords:
        text = documented[name]
        assert len(text) > 100, f"{name}: documentation too thin"
        assert "| `" in text, f"{name}: no example row in documentation"
