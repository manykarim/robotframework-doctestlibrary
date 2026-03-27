"""Tests for all LLM decision branches in visual and PDF comparison flows.

Every test uses mocks -- no real LLM API calls are made.
"""

import datetime

import numpy as np
import pytest

pytest.importorskip("pydantic", reason="LLM branch tests require optional dependencies.")

from DocTest.VisualTest import VisualTest
from DocTest.PdfTest import PdfTest
from DocTest.llm.config import LLMSettings
from DocTest.llm.types import LLMDecision, LLMDecisionLabel
from DocTest.llm.exceptions import LLMDependencyError
import DocTest.VisualTest as visual_module
import DocTest.PdfTest as pdf_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_settings(*, visual=False, pdf=False):
    return LLMSettings(
        enabled=True,
        visual_enabled=visual,
        pdf_enabled=pdf,
        provider="openai",
        models=["gpt-4o"],
        vision_models=["gpt-4o-mini"],
        api_key="test-key",
        base_url=None,
        temperature=0.2,
        max_output_tokens=None,
        request_timeout=30.0,
    )


def _make_visual_differences():
    """Return a minimal differences list matching what compare_images produces."""
    dummy_page = type("P", (), {"page_number": 1})()
    return [
        {
            "ref_page": dummy_page,
            "cand_page": dummy_page,
            "message": "diff on page 1",
            "rectangles": [{"x": 0, "y": 0, "width": 10, "height": 10}],
            "score": 0.25,
            "threshold": 0.0,
            "absolute_diff": np.zeros((4, 4), dtype=np.uint8),
            "combined_diff": np.zeros((4, 4, 3), dtype=np.uint8),
            "notes": ["sample-note"],
        }
    ]


def _make_pdf_differences():
    """Return a minimal differences list matching what compare_pdf_documents produces."""
    return [
        {
            "facet": "metadata",
            "description": "Metadata mismatch",
            "details": {"modified": datetime.datetime.now().isoformat()},
        }
    ]


def _fake_create(data, media_type):
    """Mock for create_binary_content."""
    return {"data": data, "media_type": media_type}


def _fake_load_settings_visual(overrides):
    return _dummy_settings(visual=True)


def _fake_load_settings_pdf(overrides):
    return _dummy_settings(pdf=True)


# ===========================================================================
# Part 1a: Visual compare LLM branches
# ===========================================================================


class TestVisualLLMBranches:
    """Test all LLM decision branches for _handle_llm_for_visual_differences."""

    def _call_handler(self, monkeypatch, decision_label, confidence=0.9, reason="ok"):
        """Helper that calls the visual LLM handler with a mocked assess function."""
        vt = VisualTest()
        monkeypatch.setattr(visual_module, "load_llm_settings", _fake_load_settings_visual)

        def fake_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
            return LLMDecision(
                decision=decision_label,
                confidence=confidence,
                reason=reason,
            )

        return vt._handle_llm_for_visual_differences(
            reference_image="ref.png",
            candidate_image="cand.png",
            differences=_make_visual_differences(),
            overrides={"llm_provider": "openai"},
            notes=["test-note"],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_visual_diff_fn=fake_assess,
        )

    def test_approve_decision(self, monkeypatch):
        """APPROVE: LLM says images are acceptable despite pixel differences."""
        result = self._call_handler(monkeypatch, LLMDecisionLabel.APPROVE)
        assert result is not None
        assert result.decision == LLMDecisionLabel.APPROVE
        assert result.is_positive is True
        assert result.confidence == 0.9

    def test_reject_decision(self, monkeypatch):
        """REJECT: LLM says images are truly different."""
        result = self._call_handler(monkeypatch, LLMDecisionLabel.REJECT, reason="real diff")
        assert result is not None
        assert result.decision == LLMDecisionLabel.REJECT
        assert result.is_positive is False

    def test_flag_decision(self, monkeypatch):
        """FLAG: LLM flags differences for manual review."""
        result = self._call_handler(monkeypatch, LLMDecisionLabel.FLAG, reason="needs review")
        assert result is not None
        assert result.decision == LLMDecisionLabel.FLAG
        assert result.is_positive is False

    def test_assess_raises_exception_returns_none(self, monkeypatch):
        """Error/timeout: assess function raises -> handler returns None (fallback)."""
        vt = VisualTest()
        monkeypatch.setattr(visual_module, "load_llm_settings", _fake_load_settings_visual)

        def failing_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
            raise RuntimeError("Timeout contacting LLM API")

        result = vt._handle_llm_for_visual_differences(
            reference_image="ref.png",
            candidate_image="cand.png",
            differences=_make_visual_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_visual_diff_fn=failing_assess,
        )
        assert result is None

    def test_assess_raises_llm_dependency_error_returns_none(self, monkeypatch):
        """LLMDependencyError during assess -> handler returns None."""
        vt = VisualTest()
        monkeypatch.setattr(visual_module, "load_llm_settings", _fake_load_settings_visual)

        def failing_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
            raise LLMDependencyError("Missing deps")

        result = vt._handle_llm_for_visual_differences(
            reference_image="ref.png",
            candidate_image="cand.png",
            differences=_make_visual_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_visual_diff_fn=failing_assess,
        )
        assert result is None

    def test_visual_disabled_returns_none(self, monkeypatch):
        """If visual_enabled is False in settings, handler returns None."""
        vt = VisualTest()

        def settings_disabled(overrides):
            return _dummy_settings(visual=False)

        monkeypatch.setattr(visual_module, "load_llm_settings", settings_disabled)

        result = vt._handle_llm_for_visual_differences(
            reference_image="ref.png",
            candidate_image="cand.png",
            differences=_make_visual_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_visual_diff_fn=lambda *a, **kw: None,
        )
        assert result is None

    def test_settings_load_failure_returns_none(self, monkeypatch):
        """If load_llm_settings raises, handler returns None."""
        vt = VisualTest()

        def settings_explode(overrides):
            raise ValueError("bad config")

        monkeypatch.setattr(visual_module, "load_llm_settings", settings_explode)

        result = vt._handle_llm_for_visual_differences(
            reference_image="ref.png",
            candidate_image="cand.png",
            differences=_make_visual_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_visual_diff_fn=lambda *a, **kw: None,
        )
        assert result is None

    def test_summary_contains_reference_and_candidate(self, monkeypatch):
        """The textual summary sent to the LLM contains file path info."""
        vt = VisualTest()
        monkeypatch.setattr(visual_module, "load_llm_settings", _fake_load_settings_visual)

        captured = {}

        def capture_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
            captured["summary"] = textual_summary
            return LLMDecision(decision=LLMDecisionLabel.APPROVE, confidence=0.8, reason="ok")

        vt._handle_llm_for_visual_differences(
            reference_image="path/to/ref.png",
            candidate_image="path/to/cand.png",
            differences=_make_visual_differences(),
            overrides={},
            notes=["extra-context"],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_visual_diff_fn=capture_assess,
        )
        assert "path/to/ref.png" in captured["summary"]
        assert "path/to/cand.png" in captured["summary"]

    def test_attachments_created_from_diff_images(self, monkeypatch):
        """PNG attachments are created for combined_diff and absolute_diff."""
        vt = VisualTest()
        monkeypatch.setattr(visual_module, "load_llm_settings", _fake_load_settings_visual)

        captured = {}

        def capture_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
            captured["attachments"] = attachments
            return LLMDecision(decision=LLMDecisionLabel.APPROVE, confidence=0.8, reason="ok")

        vt._handle_llm_for_visual_differences(
            reference_image="ref.png",
            candidate_image="cand.png",
            differences=_make_visual_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_visual_diff_fn=capture_assess,
        )
        # Should have attachments for combined_diff and absolute_diff
        assert len(captured["attachments"]) >= 2
        for att in captured["attachments"]:
            assert att["media_type"] == "image/png"

    def test_custom_prompt_forwarded(self, monkeypatch):
        """Custom prompt is forwarded to the assess function."""
        vt = VisualTest()
        monkeypatch.setattr(visual_module, "load_llm_settings", _fake_load_settings_visual)

        captured = {}

        def capture_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
            captured["prompt"] = system_prompt
            return LLMDecision(decision=LLMDecisionLabel.APPROVE, confidence=0.8, reason="ok")

        vt._handle_llm_for_visual_differences(
            reference_image="ref.png",
            candidate_image="cand.png",
            differences=_make_visual_differences(),
            overrides={},
            notes=[],
            custom_prompt="Be strict about fonts.",
            create_binary_content_fn=_fake_create,
            assess_visual_diff_fn=capture_assess,
        )
        assert captured["prompt"] == "Be strict about fonts."


# ===========================================================================
# Part 1b: PDF compare LLM branches
# ===========================================================================


class TestPdfLLMBranches:
    """Test all LLM decision branches for _handle_llm_for_pdf_differences."""

    def _call_handler(self, monkeypatch, tmp_path, decision_label, confidence=0.9, reason="ok"):
        """Helper that calls the PDF LLM handler with a mocked assess function."""
        pdf_test = PdfTest()
        monkeypatch.setattr(pdf_module, "load_llm_settings", _fake_load_settings_pdf)

        ref = tmp_path / "reference.pdf"
        cand = tmp_path / "candidate.pdf"
        ref.write_bytes(b"%PDF-1.7\n%EOF")
        cand.write_bytes(b"%PDF-1.7\n%EOF")

        def fake_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
            return LLMDecision(
                decision=decision_label,
                confidence=confidence,
                reason=reason,
            )

        return pdf_test._handle_llm_for_pdf_differences(
            reference_document=str(ref),
            candidate_document=str(cand),
            differences=_make_pdf_differences(),
            overrides={},
            notes=["pdf-note"],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_pdf_diff_fn=fake_assess,
        )

    def test_approve_decision(self, monkeypatch, tmp_path):
        """APPROVE: LLM says PDFs are acceptable."""
        result = self._call_handler(monkeypatch, tmp_path, LLMDecisionLabel.APPROVE)
        assert result is not None
        assert result.decision == LLMDecisionLabel.APPROVE
        assert result.is_positive is True

    def test_reject_decision(self, monkeypatch, tmp_path):
        """REJECT: LLM says PDFs are truly different."""
        result = self._call_handler(
            monkeypatch, tmp_path, LLMDecisionLabel.REJECT, reason="content mismatch"
        )
        assert result is not None
        assert result.decision == LLMDecisionLabel.REJECT
        assert result.is_positive is False

    def test_flag_decision(self, monkeypatch, tmp_path):
        """FLAG: LLM flags PDFs for manual review."""
        result = self._call_handler(monkeypatch, tmp_path, LLMDecisionLabel.FLAG, reason="review")
        assert result is not None
        assert result.decision == LLMDecisionLabel.FLAG
        assert result.is_positive is False

    def test_assess_raises_exception_returns_none(self, monkeypatch, tmp_path):
        """Error during assess -> handler returns None (fallback to baseline)."""
        pdf_test = PdfTest()
        monkeypatch.setattr(pdf_module, "load_llm_settings", _fake_load_settings_pdf)

        ref = tmp_path / "reference.pdf"
        cand = tmp_path / "candidate.pdf"
        ref.write_bytes(b"%PDF-1.7\n%EOF")
        cand.write_bytes(b"%PDF-1.7\n%EOF")

        def failing_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
            raise RuntimeError("Network error")

        result = pdf_test._handle_llm_for_pdf_differences(
            reference_document=str(ref),
            candidate_document=str(cand),
            differences=_make_pdf_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_pdf_diff_fn=failing_assess,
        )
        assert result is None

    def test_assess_raises_llm_dependency_error_returns_none(self, monkeypatch, tmp_path):
        """LLMDependencyError during assess -> handler returns None."""
        pdf_test = PdfTest()
        monkeypatch.setattr(pdf_module, "load_llm_settings", _fake_load_settings_pdf)

        ref = tmp_path / "reference.pdf"
        cand = tmp_path / "candidate.pdf"
        ref.write_bytes(b"%PDF-1.7\n%EOF")
        cand.write_bytes(b"%PDF-1.7\n%EOF")

        def failing_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
            raise LLMDependencyError("Missing openai package")

        result = pdf_test._handle_llm_for_pdf_differences(
            reference_document=str(ref),
            candidate_document=str(cand),
            differences=_make_pdf_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_pdf_diff_fn=failing_assess,
        )
        assert result is None

    def test_pdf_disabled_returns_none(self, monkeypatch, tmp_path):
        """If pdf_enabled is False in settings, handler returns None."""
        pdf_test = PdfTest()

        def settings_disabled(overrides):
            return _dummy_settings(pdf=False)

        monkeypatch.setattr(pdf_module, "load_llm_settings", settings_disabled)

        ref = tmp_path / "reference.pdf"
        cand = tmp_path / "candidate.pdf"
        ref.write_bytes(b"%PDF-1.7\n%EOF")
        cand.write_bytes(b"%PDF-1.7\n%EOF")

        result = pdf_test._handle_llm_for_pdf_differences(
            reference_document=str(ref),
            candidate_document=str(cand),
            differences=_make_pdf_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_pdf_diff_fn=lambda *a, **kw: None,
        )
        assert result is None

    def test_settings_load_failure_returns_none(self, monkeypatch, tmp_path):
        """If load_llm_settings raises, handler returns None."""
        pdf_test = PdfTest()

        def settings_explode(overrides):
            raise ValueError("corrupt config")

        monkeypatch.setattr(pdf_module, "load_llm_settings", settings_explode)

        ref = tmp_path / "reference.pdf"
        cand = tmp_path / "candidate.pdf"
        ref.write_bytes(b"%PDF-1.7\n%EOF")
        cand.write_bytes(b"%PDF-1.7\n%EOF")

        result = pdf_test._handle_llm_for_pdf_differences(
            reference_document=str(ref),
            candidate_document=str(cand),
            differences=_make_pdf_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_pdf_diff_fn=lambda *a, **kw: None,
        )
        assert result is None

    def test_missing_reference_file_still_works(self, monkeypatch, tmp_path):
        """If reference file does not exist, handler warns but still calls assess."""
        pdf_test = PdfTest()
        monkeypatch.setattr(pdf_module, "load_llm_settings", _fake_load_settings_pdf)

        cand = tmp_path / "candidate.pdf"
        cand.write_bytes(b"%PDF-1.7\n%EOF")
        nonexistent_ref = str(tmp_path / "does_not_exist.pdf")

        captured = {}

        def capture_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
            captured["attachments"] = attachments
            return LLMDecision(decision=LLMDecisionLabel.APPROVE, confidence=0.7, reason="ok")

        result = pdf_test._handle_llm_for_pdf_differences(
            reference_document=nonexistent_ref,
            candidate_document=str(cand),
            differences=_make_pdf_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_pdf_diff_fn=capture_assess,
        )
        # Should still return a decision -- missing file is a warning, not fatal
        assert result is not None
        assert result.decision == LLMDecisionLabel.APPROVE
        # Only 1 attachment (candidate) since reference was missing
        assert len(captured["attachments"]) == 1

    def test_summary_contains_difference_descriptions(self, monkeypatch, tmp_path):
        """The textual summary sent to the LLM contains difference info."""
        pdf_test = PdfTest()
        monkeypatch.setattr(pdf_module, "load_llm_settings", _fake_load_settings_pdf)

        ref = tmp_path / "reference.pdf"
        cand = tmp_path / "candidate.pdf"
        ref.write_bytes(b"%PDF-1.7\n%EOF")
        cand.write_bytes(b"%PDF-1.7\n%EOF")

        captured = {}

        def capture_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
            captured["summary"] = textual_summary
            return LLMDecision(decision=LLMDecisionLabel.APPROVE, confidence=0.8, reason="ok")

        pdf_test._handle_llm_for_pdf_differences(
            reference_document=str(ref),
            candidate_document=str(cand),
            differences=_make_pdf_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_pdf_diff_fn=capture_assess,
        )
        assert "Metadata mismatch" in captured["summary"]
        assert str(ref) in captured["summary"]

    def test_pdf_attachments_contain_both_documents(self, monkeypatch, tmp_path):
        """Both reference and candidate PDFs are attached."""
        pdf_test = PdfTest()
        monkeypatch.setattr(pdf_module, "load_llm_settings", _fake_load_settings_pdf)

        ref = tmp_path / "reference.pdf"
        cand = tmp_path / "candidate.pdf"
        ref.write_bytes(b"%PDF-1.7\n%EOF")
        cand.write_bytes(b"%PDF-1.7\n%EOF")

        captured = {}

        def capture_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
            captured["attachments"] = attachments
            return LLMDecision(decision=LLMDecisionLabel.APPROVE, confidence=0.8, reason="ok")

        pdf_test._handle_llm_for_pdf_differences(
            reference_document=str(ref),
            candidate_document=str(cand),
            differences=_make_pdf_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=_fake_create,
            assess_pdf_diff_fn=capture_assess,
        )
        assert len(captured["attachments"]) == 2
        for att in captured["attachments"]:
            assert att["media_type"] == "application/pdf"


# ===========================================================================
# Part 2: Helper function tests (_coerce_label_value, _decision_equals_flag)
# ===========================================================================


class TestDecisionHelpers:
    """Test the coercion and flag-detection helpers used after LLM decisions."""

    def test_coerce_label_from_enum(self):
        from DocTest.VisualTest import _coerce_label_value

        assert _coerce_label_value(LLMDecisionLabel.APPROVE) == "approve"
        assert _coerce_label_value(LLMDecisionLabel.REJECT) == "reject"
        assert _coerce_label_value(LLMDecisionLabel.FLAG) == "flag"

    def test_coerce_label_from_string(self):
        from DocTest.VisualTest import _coerce_label_value

        assert _coerce_label_value("approve") == "approve"
        assert _coerce_label_value("REJECT") == "REJECT"

    def test_decision_equals_flag_with_enum(self):
        from DocTest.VisualTest import _decision_equals_flag

        assert _decision_equals_flag(LLMDecisionLabel.FLAG, LLMDecisionLabel) is True
        assert _decision_equals_flag(LLMDecisionLabel.APPROVE, LLMDecisionLabel) is False

    def test_decision_equals_flag_without_enum_cls(self):
        from DocTest.VisualTest import _decision_equals_flag

        assert _decision_equals_flag("flag", None) is True
        assert _decision_equals_flag("approve", None) is False

    def test_decision_equals_flag_pdf_module(self):
        from DocTest.PdfTest import _decision_equals_flag

        assert _decision_equals_flag(LLMDecisionLabel.FLAG, LLMDecisionLabel) is True
        assert _decision_equals_flag(LLMDecisionLabel.REJECT, LLMDecisionLabel) is False


# ===========================================================================
# Part 3: Missing dependencies -- graceful degradation
# ===========================================================================


class TestMissingDependencies:
    """Verify that missing LLM dependencies do not crash the comparison flow."""

    def test_load_visual_llm_runtime_raises_on_missing_deps(self, monkeypatch):
        """_load_visual_llm_runtime raises LLMDependencyError when deps missing."""
        import DocTest.VisualTest as vmod

        # Reset the cached runtime so the import is re-attempted
        monkeypatch.setattr(vmod, "_VISUAL_LLM_RUNTIME", None)

        import builtins

        original_import = builtins.__import__

        def block_llm_client(name, *args, **kwargs):
            if "DocTest.llm.client" in name:
                raise ModuleNotFoundError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_llm_client)

        with pytest.raises(LLMDependencyError):
            vmod._load_visual_llm_runtime()

    def test_load_pdf_llm_runtime_raises_on_missing_deps(self, monkeypatch):
        """_load_pdf_llm_runtime raises LLMDependencyError when deps missing."""
        import DocTest.PdfTest as pmod

        monkeypatch.setattr(pmod, "_PDF_LLM_RUNTIME", None)

        import builtins

        original_import = builtins.__import__

        def block_llm_client(name, *args, **kwargs):
            if "DocTest.llm.client" in name:
                raise ModuleNotFoundError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_llm_client)

        with pytest.raises(LLMDependencyError):
            pmod._load_pdf_llm_runtime()

    def test_handler_returns_none_when_create_binary_raises_dep_error(self, monkeypatch):
        """If create_binary_content raises LLMDependencyError, handler returns None."""
        vt = VisualTest()
        monkeypatch.setattr(visual_module, "load_llm_settings", _fake_load_settings_visual)

        def bomb_create(data, media_type):
            raise LLMDependencyError("boom")

        result = vt._handle_llm_for_visual_differences(
            reference_image="ref.png",
            candidate_image="cand.png",
            differences=_make_visual_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=bomb_create,
            assess_visual_diff_fn=lambda *a, **kw: None,
        )
        assert result is None

    def test_pdf_handler_returns_none_when_create_binary_raises_dep_error(
        self, monkeypatch, tmp_path
    ):
        """If create_binary_content raises LLMDependencyError for PDF, handler returns None."""
        pdf_test = PdfTest()
        monkeypatch.setattr(pdf_module, "load_llm_settings", _fake_load_settings_pdf)

        ref = tmp_path / "reference.pdf"
        cand = tmp_path / "candidate.pdf"
        ref.write_bytes(b"%PDF-1.7\n%EOF")
        cand.write_bytes(b"%PDF-1.7\n%EOF")

        def bomb_create(data, media_type):
            raise LLMDependencyError("missing openai")

        result = pdf_test._handle_llm_for_pdf_differences(
            reference_document=str(ref),
            candidate_document=str(cand),
            differences=_make_pdf_differences(),
            overrides={},
            notes=[],
            custom_prompt=None,
            create_binary_content_fn=bomb_create,
            assess_pdf_diff_fn=lambda *a, **kw: None,
        )
        assert result is None


# ===========================================================================
# Part 4: LLMDecision model properties
# ===========================================================================


class TestLLMDecisionModel:
    """Test the LLMDecision pydantic model behaviour."""

    def test_is_positive_for_approve(self):
        d = LLMDecision(decision=LLMDecisionLabel.APPROVE, reason="ok")
        assert d.is_positive is True

    def test_is_positive_for_reject(self):
        d = LLMDecision(decision=LLMDecisionLabel.REJECT, reason="bad")
        assert d.is_positive is False

    def test_is_positive_for_flag(self):
        d = LLMDecision(decision=LLMDecisionLabel.FLAG, reason="unsure")
        assert d.is_positive is False

    def test_extra_fields_ignored(self):
        """Extra fields in LLM JSON response should not cause validation errors."""
        d = LLMDecision(
            decision=LLMDecisionLabel.APPROVE,
            reason="ok",
            confidence=0.95,
            unexpected_field="should be ignored",
        )
        assert d.decision == LLMDecisionLabel.APPROVE
        assert not hasattr(d, "unexpected_field")

    def test_optional_fields_default_to_none(self):
        d = LLMDecision(decision=LLMDecisionLabel.REJECT)
        assert d.confidence is None
        assert d.reason is None
        assert d.notes is None
