import datetime

import numpy as np
import pytest

pytest.importorskip("pydantic", reason="LLM support tests require optional dependencies.")

from DocTest.VisualTest import VisualTest
from DocTest.PdfTest import PdfTest
from DocTest.llm.config import LLMSettings
from DocTest.llm.types import LLMDecision, LLMDecisionLabel
import DocTest.VisualTest as visual_module
import DocTest.PdfTest as pdf_module


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


def test_handle_llm_for_visual_differences_uses_custom_prompt(monkeypatch):
    vt = VisualTest()

    captured = {}

    def fake_load(overrides):
        captured["overrides"] = overrides
        return _dummy_settings(visual=True)

    def fake_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
        captured["prompt"] = system_prompt
        captured["summary"] = textual_summary
        captured["attachments"] = attachments
        return LLMDecision(decision=LLMDecisionLabel.APPROVE, confidence=0.9, reason="ok")

    monkeypatch.setattr(visual_module, "load_llm_settings", fake_load)

    def fake_create(data, media_type):
        return {"data": data, "media_type": media_type}

    dummy_page = type("P", (), {"page_number": 1})()
    differences = [
        {
            "ref_page": dummy_page,
            "cand_page": dummy_page,
            "message": "diff",
            "rectangles": [{"x": 0, "y": 0, "width": 1, "height": 1}],
            "score": 0.3,
            "threshold": 0.0,
            "absolute_diff": np.zeros((2, 2), dtype=np.uint8),
            "combined_diff": np.zeros((2, 2, 3), dtype=np.uint8),
            "notes": ["sample-note"],
        }
    ]

    result = vt._handle_llm_for_visual_differences(
        reference_image="ref.png",
        candidate_image="cand.png",
        differences=differences,
        overrides={"llm_provider": "openai"},
        notes=["extra-note"],
        custom_prompt="Custom visual prompt.",
        create_binary_content_fn=fake_create,
        assess_visual_diff_fn=fake_assess,
    )

    assert result is not None
    assert result.decision == LLMDecisionLabel.APPROVE
    assert captured["prompt"] == "Custom visual prompt."
    assert captured["overrides"]["llm_enabled"] == "true"
    assert any("ref.png" in captured["summary"] for _ in [0])
    # ensure binary payloads were created
    assert captured["attachments"], "Expected PNG attachments for LLM payload."


def test_handle_llm_for_pdf_differences_uses_custom_prompt(monkeypatch, tmp_path):
    pdf_test = PdfTest()

    dummy_reference = tmp_path / "reference.pdf"
    dummy_candidate = tmp_path / "candidate.pdf"
    dummy_reference.write_bytes(b"%PDF-1.7\n%EOF")
    dummy_candidate.write_bytes(b"%PDF-1.7\n%EOF")

    captured = {}

    def fake_load(overrides):
        captured["overrides"] = overrides
        return _dummy_settings(pdf=True)

    def fake_assess(settings, textual_summary, attachments, extra_messages, system_prompt):
        captured["prompt"] = system_prompt
        captured["summary"] = textual_summary
        captured["attachments"] = attachments
        return LLMDecision(decision=LLMDecisionLabel.APPROVE, reason="looks good")

    monkeypatch.setattr(pdf_module, "load_llm_settings", fake_load)

    def fake_create(data, media_type):
        return {"data": data, "media_type": media_type}

    differences = [
        {
            "facet": "metadata",
            "description": "Metadata mismatch",
            "details": {"modified": datetime.datetime.now().isoformat()},
        }
    ]

    result = pdf_test._handle_llm_for_pdf_differences(
        reference_document=str(dummy_reference),
        candidate_document=str(dummy_candidate),
        differences=differences,
        overrides={},
        notes=["pdf-note"],
        custom_prompt="Custom PDF prompt.",
        create_binary_content_fn=fake_create,
        assess_pdf_diff_fn=fake_assess,
    )

    assert result is not None
    assert result.decision == LLMDecisionLabel.APPROVE
    assert captured["prompt"] == "Custom PDF prompt."
    assert captured["overrides"]["llm_pdf_enabled"] == "true"
    assert "Metadata mismatch" in captured["summary"]
    assert len(captured["attachments"]) == 2  # reference + candidate
