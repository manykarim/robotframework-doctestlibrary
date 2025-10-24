import pytest
import numpy as np

import DocTest.Ai as ai_module

from DocTest.Ai import Ai
from DocTest.llm.config import LLMSettings
from DocTest.llm.types import LLMDecision, LLMDecisionLabel
from DocTest.Ai.responses import LLMCountResponse, LLMChatResponse, LLMExtractionResult


@pytest.fixture
def ai(monkeypatch):
    monkeypatch.setattr(
        ai_module,
        "load_llm_settings",
        lambda overrides=None: LLMSettings(
            enabled=True,
            visual_enabled=True,
            pdf_enabled=True,
            provider="openai",
            models=["test-model"],
            vision_models=["test-vision"],
            api_key="dummy",
        ),
    )
    return Ai()


def test_get_text_with_llm_returns_text(monkeypatch, ai):
    monkeypatch.setattr(
        ai_module,
        "load_document_pages",
        lambda path, dpi, max_pages=None: [
            {"number": 1, "image": np.zeros((10, 10, 3), dtype=np.uint8), "text": "Invoice 123"},
            {"number": 2, "image": np.zeros((10, 10, 3), dtype=np.uint8), "text": "Page two"},
        ],
    )
    monkeypatch.setattr(
        ai_module,
        "prepare_image_attachments",
        lambda pages: [(1, b"img-bytes")],
    )
    monkeypatch.setattr(
        ai_module,
        "create_binary_content",
        lambda data, media_type: ("binary", data, media_type),
    )
    monkeypatch.setattr(
        ai_module,
        "run_structured_prompt",
        lambda settings, purpose, prompt, attachments, extra_messages, output_type: LLMExtractionResult(text="Result"),
    )
    result = ai.get_text_with_llm("dummy.pdf")
    assert result == "Result"


def test_image_should_contain_passes(monkeypatch, ai):
    monkeypatch.setattr(
        ai_module,
        "load_document_pages",
        lambda path, dpi, max_pages=None: [{"number": 1, "image": np.zeros((5, 5, 3), dtype=np.uint8), "text": "Logo"}],
    )
    monkeypatch.setattr(
        ai_module,
        "prepare_image_attachments",
        lambda pages: [(1, b"img")],
    )
    monkeypatch.setattr(
        ai_module,
        "create_binary_content",
        lambda data, media_type: ("binary", data, media_type),
    )
    monkeypatch.setattr(
        ai_module,
        "run_structured_prompt",
        lambda settings, purpose, prompt, attachments, output_type: LLMDecision(decision=LLMDecisionLabel.APPROVE),
    )
    ai.image_should_contain("dummy.png", expected="Logo")


def test_image_should_contain_negative(monkeypatch, ai):
    monkeypatch.setattr(
        ai_module,
        "load_document_pages",
        lambda path, dpi, max_pages=None: [{"number": 1, "image": np.zeros((5, 5, 3), dtype=np.uint8), "text": ""}],
    )
    monkeypatch.setattr(
        ai_module,
        "prepare_image_attachments",
        lambda pages: [(1, b"img")],
    )
    monkeypatch.setattr(
        ai_module,
        "create_binary_content",
        lambda data, media_type: ("binary", data, media_type),
    )
    monkeypatch.setattr(
        ai_module,
        "run_structured_prompt",
        lambda settings, purpose, prompt, attachments, output_type: LLMDecision(decision=LLMDecisionLabel.REJECT, reason="Not found"),
    )
    with pytest.raises(AssertionError):
        ai.image_should_contain("dummy.png", expected="Logo")


def test_get_item_count_from_image(monkeypatch, ai):
    monkeypatch.setattr(
        ai_module,
        "load_document_pages",
        lambda path, dpi, max_pages=None: [{"number": 1, "image": np.zeros((5, 5, 3), dtype=np.uint8), "text": ""}],
    )
    monkeypatch.setattr(
        ai_module,
        "prepare_image_attachments",
        lambda pages: [(1, b"img")],
    )
    monkeypatch.setattr(
        ai_module,
        "create_binary_content",
        lambda data, media_type: ("binary", data, media_type),
    )
    monkeypatch.setattr(
        ai_module,
        "run_structured_prompt",
        lambda settings, purpose, prompt, attachments, output_type: LLMCountResponse(
            item="tables",
            count=3,
        ),
    )
    count = ai.get_item_count_from_image("dummy.png", "tables")
    assert count == 3


def test_get_item_count_from_image_negative(monkeypatch, ai):
    monkeypatch.setattr(
        ai_module,
        "load_document_pages",
        lambda path, dpi, max_pages=None: [{"number": 1, "image": np.zeros((5, 5, 3), dtype=np.uint8), "text": ""}],
    )
    monkeypatch.setattr(
        ai_module,
        "prepare_image_attachments",
        lambda pages: [(1, b"img")],
    )
    monkeypatch.setattr(
        ai_module,
        "create_binary_content",
        lambda data, media_type: ("binary", data, media_type),
    )
    monkeypatch.setattr(
        ai_module,
        "run_structured_prompt",
        lambda settings, purpose, prompt, attachments, output_type: LLMCountResponse(item="none", count=0),
    )
    count = ai.get_item_count_from_image("dummy.png", "tables")
    assert count == 0


def test_get_text_with_llm_missing_document(monkeypatch, ai):
    monkeypatch.setattr(
        ai_module,
        "load_document_pages",
        lambda path, dpi, max_pages=None: [],
    )
    with pytest.raises(AssertionError):
        ai.get_text_with_llm("missing.pdf")
