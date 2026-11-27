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


# -----------------------------------------------------------------------------
# Tests for custom prompt appending behavior in assess_* functions
# -----------------------------------------------------------------------------


def test_assess_visual_diff_appends_custom_prompt_to_base():
    """Custom prompts should be appended to base prompt, not replace it."""
    from DocTest.llm.client import assess_visual_diff
    from DocTest.llm.prompts import VISUAL_SYSTEM_PROMPT

    captured_prompt = {}

    def mock_run_agent(settings, purpose, messages, output_type):
        # The first message contains the prompt
        captured_prompt["full"] = messages[0] if messages else ""
        return LLMDecision(decision=LLMDecisionLabel.REJECT, reason="test")

    import DocTest.llm.client as client_module
    original_run_agent = client_module._run_agent

    try:
        client_module._run_agent = mock_run_agent

        settings = _dummy_settings(visual=True)
        custom = "Minor font differences shall fail the comparison"

        assess_visual_diff(
            settings=settings,
            textual_summary="Test summary",
            attachments=[],
            system_prompt=custom,
        )

        full_prompt = captured_prompt["full"]
        # Verify base prompt is included
        assert "Return a JSON object" in full_prompt, "Base prompt JSON schema should be present"
        assert '"decision": "approve" | "reject" | "flag"' in full_prompt, "Decision options should be present"
        # Verify custom prompt is appended
        assert "Additional criteria for this comparison:" in full_prompt, "Custom prompt header should be present"
        assert custom in full_prompt, "Custom criteria should be in prompt"
        # Verify order: base prompt comes before custom
        base_pos = full_prompt.find("Return a JSON object")
        custom_pos = full_prompt.find(custom)
        assert base_pos < custom_pos, "Base prompt should come before custom criteria"
    finally:
        client_module._run_agent = original_run_agent


def test_assess_visual_diff_uses_only_base_prompt_when_no_custom():
    """Without custom prompt, only base prompt should be used."""
    from DocTest.llm.client import assess_visual_diff
    from DocTest.llm.prompts import VISUAL_SYSTEM_PROMPT

    captured_prompt = {}

    def mock_run_agent(settings, purpose, messages, output_type):
        captured_prompt["full"] = messages[0] if messages else ""
        return LLMDecision(decision=LLMDecisionLabel.APPROVE, reason="test")

    import DocTest.llm.client as client_module
    original_run_agent = client_module._run_agent

    try:
        client_module._run_agent = mock_run_agent

        settings = _dummy_settings(visual=True)

        assess_visual_diff(
            settings=settings,
            textual_summary="Test summary",
            attachments=[],
            system_prompt=None,
        )

        full_prompt = captured_prompt["full"]
        assert "Additional criteria" not in full_prompt, "No custom criteria header when no custom prompt"
        assert VISUAL_SYSTEM_PROMPT in full_prompt, "Base prompt should be present"
    finally:
        client_module._run_agent = original_run_agent


def test_assess_pdf_diff_appends_custom_prompt_to_base():
    """Custom prompts should be appended to base prompt for PDF assessment."""
    from DocTest.llm.client import assess_pdf_diff
    from DocTest.llm.prompts import PDF_SYSTEM_PROMPT

    captured_prompt = {}

    def mock_run_agent(settings, purpose, messages, output_type):
        captured_prompt["full"] = messages[0] if messages else ""
        return LLMDecision(decision=LLMDecisionLabel.REJECT, reason="test")

    import DocTest.llm.client as client_module
    original_run_agent = client_module._run_agent

    try:
        client_module._run_agent = mock_run_agent

        settings = _dummy_settings(pdf=True)
        custom = "Reject any metadata differences"

        assess_pdf_diff(
            settings=settings,
            textual_summary="Test PDF summary",
            attachments=[],
            system_prompt=custom,
        )

        full_prompt = captured_prompt["full"]
        # Verify base prompt is included
        assert "Respond as a JSON object" in full_prompt, "Base PDF prompt should be present"
        assert '"decision": "approve" | "reject" | "flag"' in full_prompt, "Decision options should be present"
        # Verify custom prompt is appended
        assert "Additional criteria for this comparison:" in full_prompt
        assert custom in full_prompt
    finally:
        client_module._run_agent = original_run_agent


def test_assess_visual_diff_handles_empty_custom_prompt():
    """Empty or whitespace-only custom prompts should be treated as no custom prompt."""
    from DocTest.llm.client import assess_visual_diff

    captured_prompt = {}

    def mock_run_agent(settings, purpose, messages, output_type):
        captured_prompt["full"] = messages[0] if messages else ""
        return LLMDecision(decision=LLMDecisionLabel.APPROVE, reason="test")

    import DocTest.llm.client as client_module
    original_run_agent = client_module._run_agent

    try:
        client_module._run_agent = mock_run_agent

        settings = _dummy_settings(visual=True)

        # Test with empty string
        assess_visual_diff(settings=settings, textual_summary="Test", attachments=[], system_prompt="")
        assert "Additional criteria" not in captured_prompt["full"]

        # Test with whitespace only
        assess_visual_diff(settings=settings, textual_summary="Test", attachments=[], system_prompt="   ")
        assert "Additional criteria" not in captured_prompt["full"]
    finally:
        client_module._run_agent = original_run_agent


def test_run_agent_retries_on_unexpected_model_behavior(monkeypatch):
    """Remote models occasionally exhaust output retries with malformed
    responses; _run_agent re-runs once before giving up."""
    from DocTest.llm import client as client_module

    class UnexpectedModelBehavior(Exception):
        pass

    calls = {"count": 0, "kwargs": None}

    class FakeAgent:
        def __init__(self, identifier, **kwargs):
            calls["kwargs"] = kwargs

        def run_sync(self, messages):
            calls["count"] += 1
            if calls["count"] == 1:
                raise UnexpectedModelBehavior("Exceeded maximum output retries (1)")
            return "recovered"

    monkeypatch.setattr(client_module, "_import_runtime", lambda: (FakeAgent, None))
    monkeypatch.setattr(client_module, "_configure_environment", lambda settings: None)
    monkeypatch.setattr(client_module, "_normalise_output", lambda raw, output_type: raw)

    settings = _dummy_settings()
    result = client_module._run_agent(settings, "text", ["prompt"], str)
    assert result == "recovered"
    assert calls["count"] == 2
    assert calls["kwargs"]["retries"] == settings.output_retries >= 3
    assert "output_retries" not in calls["kwargs"]


def test_run_agent_gives_up_after_one_retry(monkeypatch):
    from DocTest.llm import client as client_module

    class UnexpectedModelBehavior(Exception):
        pass

    class AlwaysFailingAgent:
        def __init__(self, identifier, **kwargs):
            pass

        def run_sync(self, messages):
            raise UnexpectedModelBehavior("Exceeded maximum output retries (3)")

    monkeypatch.setattr(client_module, "_import_runtime", lambda: (AlwaysFailingAgent, None))
    monkeypatch.setattr(client_module, "_configure_environment", lambda settings: None)

    with pytest.raises(UnexpectedModelBehavior):
        client_module._run_agent(_dummy_settings(), "text", ["prompt"], str)


def test_output_retries_setting_parses_from_env(monkeypatch):
    from DocTest.llm.config import load_llm_settings

    monkeypatch.setenv("DOCTEST_LLM_OUTPUT_RETRIES", "7")
    assert load_llm_settings().output_retries == 7
    monkeypatch.delenv("DOCTEST_LLM_OUTPUT_RETRIES")
    assert load_llm_settings().output_retries == 3
    assert load_llm_settings({"llm_output_retries": 2}).output_retries == 2


@pytest.mark.parametrize(
    "provider, expected",
    [
        ("openai", "openai-chat:gpt-4o"),
        ("", "openai-chat:gpt-4o"),
        ("anthropic", "anthropic:gpt-4o"),
    ],
)
def test_model_identifier_keeps_chat_completions_for_openai(provider, expected):
    """pydantic-ai 2.x maps the bare 'openai:' prefix to the Responses API;
    the OpenAI provider (incl. OpenAI-compatible base URLs) must stay on
    Chat Completions on every supported runtime version."""
    from DocTest.llm.client import _resolve_model_identifier

    settings = _dummy_settings()
    settings.provider = provider
    assert _resolve_model_identifier(settings, "text") == expected


def test_azure_identifier_uses_deployment():
    from DocTest.llm.client import _resolve_model_identifier

    settings = _dummy_settings()
    settings.provider = "azure"
    settings.azure_deployment = "my-deployment"
    assert _resolve_model_identifier(settings, "text") == "azure:my-deployment"


def test_runtime_import_suppresses_banner_but_respects_user_choice(monkeypatch):
    from DocTest.llm import client as client_module

    monkeypatch.setattr(client_module, "_RUNTIME_CACHE", {})
    monkeypatch.delenv("PYDANTIC_AI_NO_BANNER", raising=False)
    client_module._import_runtime()
    assert client_module.os.environ["PYDANTIC_AI_NO_BANNER"] == "1"

    monkeypatch.setattr(client_module, "_RUNTIME_CACHE", {})
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "0")
    client_module._import_runtime()
    assert client_module.os.environ["PYDANTIC_AI_NO_BANNER"] == "0"


def test_output_retry_budget_bounds_model_calls(monkeypatch):
    """With the real pydantic-ai Agent, a budget of 3 means one initial call
    plus three output-validation retries on persistently malformed output."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from DocTest.llm import client as client_module

    calls = {"count": 0}

    def malformed(messages, info):
        calls["count"] += 1
        return ModelResponse(parts=[TextPart("not json at all")])

    monkeypatch.setattr(client_module, "_configure_environment", lambda settings: None)
    monkeypatch.setattr(
        client_module, "_resolve_model_identifier", lambda settings, purpose: FunctionModel(malformed)
    )
    settings = _dummy_settings()
    settings.output_retries = 3

    with pytest.raises(Exception):
        client_module._run_agent(settings, "text", ["prompt"], LLMDecision)
    # _run_agent re-runs once after UnexpectedModelBehavior: 2 runs x (1 + 3)
    assert calls["count"] == 8


@pytest.mark.parametrize(
    "configured, expected",
    [
        ({}, {}),
        (
            {"max_output_tokens": 2048, "temperature": 0.0, "request_timeout": 45.0},
            {"max_tokens": 2048, "temperature": 0.0, "timeout": 45.0},
        ),
    ],
)
def test_configured_request_settings_reach_the_model(monkeypatch, configured, expected):
    """max_output_tokens, temperature and request_timeout are sent with each
    request; unset values are omitted so the provider default applies."""
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from DocTest.llm import client as client_module

    seen = {}

    def respond(messages, info):
        seen.update(info.model_settings or {})
        return ModelResponse(parts=[TextPart("done")])

    monkeypatch.setattr(client_module, "_configure_environment", lambda settings: None)
    monkeypatch.setattr(
        client_module, "_resolve_model_identifier", lambda settings, purpose: FunctionModel(respond)
    )
    settings = _dummy_settings()
    settings.temperature = None
    settings.request_timeout = None
    for key, value in configured.items():
        setattr(settings, key, value)

    client_module._run_agent(settings, "text", ["prompt"], str)
    assert {key: seen[key] for key in ("max_tokens", "temperature", "timeout") if key in seen} == expected


def test_request_settings_parse_from_env_and_overrides(monkeypatch):
    from DocTest.llm.config import load_llm_settings

    for name in ("DOCTEST_LLM_TEMPERATURE", "DOCTEST_LLM_MAX_OUTPUT_TOKENS", "DOCTEST_LLM_REQUEST_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    defaults = load_llm_settings()
    assert (defaults.temperature, defaults.max_output_tokens, defaults.request_timeout) == (None, None, None)

    monkeypatch.setenv("DOCTEST_LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("DOCTEST_LLM_MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("DOCTEST_LLM_REQUEST_TIMEOUT", "90")
    from_env = load_llm_settings()
    assert (from_env.temperature, from_env.max_output_tokens, from_env.request_timeout) == (0.7, 4096, 90.0)

    # an explicit numeric 0 is a value, not "unset"
    assert load_llm_settings({"llm_temperature": 0.0}).temperature == 0.0
