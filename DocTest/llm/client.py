from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional, Sequence

from .config import LLMSettings
from .exceptions import LLMDependencyError
from .prompts import PDF_SYSTEM_PROMPT, VISUAL_SYSTEM_PROMPT
from .types import LLMDecision, LLMDecisionLabel

_RUNTIME_CACHE = {}


def _import_runtime():
    if "agent" in _RUNTIME_CACHE:
        return (
            _RUNTIME_CACHE["agent"],
            _RUNTIME_CACHE["binary_content"],
        )

    try:
        from pydantic_ai import Agent, BinaryContent  # type: ignore import
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LLMDependencyError() from exc

    _RUNTIME_CACHE["agent"] = Agent
    _RUNTIME_CACHE["binary_content"] = BinaryContent
    return Agent, BinaryContent


def create_binary_content(data: bytes, media_type: str):
    """Helper used by callers to wrap binary payloads for the LLM."""
    _, BinaryContent = _import_runtime()
    return BinaryContent(data=data, media_type=media_type)


def _configure_environment(settings: LLMSettings) -> None:
    if settings.provider.startswith("azure") or settings.is_azure:
        if settings.azure_endpoint:
            os.environ["AZURE_OPENAI_ENDPOINT"] = settings.azure_endpoint
        if settings.azure_api_key:
            os.environ["AZURE_OPENAI_API_KEY"] = settings.azure_api_key
        if settings.azure_api_version:
            os.environ["OPENAI_API_VERSION"] = settings.azure_api_version
        if settings.azure_deployment:
            # Provide fallback for downstream helpers that rely on deployment name.
            os.environ.setdefault("AZURE_OPENAI_DEPLOYMENT", settings.azure_deployment)
    else:
        if settings.api_key:
            os.environ["OPENAI_API_KEY"] = settings.api_key
        if settings.base_url:
            os.environ["OPENAI_BASE_URL"] = settings.base_url


def _resolve_model_identifier(settings: LLMSettings, purpose: str) -> Optional[str]:
    model = settings.resolve_model("vision" if purpose == "vision" else "text")
    if not model:
        return None

    provider = settings.provider.strip().lower() if settings.provider else "openai"

    if provider.startswith("azure"):
        deployment = settings.azure_deployment or model
        return f"azure:{deployment}"

    if provider and provider != "openai":
        return f"{provider}:{model}"

    return f"openai:{model}"


def _parse_decision(payload) -> LLMDecision:
    if isinstance(payload, LLMDecision):
        return payload

    if isinstance(payload, dict):
        data = payload
    else:
        text = payload
        if hasattr(payload, "output"):
            text = payload.output  # pydantic-ai RunResult
        if not isinstance(text, str):
            text = str(text)

        # Try to extract JSON object even if surrounded by prose.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
        else:
            candidate = text

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            notes = candidate.strip() or ""
            lowered = notes.lower()
            approve_tokens = ("approve", "approved", "pass", "acceptable")
            reject_tokens = ("reject", "rejected", "fail", "not acceptable")

            if any(token in lowered for token in approve_tokens) and not any(
                token in lowered for token in reject_tokens
            ):
                return LLMDecision(
                    decision=LLMDecisionLabel.APPROVE,
                    reason="LLM approved comparison (fallback parsing).",
                    notes=notes or None,
                )
            if any(token in lowered for token in reject_tokens):
                return LLMDecision(
                    decision=LLMDecisionLabel.REJECT,
                    reason="LLM rejected comparison (fallback parsing).",
                    notes=notes or None,
                )
            return LLMDecision(
                decision=LLMDecisionLabel.FLAG,
                reason="LLM response was not valid JSON",
                notes=notes or None,
            )

    decision_raw = str(data.get("decision", "")).strip().lower()
    if decision_raw not in {label.value for label in LLMDecisionLabel}:
        decision = LLMDecisionLabel.FLAG
    else:
        decision = LLMDecisionLabel(decision_raw)

    confidence_value = data.get("confidence")
    try:
        confidence = float(confidence_value) if confidence_value is not None else None
    except (TypeError, ValueError):
        confidence = None

    reason = data.get("reason")
    notes = data.get("notes")

    return LLMDecision(decision=decision, confidence=confidence, reason=reason, notes=notes)


def _run_agent(
    settings: LLMSettings,
    purpose: str,
    summary: str,
    attachments: Sequence,
    extra_messages: Optional[Iterable[str]] = None,
) -> LLMDecision:
    Agent, _ = _import_runtime()
    _configure_environment(settings)

    identifier = _resolve_model_identifier(settings, purpose)
    if not identifier:
        raise RuntimeError("No LLM model configured for the requested operation.")

    agent = Agent(identifier, output_type=LLMDecision)

    message_blocks: List = []
    message_blocks.append(summary)
    if extra_messages:
        message_blocks.extend(extra_messages)

    for attachment in attachments:
        message_blocks.append(attachment)

    try:
        result = agent.run_sync(message_blocks)
    except Exception as exc:
        partial = getattr(exc, "partial_response", None)
        if partial is not None:
            return _parse_decision(partial)
        raise

    if isinstance(result, LLMDecision):
        return result
    return _parse_decision(result)


def assess_visual_diff(
    settings: LLMSettings,
    textual_summary: str,
    attachments: Sequence,
    extra_messages: Optional[Iterable[str]] = None,
    system_prompt: Optional[str] = None,
) -> LLMDecision:
    prompt = system_prompt.strip() if system_prompt else VISUAL_SYSTEM_PROMPT
    summary = f"{prompt}\n\n{textual_summary.strip()}"
    return _run_agent(settings, "vision", summary, attachments, extra_messages)


def assess_pdf_diff(
    settings: LLMSettings,
    textual_summary: str,
    attachments: Sequence,
    extra_messages: Optional[Iterable[str]] = None,
    system_prompt: Optional[str] = None,
) -> LLMDecision:
    prompt = system_prompt.strip() if system_prompt else PDF_SYSTEM_PROMPT
    summary = f"{prompt}\n\n{textual_summary.strip()}"
    return _run_agent(settings, "text", summary, attachments, extra_messages)
