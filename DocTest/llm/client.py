from __future__ import annotations

import json
import os
from typing import Any, Iterable, List, Optional, Sequence, Type

from pydantic import BaseModel, ValidationError

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


def _normalise_output(result: Any, output_type: Type[Any]) -> Any:
    if isinstance(result, output_type):
        return result

    direct_output = getattr(result, "output", None)
    if direct_output is not None:
        if isinstance(direct_output, output_type):
            return direct_output
        if issubclass(output_type, BaseModel):
            try:
                return output_type.model_validate(direct_output)
            except (ValidationError, TypeError, ValueError):
                pass

    if issubclass(output_type, BaseModel):
        candidates: List[Any] = []
        if hasattr(result, "parts"):
            for part in getattr(result, "parts", []):
                args = getattr(part, "args", None)
                if args is not None:
                    candidates.append(args)
        if isinstance(result, dict):
            candidates.append(result)
        if isinstance(result, str):
            try:
                candidates.append(json.loads(result))
            except json.JSONDecodeError:
                pass

        for candidate in candidates:
            try:
                return output_type.model_validate(candidate)
            except (ValidationError, TypeError, ValueError):
                continue
        raise AssertionError("LLM returned data that could not be parsed into the expected schema.")

    return result


def _run_agent(
    settings: LLMSettings,
    purpose: str,
    messages: Sequence,
    output_type: Type[Any],
) -> Any:
    Agent, _ = _import_runtime()
    _configure_environment(settings)

    identifier = _resolve_model_identifier(settings, purpose)
    if not identifier:
        raise RuntimeError("No LLM model configured for the requested operation.")

    agent = Agent(identifier, output_type=output_type)

    try:
        raw = agent.run_sync(list(messages))
        return _normalise_output(raw, output_type)
    except Exception as exc:
        partial = getattr(exc, "partial_response", None)
        if partial is not None:
            return _normalise_output(partial, output_type)
        raise


def assess_visual_diff(
    settings: LLMSettings,
    textual_summary: str,
    attachments: Sequence,
    extra_messages: Optional[Iterable[str]] = None,
    system_prompt: Optional[str] = None,
) -> LLMDecision:
    """Assess visual differences using LLM.

    The base VISUAL_SYSTEM_PROMPT is always used to ensure proper JSON schema
    and decision format. If a custom system_prompt is provided, it is appended
    as additional criteria rather than replacing the base prompt entirely.
    This ensures consistent response format while allowing custom decision logic.
    """
    if system_prompt and system_prompt.strip():
        prompt = f"{VISUAL_SYSTEM_PROMPT}\n\nAdditional criteria for this comparison:\n{system_prompt.strip()}"
    else:
        prompt = VISUAL_SYSTEM_PROMPT
    blocks: List = [f"{prompt}\n\n{textual_summary.strip()}"]
    if extra_messages:
        blocks.extend(extra_messages)
    blocks.extend(attachments)
    result = _run_agent(settings, "vision", blocks, LLMDecision)
    if isinstance(result, LLMDecision):
        return result
    return _parse_decision(result)


def assess_pdf_diff(
    settings: LLMSettings,
    textual_summary: str,
    attachments: Sequence,
    extra_messages: Optional[Iterable[str]] = None,
    system_prompt: Optional[str] = None,
) -> LLMDecision:
    """Assess PDF differences using LLM.

    The base PDF_SYSTEM_PROMPT is always used to ensure proper JSON schema
    and decision format. If a custom system_prompt is provided, it is appended
    as additional criteria rather than replacing the base prompt entirely.
    This ensures consistent response format while allowing custom decision logic.
    """
    if system_prompt and system_prompt.strip():
        prompt = f"{PDF_SYSTEM_PROMPT}\n\nAdditional criteria for this comparison:\n{system_prompt.strip()}"
    else:
        prompt = PDF_SYSTEM_PROMPT
    blocks: List = [f"{prompt}\n\n{textual_summary.strip()}"]
    if extra_messages:
        blocks.extend(extra_messages)
    blocks.extend(attachments)
    result = _run_agent(settings, "text", blocks, LLMDecision)
    if isinstance(result, LLMDecision):
        return result
    return _parse_decision(result)


def run_structured_prompt(
    settings: LLMSettings,
    purpose: str,
    prompt: str,
    attachments: Sequence,
    extra_messages: Optional[Iterable[str]] = None,
    output_type: Type[Any] = str,
) -> Any:
    blocks: List = [prompt]
    if extra_messages:
        blocks.extend(extra_messages)
    blocks.extend(attachments)
    return _run_agent(settings, purpose, blocks, output_type)
