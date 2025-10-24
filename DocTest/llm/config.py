from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]


_DOTENV_LOADED = False
_ENV_SNAPSHOT: Dict[str, str] = {}

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"

DEFAULT_MODELS = ["gpt-5", "gpt-4o"]
DEFAULT_VISION_MODELS = ["gpt-5-mini", "gpt-4o-mini"]


def _load_dotenv_once() -> None:
    global _DOTENV_LOADED, _ENV_SNAPSHOT

    if _DOTENV_LOADED:
        return

    if ENV_FILE.exists():
        if load_dotenv is not None:
            load_dotenv(dotenv_path=ENV_FILE, override=True)
        else:
            _manual_parse_env_file(ENV_FILE)

    _ENV_SNAPSHOT = dict(os.environ)
    _DOTENV_LOADED = True


def _manual_parse_env_file(env_file: Path) -> None:
    """Lightweight .env parser used when python-dotenv is not available."""
    try:
        content = env_file.read_text(encoding="utf-8")
    except OSError:
        return

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]

        os.environ[key] = value


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _split_models(value: Optional[str], defaults: Iterable[str]) -> List[str]:
    if not value:
        return [model for model in defaults if model]
    tokens = [token.strip() for token in value.split(",")]
    return [token for token in tokens if token]


def _as_float(value: Optional[str], default: Optional[float]) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _as_int(value: Optional[str], default: Optional[int]) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class LLMSettings:
    enabled: bool = False
    visual_enabled: bool = False
    pdf_enabled: bool = False
    provider: str = "openai"
    models: List[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    vision_models: List[str] = field(default_factory=lambda: list(DEFAULT_VISION_MODELS))
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_output_tokens: Optional[int] = None
    request_timeout: Optional[float] = 30.0

    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    azure_api_version: Optional[str] = None
    azure_api_key: Optional[str] = None

    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def is_azure(self) -> bool:
        return bool(self.azure_endpoint or self.azure_deployment)

    def resolve_model(self, purpose: str) -> Optional[str]:
        candidates = self.vision_models if purpose == "vision" else self.models
        return next((model for model in candidates if model), None)

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.azure_deployment,
            "azure_api_version": self.azure_api_version,
        }


def load_llm_settings(overrides: Optional[Dict[str, Optional[str]]] = None) -> LLMSettings:
    """Return merged LLM settings from .env, environment, and runtime overrides."""

    _load_dotenv_once()

    overrides = overrides or {}

    env_get = os.environ.get

    enabled = _as_bool(overrides.get("llm_enabled") or env_get("DOCTEST_LLM_ENABLED"), False)
    visual_enabled = _as_bool(
        overrides.get("llm_visual_enabled") or env_get("DOCTEST_LLM_VISUAL"), enabled
    )
    pdf_enabled = _as_bool(
        overrides.get("llm_pdf_enabled") or env_get("DOCTEST_LLM_PDF"), enabled
    )

    provider = (overrides.get("llm_provider") or env_get("DOCTEST_LLM_PROVIDER") or "openai").strip()

    models = _split_models(
        overrides.get("llm_models") or env_get("DOCTEST_LLM_MODEL"), DEFAULT_MODELS
    )
    vision_models = _split_models(
        overrides.get("llm_vision_models") or env_get("DOCTEST_LLM_VISION_MODEL"),
        DEFAULT_VISION_MODELS,
    )

    api_key = overrides.get("llm_api_key") or env_get("DOCTEST_LLM_API_KEY") or env_get("OPENAI_API_KEY")
    base_url = overrides.get("llm_base_url") or env_get("DOCTEST_LLM_BASE_URL") or env_get("OPENAI_BASE_URL")

    temperature = _as_float(
        overrides.get("llm_temperature") or env_get("DOCTEST_LLM_TEMPERATURE"), 0.2
    )
    max_output_tokens = _as_int(
        overrides.get("llm_max_output_tokens") or env_get("DOCTEST_LLM_MAX_OUTPUT_TOKENS"), None
    )
    request_timeout = _as_float(
        overrides.get("llm_request_timeout") or env_get("DOCTEST_LLM_REQUEST_TIMEOUT"), 30.0
    )

    azure_endpoint = overrides.get("azure_openai_endpoint") or env_get("AZURE_OPENAI_ENDPOINT")
    azure_deployment = overrides.get("azure_openai_deployment") or env_get("AZURE_OPENAI_DEPLOYMENT")
    azure_api_version = overrides.get("azure_openai_api_version") or env_get("AZURE_OPENAI_API_VERSION")
    azure_api_key = overrides.get("azure_openai_api_key") or env_get("AZURE_OPENAI_API_KEY") or env_get("OPENAI_API_KEY")

    extra = {
        key: overrides[key]
        for key in overrides
        if key not in {
            "llm_enabled",
            "llm_visual_enabled",
            "llm_pdf_enabled",
            "llm_provider",
            "llm_models",
            "llm_vision_models",
            "llm_api_key",
            "llm_base_url",
            "llm_temperature",
            "llm_max_output_tokens",
            "llm_request_timeout",
            "azure_openai_endpoint",
            "azure_openai_deployment",
            "azure_openai_api_version",
            "azure_openai_api_key",
        }
    }

    settings = LLMSettings(
        enabled=enabled,
        visual_enabled=visual_enabled,
        pdf_enabled=pdf_enabled,
        provider=provider,
        models=models,
        vision_models=vision_models,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature if temperature is not None else 0.2,
        max_output_tokens=max_output_tokens,
        request_timeout=request_timeout,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        azure_api_version=azure_api_version,
        azure_api_key=azure_api_key,
        extra=extra,
    )

    return settings


def ensure_dotenv_loaded() -> None:
    """Expose .env loading so other modules can trigger it eagerly."""
    _load_dotenv_once()
