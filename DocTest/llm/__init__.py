"""
Utility helpers for optional LLM-powered comparisons.

The module is deliberately lightweight so importing the main package keeps
working even when the optional ``robotframework-doctestlibrary[ai]`` extra
isn't installed.  Downstream modules perform dependency checks on demand.
"""

from __future__ import annotations

from .config import LLMSettings, ensure_dotenv_loaded, load_llm_settings
from .exceptions import LLMDependencyError

__all__ = [
    "LLMSettings",
    "LLMDependencyError",
    "ensure_dotenv_loaded",
    "load_llm_settings",
]
