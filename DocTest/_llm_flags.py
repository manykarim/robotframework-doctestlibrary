"""Shared helpers for LLM-related option parsing and decision handling.

Single source for helpers that were previously byte-identical copies in
VisualTest and PdfTest. The originals remain importable from those modules
for backwards compatibility.
"""

from typing import Any


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    return bool(value)


def _coerce_label_value(label: Any) -> str:
    value = getattr(label, "value", label)
    return str(value)


def _decision_equals_flag(label: Any, enum_cls: Any) -> bool:
    candidate = _coerce_label_value(label).lower()
    if enum_cls is None:
        return candidate == "flag"
    flag_member = getattr(enum_cls, "FLAG", None)
    if flag_member is None:
        return candidate == "flag"
    flag_value = _coerce_label_value(flag_member).lower()
    return candidate == flag_value
