from __future__ import annotations

from typing import Dict


_LIGATURE_MAP: Dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
}


def normalize_ligatures(text: str) -> str:
    """Replace known typographic ligatures with their ASCII equivalents."""
    if not text:
        return text or ""
    # Use a list comprehension for performance since ligatures are sparse.
    return "".join(_LIGATURE_MAP.get(char, char) for char in text)