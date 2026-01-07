from __future__ import annotations

from typing import Dict, Optional


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


def apply_character_replacements(
    text: str,
    replacements: Optional[Dict[str, str]] = None,
) -> str:
    """Apply custom character replacements to text.

    This function replaces characters in the input text based on the provided
    mapping dictionary. Common use cases include normalizing non-breaking spaces
    (U+00A0) to regular spaces, or converting typographic dashes to ASCII hyphens.

    Args:
        text: Input text to transform.
        replacements: Dict mapping source characters/strings to their replacements.
            If None or empty, returns the original text unchanged.

    Returns:
        Transformed text with all replacements applied.

    Examples:
        >>> apply_character_replacements("hello\\u00A0world", {"\u00A0": " "})
        'hello world'
        >>> apply_character_replacements("test\\u2013value", {"\u2013": "-"})
        'test-value'
    """
    if not text or not replacements:
        return text or ""

    for source, target in replacements.items():
        text = text.replace(source, target)
    return text