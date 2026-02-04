from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple


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


_WORD_BOUNDARY_CONNECTORS: Set[str] = frozenset("/\\-")


def merge_split_words(
    words: List[str],
    tokens: "List[WordToken]",
    connectors: Set[str] | None = None,
) -> "Tuple[List[str], List[WordToken]]":
    """Merge word tokens that were split across PDF line boundaries.

    When text reflows across lines in a PDF, words containing connector
    characters (like ``/``, ``-``, ``\\``) can be split into separate tokens.
    For example, ``JS2_D48/F16/H8`` may become ``["JS2_D48/F16/", "H8"]``
    when the line break falls after the ``/``.

    This function detects such splits by looking for tokens from consecutive
    lines where the preceding token ends with a connector character, and
    merges them back into a single token.

    Args:
        words: Flat list of word strings.
        tokens: Corresponding WordToken provenance objects.
        connectors: Set of characters that indicate a word was split.
            Defaults to ``_WORD_BOUNDARY_CONNECTORS`` (``/``, ``\\``, ``-``).

    Returns:
        Tuple of (merged_words, merged_tokens) with reduced length.
    """
    if not words or len(words) <= 1:
        return list(words), list(tokens)

    if connectors is None:
        connectors = _WORD_BOUNDARY_CONNECTORS

    merged_words: List[str] = [words[0]]
    merged_tokens: List[tokens[0].__class__] = [tokens[0]]

    for i in range(1, len(words)):
        prev_token = merged_tokens[-1]
        curr_token = tokens[i]
        prev_word = merged_words[-1]

        # Only merge if tokens are from different lines AND previous word ends with connector.
        # Skip standalone connectors (e.g. a bare "-" used as punctuation, not a split word).
        if (prev_token.source_line_index != curr_token.source_line_index
                and prev_word
                and prev_word[-1] in connectors
                and len(prev_word) > 1):
            # Merge: concatenate words, keep first token's provenance
            merged_words[-1] = prev_word + words[i]
            # Update token with merged text
            from DocTest.PdfStructureModels import WordToken
            merged_tokens[-1] = WordToken(
                text=merged_words[-1],
                source_page=prev_token.source_page,
                source_line_index=prev_token.source_line_index,
                word_index=prev_token.word_index,
            )
        else:
            merged_words.append(words[i])
            merged_tokens.append(tokens[i])

    return merged_words, merged_tokens


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