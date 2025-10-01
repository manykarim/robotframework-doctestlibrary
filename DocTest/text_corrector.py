"""Utilities for post-processing OCR tokens produced by OCRS.

The Rust-based OCR backend occasionally misclassifies characters with similar
shapes (for example interpreting ``V`` as ``Y``).  To preserve backwards
compatibility with existing suites we perform lightweight corrections based on
common English vocabulary.  The corrections are intentionally conservative so
that serial numbers or identifiers are not rewritten unexpectedly.
"""

from __future__ import annotations

import difflib
import re
import string
from functools import lru_cache
from importlib import resources
from typing import Iterable, Optional


WORDLIST_RESOURCE = "data/google-10000-english.txt"


@lru_cache(maxsize=1)
def _load_common_words() -> set[str]:
    """Return a set of common English words for spell correction."""

    try:
        wordlist_path = resources.files("DocTest").joinpath(WORDLIST_RESOURCE)
    except (ModuleNotFoundError, FileNotFoundError):
        return set()

    try:
        with wordlist_path.open("r", encoding="utf-8") as handle:
            words = {line.strip().lower() for line in handle if line.strip()}
    except FileNotFoundError:
        return set()
    return words


AMBIGUOUS_DIGIT_MAP = str.maketrans({
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "l": "1",
    "o": "0",
})

AMBIGUOUS_LETTER_MAP = str.maketrans({
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "g",
})


def _apply_case(template: str, candidate: str) -> str:
    if template.isupper():
        return candidate.upper()
    if template.islower():
        return candidate.lower()
    if template.istitle():
        return candidate.title()
    return candidate


def _split_hyphenated(token: str) -> Iterable[str]:
    yield from token.split("-")


def _normalize_date_like(token: str) -> Optional[str]:
    parts = token.split("-")
    if len(parts) != 3:
        return None

    day_raw, month_raw, year_raw = parts

    day_candidate = day_raw.translate(AMBIGUOUS_DIGIT_MAP)
    day_digits = "".join(ch for ch in day_candidate if ch.isdigit())
    if len(day_digits) >= 2:
        day = day_digits[-2:]
    else:
        return None

    month_candidate = month_raw.translate(AMBIGUOUS_LETTER_MAP)
    month_letters = "".join(ch for ch in month_candidate if ch.isalpha())
    if len(month_letters) < 3:
        return None
    month = month_letters[:3].title()

    year_candidate = year_raw.translate(AMBIGUOUS_DIGIT_MAP)
    year_digits = "".join(ch for ch in year_candidate if ch.isdigit())
    if len(year_digits) >= 4:
        year = year_digits[:4]
    else:
        return None

    normalized = f"{day}-{month}-{year}"
    if re.fullmatch(r"\d{2}-[A-Za-z]{3}-\d{4}", normalized):
        return normalized
    return None


def correct_token(token: str) -> str:
    """Return a normalized representation of a token.

    The function performs two passes:

    1. Convert ambiguous characters that should be digits (eg. ``O`` -> ``0``)
       when the token contains at least one digit.
    2. Attempt a conservative spell-correction for purely alphabetic tokens
       using a curated list of common English words.
    """

    if not token:
        return ""

    token = token.strip()
    if not token:
        return ""

    date_like = _normalize_date_like(token)
    if date_like:
        return date_like

    has_digit = any(ch.isdigit() for ch in token)
    if has_digit:
        translated = token.translate(AMBIGUOUS_DIGIT_MAP)
        token = translated

    letters_only = all(ch.isalpha() or ch == "-" for ch in token)
    if not letters_only:
        return token

    words = _load_common_words()
    if not words:
        return token

    parts = list(_split_hyphenated(token))
    corrected_parts = []
    for part in parts:
        lower = part.lower()
        if lower in words:
            corrected_parts.append(_apply_case(part, lower))
            continue

        if len(part) <= 2 or any(ch not in string.ascii_letters for ch in part):
            corrected_parts.append(part)
            continue

        matches = difflib.get_close_matches(lower, words, n=1, cutoff=0.87)
        if matches:
            corrected_parts.append(_apply_case(part, matches[0]))
        else:
            corrected_parts.append(part)

    return "-".join(corrected_parts)
