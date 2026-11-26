"""Tolerant OCR assertions shared by the OCR tests."""

def contains_word_like(text, word, min_ratio=0.8):
    """True if some token of *text* reads as *word* up to OCR noise.

    Small, decorative text (the 8 px "SOUVENIR" sign in Beach_date.png) is
    read with one-character differences by different Tesseract models
    (e.g. "souveonR" with Ubuntu 26.04's eng model). 0.8 accepts a single
    substitution, insertion or deletion and rejects partial or unrelated
    words ("SOUV" 0.67, "SOURCE" 0.57).
    """
    from difflib import SequenceMatcher

    word = word.upper()
    return any(SequenceMatcher(None, token.upper(), word).ratio() >= min_ratio for token in text.split())
