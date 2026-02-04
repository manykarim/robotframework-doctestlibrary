from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from DocTest.TextNormalization import apply_character_replacements, normalize_ligatures


__all__ = [
    "TextSpan",
    "TextLine",
    "TextBlock",
    "PageStructure",
    "DocumentStructure",
    "StructureExtractionConfig",
    "WordToken",
    "strip_font_subset",
    "collapse_whitespace",
    "round_bbox",
    "build_page_structure",
    "build_page_structure_from_words",
    "flatten_document_text",
    "flatten_document_words",
]


@dataclass
class TextSpan:
    """Represents a single PyMuPDF span inside a line."""

    text: str
    font: Optional[str]
    size: float


@dataclass
class TextLine:
    """Normalized text line with aggregated spans and layout data."""

    index: int
    text: str
    bbox: Tuple[float, float, float, float]
    fonts: Set[str] = field(default_factory=set)
    spans: List[TextSpan] = field(default_factory=list)


@dataclass
class TextBlock:
    """Text block containing ordered lines."""

    index: int
    bbox: Tuple[float, float, float, float]
    lines: List[TextLine] = field(default_factory=list)

    @property
    def line_count(self) -> int:
        return len(self.lines)


@dataclass
class PageStructure:
    """Structured representation of a PDF page."""

    page_number: int
    width: float
    height: float
    blocks: List[TextBlock] = field(default_factory=list)

    @property
    def line_count(self) -> int:
        return sum(block.line_count for block in self.blocks)


@dataclass
class StructureExtractionConfig:
    """Controls how text spanning and whitespace/font normalization is performed."""

    collapse_whitespace: bool = True
    strip_font_subset: bool = True
    whitespace_replacement: str = " "
    strip_line_edges: bool = True
    drop_empty_lines: bool = True
    round_precision: Optional[int] = 3
    normalize_ligatures: bool = False
    character_replacements: Optional[Dict[str, str]] = None
    spatial_word_sorting: bool = False

    def __hash__(self) -> int:  # Allow usage as dictionary key for caching.
        # Convert character_replacements dict to a hashable tuple of sorted items
        replacements_hash = (
            tuple(sorted(self.character_replacements.items()))
            if self.character_replacements
            else ()
        )
        return hash(
            (
                self.collapse_whitespace,
                self.strip_font_subset,
                self.whitespace_replacement,
                self.strip_line_edges,
                self.drop_empty_lines,
                self.round_precision,
                self.normalize_ligatures,
                replacements_hash,
                self.spatial_word_sorting,
            )
        )


@dataclass
class DocumentStructure:
    """Structured representation of an entire PDF document."""

    pages: List[PageStructure]
    config: StructureExtractionConfig

    @property
    def page_count(self) -> int:
        return len(self.pages)


@dataclass(frozen=True)
class WordToken:
    """A single word token extracted from a document, with provenance metadata."""
    text: str
    source_page: int
    source_line_index: int
    word_index: int


def flatten_document_text(structure: DocumentStructure) -> List[str]:
    """Extract all text lines from a document in reading order, ignoring page boundaries.

    This function traverses all pages, blocks, and lines in the document structure
    and returns a flat list of text strings in the order they appear. Useful for
    comparing document content when text may reflow across pages due to font or
    layout changes.

    Args:
        structure: A DocumentStructure containing pages with text blocks and lines.

    Returns:
        A list of text strings from all lines in document reading order.
    """
    texts: List[str] = []
    for page in structure.pages:
        for block in page.blocks:
            for line in block.lines:
                if line.text:
                    texts.append(line.text)
    return texts


def flatten_document_words(
    structure: DocumentStructure,
    *,
    normalize_word_boundaries: bool = False,
    normalize_ligatures_in_words: bool = False,
) -> Tuple[List[str], List[WordToken]]:
    """Extract all words from a document in reading order, ignoring page/line boundaries.

    Splits every text line on whitespace to produce individual word tokens.
    This enables comparison at word granularity, making the comparison resilient
    to text reflow caused by font or layout changes.

    Args:
        structure: A DocumentStructure containing pages with text blocks and lines.
        normalize_word_boundaries: When True, merge tokens that were split
            across line boundaries by connector characters (``/``, ``-``, ``\\``).
        normalize_ligatures_in_words: When True, replace known typographic
            ligatures with their ASCII equivalents in each word.

    Returns:
        A tuple of:
        - words: Flat list of word strings for use with SequenceMatcher.
        - tokens: Corresponding list of WordToken objects with provenance.
    """
    words: List[str] = []
    tokens: List[WordToken] = []
    global_line_index = 0
    word_index = 0

    for page in structure.pages:
        for block in page.blocks:
            for line in block.lines:
                if not line.text:
                    global_line_index += 1
                    continue
                line_words = line.text.split()
                for w in line_words:
                    words.append(w)
                    tokens.append(
                        WordToken(
                            text=w,
                            source_page=page.page_number,
                            source_line_index=global_line_index,
                            word_index=word_index,
                        )
                    )
                    word_index += 1
                global_line_index += 1

    # Apply ligature normalization to individual words if requested
    if normalize_ligatures_in_words:
        from DocTest.TextNormalization import normalize_ligatures
        words = [normalize_ligatures(w) for w in words]
        tokens = [
            WordToken(
                text=normalize_ligatures(t.text),
                source_page=t.source_page,
                source_line_index=t.source_line_index,
                word_index=t.word_index,
            )
            for t in tokens
        ]

    # Merge words split across line boundaries
    if normalize_word_boundaries:
        from DocTest.TextNormalization import merge_split_words
        words, tokens = merge_split_words(words, tokens)

    return words, tokens


def strip_font_subset(font_name: Optional[str]) -> Optional[str]:
    """Drop random subset prefixes inserted by PDF generators."""

    if not font_name:
        return font_name
    if "+" in font_name:
        prefix, suffix = font_name.split("+", 1)
        # Subset prefixes are typically uppercase ASCII and between 4-8 chars.
        if prefix.isupper() and 3 < len(prefix) < 9:
            return suffix or font_name
    return font_name


def collapse_whitespace(text: str, replacement: str = " ") -> str:
    """Collapse all runs of whitespace into a single replacement char."""

    if not text:
        return ""
    return replacement.join(text.split())


def round_bbox(bbox: Sequence[float], precision: Optional[int]) -> Tuple[float, float, float, float]:
    """Round bbox coordinates to the requested precision (if any)."""

    if precision is None:
        return tuple(float(value) for value in bbox)  # type: ignore[return-value]
    return tuple(round(float(value), precision) for value in bbox)  # type: ignore[return-value]


def _sanitize_span_text(text: str, config: StructureExtractionConfig) -> str:
    # Apply character replacements first (before other normalization)
    if config.character_replacements:
        text = apply_character_replacements(text, config.character_replacements)
    if config.collapse_whitespace:
        text = collapse_whitespace(text, config.whitespace_replacement)
    if config.strip_line_edges:
        text = text.strip()
    if config.normalize_ligatures:
        text = normalize_ligatures(text)
    return text


def _normalise_font(font: Optional[str], config: StructureExtractionConfig) -> Optional[str]:
    if not font:
        return None
    return strip_font_subset(font) if config.strip_font_subset else font


def build_page_structure(
    page_number: int,
    pdf_dict: Optional[Dict],
    config: Optional[StructureExtractionConfig] = None,
    dpi: Optional[int] = None,
    image_shape: Optional[Tuple[int, int, int]] = None,
) -> PageStructure:
    """Convert PyMuPDF `get_text(\"dict\")` output into a structured representation."""

    config = config or StructureExtractionConfig()
    width = 0.0
    height = 0.0
    blocks_payload: Iterable[Dict] = []

    if pdf_dict:
        width = float(pdf_dict.get("width") or 0.0)
        height = float(pdf_dict.get("height") or 0.0)
        blocks_payload = pdf_dict.get("blocks") or []

    if (width == 0.0 or height == 0.0) and image_shape and dpi:
        px_height, px_width = image_shape[:2]
        width = px_width * 72.0 / dpi
        height = px_height * 72.0 / dpi

    blocks: List[TextBlock] = []
    global_line_index = 0
    block_index = 0

    for block in blocks_payload:
        if block.get("type") != 0:  # Skip non-text blocks.
            continue

        block_lines: List[TextLine] = []
        for line in block.get("lines", []):
            spans_payload = line.get("spans") or []
            line_spans: List[TextSpan] = []
            line_fonts: Set[str] = set()
            text_parts: List[str] = []

            for span in spans_payload:
                raw_text = span.get("text") or ""
                normalised_text = _sanitize_span_text(raw_text, config)
                font_name = _normalise_font(span.get("font"), config)
                if font_name:
                    line_fonts.add(font_name)
                if normalised_text:
                    text_parts.append(normalised_text)
                line_spans.append(
                    TextSpan(
                        text=normalised_text,
                        font=font_name,
                        size=float(span.get("size") or 0.0),
                    )
                )

            line_text = (
                config.whitespace_replacement.join(text_parts)
                if text_parts
                else ""
            )
            if config.strip_line_edges:
                line_text = line_text.strip()

            if config.drop_empty_lines and not line_text:
                continue

            bbox = round_bbox(line.get("bbox", (0.0, 0.0, 0.0, 0.0)), config.round_precision)
            block_lines.append(
                TextLine(
                    index=global_line_index,
                    text=line_text,
                    bbox=bbox,
                    fonts=line_fonts,
                    spans=line_spans,
                )
            )
            global_line_index += 1

        if not block_lines:
            continue

        block_bbox = round_bbox(block.get("bbox", (0.0, 0.0, 0.0, 0.0)), config.round_precision)
        blocks.append(
            TextBlock(
                index=block_index,
                bbox=block_bbox,
                lines=block_lines,
            )
        )
        block_index += 1

    return PageStructure(
        page_number=page_number,
        width=width,
        height=height,
        blocks=blocks,
    )


def build_page_structure_from_words(
    page_number: int,
    pdf_text_words: Optional[List],
    config: Optional[StructureExtractionConfig] = None,
    *,
    page_width: float = 0.0,
    page_height: float = 0.0,
    dpi: Optional[int] = None,
    image_shape: Optional[Tuple[int, int, int]] = None,
) -> PageStructure:
    """Build a ``PageStructure`` from PyMuPDF ``get_text("words")`` output.

    This bypasses block-level extraction entirely, grouping individual word
    bounding boxes into lines using adaptive Y-proximity.  The result is
    immune to block fragmentation caused by different PDF generators.

    Each word tuple from PyMuPDF has the form::

        (x0, y0, x1, y1, "word", block_no, line_no, word_no)

    Words are grouped into lines when their vertical midpoints are within
    half the minimum word height of each other.  Within each line, words
    are ordered left-to-right by ``x0``.  Lines are ordered top-to-bottom.
    Each line becomes its own ``TextBlock`` (single-line blocks).

    Args:
        page_number: Zero-based page index.
        pdf_text_words: List of word tuples from ``page.get_text("words")``.
        config: Normalization settings (whitespace, ligatures, etc.).
        page_width: Page width in points.
        page_height: Page height in points.
        dpi: Optional DPI for computing page dimensions from ``image_shape``.
        image_shape: ``(height, width, channels)`` array shape, used with *dpi*
            to derive page dimensions when ``page_width``/``page_height`` are zero.

    Returns:
        A ``PageStructure`` with one block per reconstructed text line.
    """
    config = config or StructureExtractionConfig()

    width = page_width
    height = page_height
    if (width == 0.0 or height == 0.0) and image_shape and dpi:
        px_height, px_width = image_shape[:2]
        width = px_width * 72.0 / dpi
        height = px_height * 72.0 / dpi

    if not pdf_text_words:
        return PageStructure(
            page_number=page_number,
            width=width,
            height=height,
            blocks=[],
        )

    # --- Group words into visual lines by Y-proximity ---
    # Sort by vertical midpoint first, then horizontal position.
    sorted_words = sorted(pdf_text_words, key=lambda w: ((w[1] + w[3]) / 2.0, w[0]))

    lines: List[List] = []  # Each element: list of word tuples
    line_y_mid: List[float] = []  # Representative Y midpoint per line
    line_min_height: List[float] = []  # Cached minimum word height per line

    for word in sorted_words:
        w_y0, w_y1 = float(word[1]), float(word[3])
        w_mid = (w_y0 + w_y1) / 2.0
        w_height = max(w_y1 - w_y0, 1.0)

        # Search backward from most recent line (words are Y-sorted, so the
        # most recent line is the most likely match).  Break early once we
        # move past the tolerance range.
        merged = False
        max_possible_tolerance = w_height * 0.5
        for idx in range(len(lines) - 1, -1, -1):
            ly_mid = line_y_mid[idx]
            delta = abs(w_mid - ly_mid)
            if delta > max_possible_tolerance and w_mid > ly_mid:
                break  # Past tolerance; earlier lines are even further away.
            tolerance = min(line_min_height[idx], w_height) * 0.5
            if delta <= tolerance:
                lines[idx].append(word)
                n = len(lines[idx])
                line_y_mid[idx] = ly_mid + (w_mid - ly_mid) / n
                if w_height < line_min_height[idx]:
                    line_min_height[idx] = w_height
                merged = True
                break
        if not merged:
            lines.append([word])
            line_y_mid.append(w_mid)
            line_min_height.append(w_height)

    # Sort lines top-to-bottom by midpoint, words left-to-right within each.
    indexed_lines = sorted(enumerate(lines), key=lambda pair: line_y_mid[pair[0]])

    blocks: List[TextBlock] = []
    global_line_index = 0
    block_index = 0

    for _orig_idx, line_words in indexed_lines:
        line_words_sorted = sorted(line_words, key=lambda w: float(w[0]))

        # Build text from words, applying normalization.
        text_parts: List[str] = []
        for w in line_words_sorted:
            raw = str(w[4])
            normalized = _sanitize_span_text(raw, config)
            if normalized:
                text_parts.append(normalized)

        line_text = config.whitespace_replacement.join(text_parts) if text_parts else ""
        if config.strip_line_edges:
            line_text = line_text.strip()
        if config.drop_empty_lines and not line_text:
            continue

        # Compute line bbox as union of all word bboxes.
        x0 = min(float(w[0]) for w in line_words_sorted)
        y0 = min(float(w[1]) for w in line_words_sorted)
        x1 = max(float(w[2]) for w in line_words_sorted)
        y1 = max(float(w[3]) for w in line_words_sorted)
        bbox = round_bbox((x0, y0, x1, y1), config.round_precision)

        text_line = TextLine(
            index=global_line_index,
            text=line_text,
            bbox=bbox,
            fonts=set(),
            spans=[TextSpan(text=line_text, font=None, size=0.0)],
        )
        blocks.append(
            TextBlock(
                index=block_index,
                bbox=bbox,
                lines=[text_line],
            )
        )
        global_line_index += 1
        block_index += 1

    return PageStructure(
        page_number=page_number,
        width=width,
        height=height,
        blocks=blocks,
    )
