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
    "strip_font_subset",
    "collapse_whitespace",
    "round_bbox",
    "build_page_structure",
    "flatten_document_text",
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
