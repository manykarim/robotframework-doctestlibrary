"""Repetition-based header/footer detection for PDF structure comparison.

Scans configurable vertical regions at the top/bottom of each page, identifies
text lines that repeat across multiple pages (with digit normalization for page
numbers), and removes them from the DocumentStructure before comparison.

This module is a pure-function domain service with no side effects, no Robot
Framework dependency, and no I/O.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Set

from DocTest.PdfStructureModels import (
    DocumentStructure,
    PageStructure,
    TextBlock,
    TextLine,
)

__all__ = [
    "HeaderFooterConfig",
    "DetectionResult",
    "detect_repeating_headers_footers",
    "strip_detected_headers_footers",
    "filter_headers_footers",
]

_DIGIT_RUN_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class HeaderFooterConfig:
    """Configuration for repetition-based header/footer detection."""

    header_scan_height: float = 0.0
    footer_scan_height: float = 0.0
    repeat_threshold: int = 2

    @property
    def enabled(self) -> bool:
        """Return True if at least one scan region is configured."""
        return self.header_scan_height > 0 or self.footer_scan_height > 0


@dataclass(frozen=True)
class DetectionResult:
    """Immutable record of which normalized keys were detected as headers/footers."""

    header_keys: FrozenSet[str]
    footer_keys: FrozenSet[str]

    @property
    def has_detections(self) -> bool:
        return bool(self.header_keys or self.footer_keys)


def _normalize_for_grouping(text: str) -> str:
    """Replace all digit runs with '#' so page-number variants group together.

    Examples:
        "Page 1 of 5"  -> "Page # of #"
        "ACME Corp"    -> "ACME Corp"   (no digits, unchanged)
        "- 3 -"        -> "- # -"
    """
    return _DIGIT_RUN_RE.sub("#", text)


def detect_repeating_headers_footers(
    structure: DocumentStructure,
    config: HeaderFooterConfig,
) -> DetectionResult:
    """Scan a DocumentStructure and identify text that repeats across pages
    in the header/footer regions.

    Args:
        structure: The document to scan.
        config: Detection parameters (scan heights and threshold).

    Returns:
        A DetectionResult containing the normalized keys of detected
        header and footer lines.
    """
    if not config.enabled:
        return DetectionResult(header_keys=frozenset(), footer_keys=frozenset())

    header_candidates: Dict[str, Set[int]] = defaultdict(set)
    footer_candidates: Dict[str, Set[int]] = defaultdict(set)

    for page in structure.pages:
        footer_boundary = page.height - config.footer_scan_height

        for block in page.blocks:
            for line in block.lines:
                text = line.text or ""
                if not text:
                    continue
                key = _normalize_for_grouping(text)

                # Check header region
                if config.header_scan_height > 0 and line.bbox[1] < config.header_scan_height:
                    header_candidates[key].add(page.page_number)

                # Check footer region
                if config.footer_scan_height > 0 and line.bbox[3] > footer_boundary:
                    footer_candidates[key].add(page.page_number)

    detected_header_keys = frozenset(
        key for key, pages in header_candidates.items() if len(pages) >= config.repeat_threshold
    )
    detected_footer_keys = frozenset(
        key for key, pages in footer_candidates.items() if len(pages) >= config.repeat_threshold
    )

    return DetectionResult(
        header_keys=detected_header_keys,
        footer_keys=detected_footer_keys,
    )


def strip_detected_headers_footers(
    structure: DocumentStructure,
    detection: DetectionResult,
    config: HeaderFooterConfig,
) -> DocumentStructure:
    """Remove detected header/footer lines from a DocumentStructure.

    Only lines that (a) match a detected normalized key AND (b) fall within
    the configured scan region are removed. Body lines with identical text
    are preserved.

    Args:
        structure: The document to filter.
        detection: The detection result from detect_repeating_headers_footers().
        config: The same config used for detection (needed for region bounds).

    Returns:
        A new DocumentStructure with header/footer lines removed.
    """
    if not detection.has_detections:
        return structure

    filtered_pages: List[PageStructure] = []

    for page in structure.pages:
        footer_boundary = page.height - config.footer_scan_height
        new_blocks: List[TextBlock] = []
        next_line_index = 0

        for block in page.blocks:
            new_lines: List[TextLine] = []

            for line in block.lines:
                text = line.text or ""
                key = _normalize_for_grouping(text)

                # Remove if line is in header region AND matches a detected header key
                if (
                    config.header_scan_height > 0
                    and line.bbox[1] < config.header_scan_height
                    and key in detection.header_keys
                ):
                    continue

                # Remove if line is in footer region AND matches a detected footer key
                if (
                    config.footer_scan_height > 0
                    and line.bbox[3] > footer_boundary
                    and key in detection.footer_keys
                ):
                    continue

                new_lines.append(
                    TextLine(
                        index=next_line_index,
                        text=text,
                        bbox=line.bbox,
                        fonts=set(line.fonts),
                        spans=list(line.spans),
                    )
                )
                next_line_index += 1

            if new_lines:
                new_blocks.append(
                    TextBlock(
                        index=block.index,
                        bbox=block.bbox,
                        lines=new_lines,
                    )
                )

        filtered_pages.append(
            PageStructure(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                blocks=new_blocks,
            )
        )

    return DocumentStructure(pages=filtered_pages, config=structure.config)


def filter_headers_footers(
    structure: DocumentStructure,
    config: HeaderFooterConfig,
) -> DocumentStructure:
    """Convenience function: detect and strip in one call.

    Equivalent to:
        detection = detect_repeating_headers_footers(structure, config)
        return strip_detected_headers_footers(structure, detection, config)

    Args:
        structure: The document to process.
        config: Detection parameters.

    Returns:
        A new DocumentStructure with detected repeating headers/footers removed.
        If config.enabled is False, returns the input unchanged.
    """
    if not config.enabled:
        return structure
    detection = detect_repeating_headers_footers(structure, config)
    return strip_detected_headers_footers(structure, detection, config)
