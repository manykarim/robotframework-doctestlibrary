from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from DocTest.PdfStructureModels import DocumentStructure, PageStructure, TextLine


__all__ = [
    "StructureTolerance",
    "LineDifference",
    "StructureComparisonResult",
    "compare_document_structures",
]


@dataclass
class StructureTolerance:
    """Absolute and relative tolerances (in PDF points) for layout deltas."""

    position: float = 15.0
    size: float = 15.0
    relative: float = 0.05  # 5% relative tolerance fallback.


@dataclass
class LineDifference:
    """Details about a single line mismatch between two PDF pages."""

    page: int
    diff_type: str
    message: str
    ref_text: Optional[str] = None
    cand_text: Optional[str] = None
    ref_bbox: Optional[Tuple[float, float, float, float]] = None
    cand_bbox: Optional[Tuple[float, float, float, float]] = None
    deltas: Optional[Dict[str, float]] = None
    reference_index: Optional[int] = None
    candidate_index: Optional[int] = None


@dataclass
class StructureComparisonResult:
    """Aggregate differences found during structure comparison."""

    passed: bool = True
    page_differences: Dict[int, List[LineDifference]] = field(default_factory=dict)
    summary: List[str] = field(default_factory=list)

    def add_difference(self, diff: LineDifference):
        self.passed = False
        self.page_differences.setdefault(diff.page, []).append(diff)

    def extend_summary(self, message: str):
        self.summary.append(message)


def compare_document_structures(
    reference: DocumentStructure,
    candidate: DocumentStructure,
    tolerance: StructureTolerance,
    *,
    case_sensitive: bool = True,
) -> StructureComparisonResult:
    """Compare two structured PDF representations using geometric tolerances."""

    result = StructureComparisonResult()

    if reference.page_count != candidate.page_count:
        result.extend_summary(
            f"Page count mismatch: reference={reference.page_count}, candidate={candidate.page_count}"
        )
        result.passed = False

    common_pages = min(reference.page_count, candidate.page_count)

    for index in range(common_pages):
        _compare_page(
            ref_page=reference.pages[index],
            cand_page=candidate.pages[index],
            tolerance=tolerance,
            case_sensitive=case_sensitive,
            result=result,
        )

    for page in reference.pages[common_pages:]:
        result.add_difference(
            LineDifference(
                page=page.page_number,
                diff_type="missing_page",
                message=f"Reference page {page.page_number} missing in candidate document.",
            )
        )

    for page in candidate.pages[common_pages:]:
        result.add_difference(
            LineDifference(
                page=page.page_number,
                diff_type="extra_page",
                message=f"Candidate page {page.page_number} has no counterpart in reference document.",
            )
        )

    return result


def _compare_page(
    ref_page: PageStructure,
    cand_page: PageStructure,
    tolerance: StructureTolerance,
    *,
    case_sensitive: bool,
    result: StructureComparisonResult,
):
    if len(ref_page.blocks) != len(cand_page.blocks):
        result.add_difference(
            LineDifference(
                page=ref_page.page_number,
                diff_type="block_count_mismatch",
                message=f"Block count mismatch on page {ref_page.page_number}: reference={len(ref_page.blocks)}, candidate={len(cand_page.blocks)}",
            )
        )

    ref_lines = _flatten_lines(ref_page)
    cand_lines = _flatten_lines(cand_page)

    ref_texts = [_normalise_text(line.text, case_sensitive) for line in ref_lines]
    cand_texts = [_normalise_text(line.text, case_sensitive) for line in cand_lines]

    matcher = difflib.SequenceMatcher(a=ref_texts, b=cand_texts, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                _check_line_geometry(
                    ref_page=ref_page,
                    cand_page=cand_page,
                    ref_line=ref_lines[i1 + offset],
                    cand_line=cand_lines[j1 + offset],
                    tolerance=tolerance,
                    result=result,
                )
        elif tag == "replace":
            _record_line_replacements(
                ref_page=ref_page,
                cand_page=cand_page,
                ref_segment=ref_lines[i1:i2],
                cand_segment=cand_lines[j1:j2],
                result=result,
            )
        elif tag == "delete":
            for line in ref_lines[i1:i2]:
                result.add_difference(
                    LineDifference(
                        page=ref_page.page_number,
                        diff_type="missing_line",
                        message=f"Line missing in candidate: '{line.text}'",
                        ref_text=line.text,
                        ref_bbox=line.bbox,
                        reference_index=line.index,
                    )
                )
        elif tag == "insert":
            for line in cand_lines[j1:j2]:
                result.add_difference(
                    LineDifference(
                        page=ref_page.page_number,
                        diff_type="extra_line",
                        message=f"Extra line in candidate: '{line.text}'",
                        cand_text=line.text,
                        cand_bbox=line.bbox,
                        candidate_index=line.index,
                    )
                )


def _record_line_replacements(
    ref_page: PageStructure,
    cand_page: PageStructure,
    ref_segment: Sequence[TextLine],
    cand_segment: Sequence[TextLine],
    result: StructureComparisonResult,
):
    max_len = max(len(ref_segment), len(cand_segment))
    for offset in range(max_len):
        ref_line = ref_segment[offset] if offset < len(ref_segment) else None
        cand_line = cand_segment[offset] if offset < len(cand_segment) else None

        message = "Text mismatch"
        if ref_line and cand_line:
            message = f"Text mismatch: reference='{ref_line.text}', candidate='{cand_line.text}'"
        elif ref_line and not cand_line:
            message = f"Line missing in candidate: '{ref_line.text}'"
        elif cand_line and not ref_line:
            message = f"Unexpected line in candidate: '{cand_line.text}'"

        result.add_difference(
            LineDifference(
                page=ref_page.page_number,
                diff_type="text_mismatch",
                message=message,
                ref_text=ref_line.text if ref_line else None,
                cand_text=cand_line.text if cand_line else None,
                ref_bbox=ref_line.bbox if ref_line else None,
                cand_bbox=cand_line.bbox if cand_line else None,
                reference_index=ref_line.index if ref_line else None,
                candidate_index=cand_line.index if cand_line else None,
            )
        )


def _check_line_geometry(
    ref_page: PageStructure,
    cand_page: PageStructure,
    ref_line: TextLine,
    cand_line: TextLine,
    tolerance: StructureTolerance,
    result: StructureComparisonResult,
):
    deltas = _compute_bbox_deltas(ref_line.bbox, cand_line.bbox)

    failing_axes: Dict[str, float] = {}

    if not _is_within_tolerance(
        deltas["left"],
        reference_values=(ref_line.bbox[0], cand_line.bbox[0]),
        tolerance_value=tolerance.position,
        relative_tolerance=tolerance.relative,
    ):
        failing_axes["left"] = deltas["left"]

    if not _is_within_tolerance(
        deltas["top"],
        reference_values=(ref_line.bbox[1], cand_line.bbox[1]),
        tolerance_value=tolerance.position,
        relative_tolerance=tolerance.relative,
    ):
        failing_axes["top"] = deltas["top"]

    if not _is_within_tolerance(
        deltas["width"],
        reference_values=(_width(ref_line.bbox), _width(cand_line.bbox)),
        tolerance_value=tolerance.size,
        relative_tolerance=tolerance.relative,
    ):
        failing_axes["width"] = deltas["width"]

    if not _is_within_tolerance(
        deltas["height"],
        reference_values=(_height(ref_line.bbox), _height(cand_line.bbox)),
        tolerance_value=tolerance.size,
        relative_tolerance=tolerance.relative,
    ):
        failing_axes["height"] = deltas["height"]

    if failing_axes:
        detail = ", ".join(f"{axis}={value:.3f}" for axis, value in failing_axes.items())
        result.add_difference(
            LineDifference(
                page=ref_page.page_number,
                diff_type="geometry_mismatch",
                message=f"Line layout delta outside tolerance for '{ref_line.text}': {detail}",
                ref_text=ref_line.text,
                cand_text=cand_line.text,
                ref_bbox=ref_line.bbox,
                cand_bbox=cand_line.bbox,
                deltas=deltas,
                reference_index=ref_line.index,
                candidate_index=cand_line.index,
            )
        )


def _flatten_lines(page: PageStructure) -> List[TextLine]:
    lines: List[TextLine] = []
    for block in page.blocks:
        lines.extend(block.lines)
    return lines


def _normalise_text(text: str, case_sensitive: bool) -> str:
    return text if case_sensitive else text.lower()


def _compute_bbox_deltas(
    ref_bbox: Tuple[float, float, float, float], cand_bbox: Tuple[float, float, float, float]
) -> Dict[str, float]:
    ref_left, ref_top, ref_right, ref_bottom = ref_bbox
    cand_left, cand_top, cand_right, cand_bottom = cand_bbox

    return {
        "left": cand_left - ref_left,
        "top": cand_top - ref_top,
        "width": (cand_right - cand_left) - (ref_right - ref_left),
        "height": (cand_bottom - cand_top) - (ref_bottom - ref_top),
    }


def _is_within_tolerance(
    delta: float,
    *,
    reference_values: Iterable[float],
    tolerance_value: float,
    relative_tolerance: float,
) -> bool:
    if abs(delta) <= tolerance_value:
        return True

    max_reference = max((abs(value) for value in reference_values), default=0.0)
    if max_reference == 0.0:
        return False

    return abs(delta) <= max_reference * relative_tolerance


def _width(bbox: Tuple[float, float, float, float]) -> float:
    return bbox[2] - bbox[0]


def _height(bbox: Tuple[float, float, float, float]) -> float:
    return bbox[3] - bbox[1]
