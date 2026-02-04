"""Consolidated HTML report builder for PDF structure comparison results.

Transforms a StructureComparisonResult into a single HTML fragment suitable
for rendering inside Robot Framework's log.html via logger.info(msg, html=True).
"""

from __future__ import annotations

import html as html_module
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from DocTest.PdfStructureComparator import (
    DocumentTextDifference,
    DocumentWordDifference,
    LineDifference,
    StructureComparisonResult,
)

__all__ = [
    "build_structure_report",
    "build_structure_report_plain_text",
    "ReportMetadata",
]

DEFAULT_CONTEXT_LINES = 3
MAX_TEXT_DISPLAY_LENGTH = 120
MAX_HUNKS_BEFORE_COLLAPSE = 50


@dataclass
class ReportMetadata:
    """Metadata displayed in the report header."""
    reference_name: str = ""
    candidate_name: str = ""
    comparison_mode: str = ""
    page_count_ref: Optional[int] = None
    page_count_cand: Optional[int] = None
    exclusions_applied: List[str] = field(default_factory=list)


@dataclass
class ReportSummary:
    """Aggregate statistics for the comparison."""
    total_differences: int = 0
    missing_count: int = 0
    extra_count: int = 0
    mismatch_count: int = 0
    geometry_count: int = 0
    other_count: int = 0
    hunk_count: int = 0


def _escape(text: str) -> str:
    return html_module.escape(str(text), quote=True)


def _truncate(text: str, max_length: int = MAX_TEXT_DISPLAY_LENGTH) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _classify_diff_type(diff_type: str) -> str:
    """Map diff_type string to category."""
    if diff_type in ("missing_line", "missing_text", "missing_page", "missing_words"):
        return "missing"
    elif diff_type in ("extra_line", "extra_text", "extra_page", "extra_words"):
        return "extra"
    elif diff_type in ("text_mismatch", "word_mismatch"):
        return "mismatch"
    elif diff_type == "geometry_mismatch":
        return "geometry"
    else:
        return "other"


def _get_diff_display(diff: Any) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Extract category, message, ref_text, cand_text from any diff type."""
    category = _classify_diff_type(diff.diff_type)
    message = diff.message

    ref_text = None
    cand_text = None

    if isinstance(diff, LineDifference):
        ref_text = diff.ref_text
        cand_text = diff.cand_text
    elif isinstance(diff, DocumentTextDifference):
        ref_text = diff.ref_text
        cand_text = diff.cand_text
    elif isinstance(diff, DocumentWordDifference):
        ref_text = " ".join(diff.ref_words) if diff.ref_words else None
        cand_text = " ".join(diff.cand_words) if diff.cand_words else None

    return category, message, ref_text, cand_text


_CATEGORY_STYLES = {
    "missing": ("#fdd", "-"),
    "extra": ("#dfd", "+"),
    "mismatch": ("#ffd", "~"),
    "geometry": ("#eee", "\u0394"),  # delta symbol
    "other": ("#eee", "!"),
}


def _compute_summary(result: StructureComparisonResult) -> ReportSummary:
    """Compute aggregate statistics from a comparison result."""
    summary = ReportSummary()

    for diffs in result.page_differences.values():
        for d in diffs:
            cat = _classify_diff_type(d.diff_type)
            if cat == "missing": summary.missing_count += 1
            elif cat == "extra": summary.extra_count += 1
            elif cat == "mismatch": summary.mismatch_count += 1
            elif cat == "geometry": summary.geometry_count += 1
            else: summary.other_count += 1

    for d in result.document_differences:
        cat = _classify_diff_type(d.diff_type)
        if cat == "missing": summary.missing_count += 1
        elif cat == "extra": summary.extra_count += 1
        elif cat == "mismatch": summary.mismatch_count += 1
        else: summary.other_count += 1

    if hasattr(result, 'word_differences'):
        for d in result.word_differences:
            cat = _classify_diff_type(d.diff_type)
            if cat == "missing": summary.missing_count += 1
            elif cat == "extra": summary.extra_count += 1
            elif cat == "mismatch": summary.mismatch_count += 1
            else: summary.other_count += 1

    summary.total_differences = (
        summary.missing_count + summary.extra_count +
        summary.mismatch_count + summary.geometry_count + summary.other_count
    )
    return summary


def _render_diff_html(diff: Any) -> str:
    """Render a single difference as an HTML div with color coding."""
    category, message, ref_text, cand_text = _get_diff_display(diff)
    bg, symbol = _CATEGORY_STYLES.get(category, ("#eee", "?"))

    parts = []
    parts.append(f'<div style="background:{bg};padding:1px 4px;margin:1px 0;">')

    if category == "mismatch" and ref_text and cand_text:
        parts.append(f'<b>{_escape(symbol)}</b> ref: &quot;{_escape(_truncate(ref_text))}&quot;')
        parts.append(f'<br/>&nbsp;&nbsp;cand: &quot;{_escape(_truncate(cand_text))}&quot;')
    elif category == "missing" and ref_text:
        parts.append(f'<b>{_escape(symbol)}</b> &quot;{_escape(_truncate(ref_text))}&quot;')
    elif category == "extra" and cand_text:
        parts.append(f'<b>{_escape(symbol)}</b> &quot;{_escape(_truncate(cand_text))}&quot;')
    elif category == "geometry":
        deltas_str = ""
        if hasattr(diff, 'deltas') and diff.deltas:
            deltas_str = " (" + ", ".join(f"{k}={v:.3f}" for k, v in diff.deltas.items()) + ")"
        text_display = ref_text or cand_text or ""
        parts.append(f'<b>{_escape(symbol)}</b> &quot;{_escape(_truncate(text_display))}&quot;{_escape(deltas_str)}')
    else:
        parts.append(f'<b>{_escape(symbol)}</b> {_escape(_truncate(message))}')

    parts.append('</div>')
    return "".join(parts)


def _render_diff_plain(diff: Any) -> str:
    """Render a single difference as plain text."""
    category, message, ref_text, cand_text = _get_diff_display(diff)
    _, symbol = _CATEGORY_STYLES.get(category, ("", "?"))

    if category == "mismatch" and ref_text and cand_text:
        return f'  {symbol} ref: "{_truncate(ref_text)}"\n    cand: "{_truncate(cand_text)}"'
    elif category == "missing" and ref_text:
        return f'  {symbol} "{_truncate(ref_text)}"'
    elif category == "extra" and cand_text:
        return f'  {symbol} "{_truncate(cand_text)}"'
    else:
        return f'  {symbol} {_truncate(message)}'


def _get_diff_index(diff: Any) -> int:
    """Extract the primary positional index from a difference object."""
    if isinstance(diff, LineDifference):
        idx = diff.reference_index if diff.reference_index is not None else diff.candidate_index
        return idx if idx is not None else 999999
    elif isinstance(diff, DocumentTextDifference):
        idx = diff.ref_index if diff.ref_index is not None else diff.cand_index
        return idx if idx is not None else 999999
    elif isinstance(diff, DocumentWordDifference):
        idx = diff.ref_start_index if diff.ref_start_index is not None else diff.cand_start_index
        return idx if idx is not None else 999999
    return 999999


def _group_into_hunks(
    differences: Sequence[Any],
    context_lines: int,
    source_texts: Optional[List[str]] = None,
) -> List[dict]:
    """Group contiguous differences into hunks with context.

    Returns list of dicts: {start_index, end_index, differences, context_before, context_after}
    """
    if not differences:
        return []

    sorted_diffs = sorted(differences, key=_get_diff_index)
    merge_threshold = 2 * context_lines + 1

    hunks = []
    current_diffs = [sorted_diffs[0]]
    current_start = _get_diff_index(sorted_diffs[0])
    current_end = current_start

    for diff in sorted_diffs[1:]:
        idx = _get_diff_index(diff)
        if idx - current_end <= merge_threshold:
            current_diffs.append(diff)
            current_end = max(current_end, idx)
        else:
            # Finalize current hunk
            ctx_before = []
            ctx_after = []
            if source_texts:
                start = max(0, current_start - context_lines)
                ctx_before = source_texts[start:current_start]
                end_pos = min(len(source_texts), current_end + context_lines + 1)
                ctx_after = source_texts[current_end + 1:end_pos]
            hunks.append({
                "start_index": current_start,
                "end_index": current_end,
                "differences": current_diffs,
                "context_before": ctx_before,
                "context_after": ctx_after,
            })
            current_diffs = [diff]
            current_start = idx
            current_end = idx

    # Finalize last hunk
    ctx_before = []
    ctx_after = []
    if source_texts:
        start = max(0, current_start - context_lines)
        ctx_before = source_texts[start:current_start]
        end_pos = min(len(source_texts), current_end + context_lines + 1)
        ctx_after = source_texts[current_end + 1:end_pos]
    hunks.append({
        "start_index": current_start,
        "end_index": current_end,
        "differences": current_diffs,
        "context_before": ctx_before,
        "context_after": ctx_after,
    })

    return hunks


def build_structure_report(
    result: StructureComparisonResult,
    *,
    metadata: Optional[ReportMetadata] = None,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    reference_texts: Optional[List[str]] = None,
    candidate_texts: Optional[List[str]] = None,
) -> str:
    """Build a consolidated HTML report from a structure comparison result.

    Returns an HTML string suitable for logger.info(msg, html=True).
    Returns empty string if result.passed is True.
    """
    if result.passed:
        return ""

    summary = _compute_summary(result)
    parts = []

    # Outer container
    parts.append('<div style="font-family:monospace;font-size:12px;border:1px solid #ccc;'
                 'border-radius:4px;margin:4px 0;max-width:100%;overflow-x:auto;">')

    # Title
    parts.append('<div style="background:#f0f0f0;padding:8px 12px;border-bottom:1px solid #ccc;'
                 'font-weight:bold;font-size:13px;">PDF Structure Comparison Report</div>')

    # Metadata
    if metadata:
        parts.append('<div style="padding:6px 12px;border-bottom:1px solid #eee;font-size:11px;">')
        parts.append(f'<div><b>Reference:</b> {_escape(metadata.reference_name)}</div>')
        parts.append(f'<div><b>Candidate:</b> {_escape(metadata.candidate_name)}</div>')
        mode_str = _escape(metadata.comparison_mode)
        page_str = ""
        if metadata.page_count_ref is not None or metadata.page_count_cand is not None:
            page_str = f' | <b>Pages:</b> {metadata.page_count_ref or "?"} ref / {metadata.page_count_cand or "?"} cand'
        parts.append(f'<div><b>Mode:</b> {mode_str}{page_str}</div>')
        if metadata.exclusions_applied:
            exc_str = ", ".join(_escape(e) for e in metadata.exclusions_applied)
            parts.append(f'<div><b>Exclusions:</b> {exc_str}</div>')
        parts.append('</div>')

    # Summary
    parts.append('<div style="padding:8px 12px;border-bottom:1px solid #ccc;background:#fafafa;">')
    parts.append(f'<div><b>{summary.total_differences}</b> difference(s)</div>')
    parts.append('<div style="margin-top:4px;">')
    parts.append(f'<span style="background:#fdd;padding:2px 6px;border-radius:2px;margin-right:4px;">{summary.missing_count} missing</span>')
    parts.append(f'<span style="background:#dfd;padding:2px 6px;border-radius:2px;margin-right:4px;">{summary.extra_count} extra</span>')
    parts.append(f'<span style="background:#ffd;padding:2px 6px;border-radius:2px;margin-right:4px;">{summary.mismatch_count} mismatch</span>')
    if summary.geometry_count:
        parts.append(f'<span style="background:#eee;padding:2px 6px;border-radius:2px;margin-right:4px;">{summary.geometry_count} geometry</span>')
    if summary.other_count:
        parts.append(f'<span style="background:#eee;padding:2px 6px;border-radius:2px;">{summary.other_count} other</span>')
    parts.append('</div></div>')

    # Content sections
    parts.append('<div style="padding:4px 12px;">')

    total_hunks = 0

    # Page-level differences
    if result.page_differences:
        for page_num in sorted(result.page_differences.keys()):
            diffs = result.page_differences[page_num]
            hunks = _group_into_hunks(diffs, context_lines, reference_texts)
            total_hunks += len(hunks)
            parts.append(f'<div style="font-weight:bold;margin:8px 0 4px;border-bottom:1px solid #eee;padding-bottom:4px;">'
                         f'Page {page_num} &mdash; {len(hunks)} hunk(s), {len(diffs)} difference(s)</div>')
            for i, hunk in enumerate(hunks):
                if total_hunks > MAX_HUNKS_BEFORE_COLLAPSE and i > 0:
                    parts.append(f'<div style="color:#888;font-style:italic;margin:4px 0;">... and more hunks (showing first {MAX_HUNKS_BEFORE_COLLAPSE})</div>')
                    break
                _render_hunk_to_parts(parts, hunk, i + 1)

    # Document-level differences
    if result.document_differences:
        hunks = _group_into_hunks(result.document_differences, context_lines, reference_texts)
        total_hunks += len(hunks)
        parts.append(f'<div style="font-weight:bold;margin:8px 0 4px;border-bottom:1px solid #eee;padding-bottom:4px;">'
                     f'Document (text-only) &mdash; {len(hunks)} hunk(s), {len(result.document_differences)} difference(s)</div>')
        for i, hunk in enumerate(hunks):
            _render_hunk_to_parts(parts, hunk, i + 1)

    # Word-level differences
    if hasattr(result, 'word_differences') and result.word_differences:
        hunks = _group_into_hunks(result.word_differences, context_lines, reference_texts)
        total_hunks += len(hunks)
        parts.append(f'<div style="font-weight:bold;margin:8px 0 4px;border-bottom:1px solid #eee;padding-bottom:4px;">'
                     f'Document (word-level) &mdash; {len(hunks)} hunk(s), {len(result.word_differences)} difference(s)</div>')
        rendered = 0
        for i, hunk in enumerate(hunks):
            if rendered >= MAX_HUNKS_BEFORE_COLLAPSE:
                remaining = len(hunks) - rendered
                parts.append(f'<div style="color:#888;font-style:italic;margin:4px 0;">... {remaining} more hunk(s) not shown</div>')
                break
            _render_hunk_to_parts(parts, hunk, i + 1)
            rendered += 1

    # Summary line
    if result.summary:
        parts.append('<div style="margin-top:8px;padding-top:4px;border-top:1px solid #eee;color:#666;font-size:11px;">')
        for entry in result.summary:
            parts.append(f'<div>{_escape(str(entry))}</div>')
        parts.append('</div>')

    parts.append('</div>')  # close content
    parts.append('</div>')  # close outer container

    summary.hunk_count = total_hunks
    return "\n".join(parts)


def _render_hunk_to_parts(parts: List[str], hunk: dict, hunk_number: int) -> None:
    """Render a hunk into the HTML parts list."""
    start = hunk["start_index"]
    end = hunk["end_index"]
    label = f"line {start}" if start == end else f"lines {start}&ndash;{end}"

    parts.append(f'<div style="margin:4px 0;padding:4px 8px;border-left:3px solid #888;background:#fafafa;">')
    parts.append(f'<div style="font-size:10px;color:#888;margin-bottom:2px;">Hunk {hunk_number} ({label})</div>')

    # Context before
    if hunk["context_before"]:
        ctx = " ".join(_truncate(t, 40) for t in hunk["context_before"])
        parts.append(f'<div style="color:#888;">...{_escape(ctx)}...</div>')

    # Differences
    for diff in hunk["differences"]:
        parts.append(_render_diff_html(diff))

    # Context after
    if hunk["context_after"]:
        ctx = " ".join(_truncate(t, 40) for t in hunk["context_after"])
        parts.append(f'<div style="color:#888;">...{_escape(ctx)}...</div>')

    parts.append('</div>')


def build_structure_report_plain_text(
    result: StructureComparisonResult,
    *,
    metadata: Optional[ReportMetadata] = None,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    reference_texts: Optional[List[str]] = None,
    candidate_texts: Optional[List[str]] = None,
) -> str:
    """Build a plain-text version of the consolidated report.

    Returns empty string if result.passed is True.
    """
    if result.passed:
        return ""

    summary = _compute_summary(result)
    lines = []

    lines.append("=" * 60)
    lines.append("PDF Structure Comparison Report")
    lines.append("=" * 60)

    if metadata:
        lines.append(f"Reference: {metadata.reference_name}")
        lines.append(f"Candidate: {metadata.candidate_name}")
        lines.append(f"Mode: {metadata.comparison_mode}")
        if metadata.exclusions_applied:
            lines.append(f"Exclusions: {', '.join(metadata.exclusions_applied)}")

    lines.append("-" * 60)
    lines.append(f"{summary.total_differences} difference(s): "
                 f"{summary.missing_count} missing, {summary.extra_count} extra, "
                 f"{summary.mismatch_count} mismatch, {summary.geometry_count} geometry, "
                 f"{summary.other_count} other")
    lines.append("-" * 60)

    # Page-level
    if result.page_differences:
        for page_num in sorted(result.page_differences.keys()):
            diffs = result.page_differences[page_num]
            hunks = _group_into_hunks(diffs, context_lines, reference_texts)
            lines.append(f"\nPage {page_num} -- {len(hunks)} hunk(s), {len(diffs)} difference(s)")
            for i, hunk in enumerate(hunks):
                _render_hunk_plain(lines, hunk, i + 1)

    # Document-level
    if result.document_differences:
        hunks = _group_into_hunks(result.document_differences, context_lines, reference_texts)
        lines.append(f"\nDocument (text-only) -- {len(hunks)} hunk(s), {len(result.document_differences)} difference(s)")
        for i, hunk in enumerate(hunks):
            _render_hunk_plain(lines, hunk, i + 1)

    # Word-level
    if hasattr(result, 'word_differences') and result.word_differences:
        hunks = _group_into_hunks(result.word_differences, context_lines, reference_texts)
        lines.append(f"\nDocument (word-level) -- {len(hunks)} hunk(s), {len(result.word_differences)} difference(s)")
        for i, hunk in enumerate(hunks):
            if i >= MAX_HUNKS_BEFORE_COLLAPSE:
                lines.append(f"  ... {len(hunks) - i} more hunk(s) not shown")
                break
            _render_hunk_plain(lines, hunk, i + 1)

    if result.summary:
        lines.append("")
        for entry in result.summary:
            lines.append(f"Note: {entry}")

    lines.append("=" * 60)
    return "\n".join(lines)


def _render_hunk_plain(lines: List[str], hunk: dict, hunk_number: int) -> None:
    """Render a hunk into the plain text lines list."""
    start = hunk["start_index"]
    end = hunk["end_index"]
    label = f"line {start}" if start == end else f"lines {start}-{end}"
    lines.append(f"  Hunk {hunk_number} ({label})")

    if hunk["context_before"]:
        ctx = " ".join(_truncate(t, 40) for t in hunk["context_before"])
        lines.append(f"  ...{ctx}...")

    for diff in hunk["differences"]:
        lines.append(_render_diff_plain(diff))

    if hunk["context_after"]:
        ctx = " ".join(_truncate(t, 40) for t in hunk["context_after"])
        lines.append(f"  ...{ctx}...")
