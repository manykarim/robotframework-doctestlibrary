from inspect import signature
from pprint import pprint, pformat
from deepdiff import DeepDiff
from robot.api import logger
from robot.api.deco import keyword, library
import fitz
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Pattern, Tuple
from DocTest.config import DEFAULT_DPI
from DocTest.DocumentRepresentation import DocumentRepresentation
from DocTest.Downloader import is_url, download_file_from_url
from DocTest.PdfStructureComparator import (
    StructureTolerance,
    compare_document_structures,
    compare_document_text_only,
)
from DocTest.PdfStructureModels import (
    DocumentStructure,
    PageStructure,
    StructureExtractionConfig,
    TextBlock,
    TextLine,
)
from DocTest.llm import LLMDependencyError, load_llm_settings
from DocTest.TextNormalization import apply_character_replacements

if TYPE_CHECKING:  # pragma: no cover - typing only
    from DocTest.llm.types import LLMDecision, LLMDecisionLabel
else:  # pragma: no cover - runtime fallback
    LLMDecision = Any  # type: ignore[misc, assignment]
    LLMDecisionLabel = Any  # type: ignore[misc, assignment]


_PDF_LLM_RUNTIME: Optional[Tuple[Any, Any, Any]] = None


def _coerce_label_value(label: Any) -> str:
    value = getattr(label, "value", label)
    return str(value)


def _decision_equals_flag(label: Any, enum_cls: Any) -> bool:
    candidate = _coerce_label_value(label).lower()
    if enum_cls is None:
        return candidate == "flag"
    flag_member = getattr(enum_cls, "FLAG", None)
    if flag_member is None:
        return candidate == "flag"
    flag_value = _coerce_label_value(flag_member).lower()
    return candidate == flag_value


def _load_pdf_llm_runtime() -> Tuple[Any, Any, Any]:
    global _PDF_LLM_RUNTIME
    if _PDF_LLM_RUNTIME is not None:
        return _PDF_LLM_RUNTIME
    try:
        from DocTest.llm.client import assess_pdf_diff, create_binary_content
        from DocTest.llm.types import LLMDecisionLabel as _LLMDecisionLabel
    except Exception as exc:  # pragma: no cover - optional dependency missing
        raise LLMDependencyError() from exc
    _PDF_LLM_RUNTIME = (assess_pdf_diff, create_binary_content, _LLMDecisionLabel)
    return _PDF_LLM_RUNTIME


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    return bool(value)


def _as_float(value, default):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int_or_none(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

ROBOT_AUTO_KEYWORDS = False

class PdfTest(object):
    """Robot Framework library for PDF content and structure comparison.

    Optional initialization arguments:

    | =Argument= | =Description= |
    | ``character_replacements`` | Dict mapping characters to replacements, applied to all text extraction/comparison. Example: ``{'\u00A0': ' '}`` to normalize non-breaking spaces. |
    """

    def __init__(self, character_replacements: Optional[Dict[str, str]] = None, **kwargs):
        fitz.TOOLS.set_aa_level(0)
        self.character_replacements = self._parse_character_replacements(character_replacements)

    def _parse_character_replacements(
        self, value: Optional[Any]
    ) -> Optional[Dict[str, str]]:
        """Parse character_replacements from various input formats."""
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                logger.warn(f"Invalid character_replacements JSON: {value}")
        return None
    
    @keyword
    def compare_pdf_documents(self, reference_document, candidate_document, **kwargs):
        """Compare two PDF documents by metadata, content, and optionally layout.

        ``reference_document`` and ``candidate_document`` may be file paths or URLs.
        Use the ``compare`` argument (comma or ``|`` separated) to select which facets to
        validate. Supported values:

            - ``all`` (default)
            - ``metadata``
            - ``text``
            - ``fonts``
            - ``images``
            - ``signatures``
            - ``structure`` (layout + line/geometry comparison that ignores font subset differences)

        Masking dynamic regions:

            - ``mask`` accepts the same formats as :keyword:`Compare Images` – JSON/dict/list payloads,
              ``top:percent`` strings, or a path to a JSON mask file. Use ``unit=pt`` to provide
              coordinates directly in PDF points.
            - ``text_mask_patterns`` (list or string) allows regex-based filtering of text lines.

        Ligature handling:

            - ``ignore_ligatures`` (bool, default ``False``) replaces common ligatures (e.g. ``ﬁ``)
              with their ASCII counterparts before comparing text or structure.

        When ``structure`` is requested the following optional kwargs control behaviour:

            - ``structure_position_tolerance`` (float, default ``15.0``) – max positional delta in points
            - ``structure_size_tolerance`` (float, default equals position tolerance) – max width/height delta in points
            - ``structure_relative_tolerance`` (float, default ``0.05``) – fallback percentage tolerance
            - ``structure_case_sensitive`` (bool, default ``True``) – toggle case-sensitive text matching
            - ``structure_strip_font_subset`` (bool, default ``True``) – drop font subset prefixes
            - ``structure_collapse_whitespace`` (bool, default ``True``)
            - ``structure_strip_line_edges`` (bool, default ``True``)
            - ``structure_drop_empty_lines`` (bool, default ``True``)
            - ``structure_whitespace_replacement`` (str, default single space)
            - ``structure_round_precision`` (int or ``None``, default ``3``)
            - ``structure_dpi`` (int, default: library DPI)
            - ``ignore_page_boundaries`` (bool, default ``False``) – ignore page breaks and compare text in reading order across entire document
            - ``check_geometry`` (bool, default ``True``) – when ``False``, skip line position/size comparison
            - ``check_block_count`` (bool, default ``True``) – when ``False``, skip block count validation

        Optional LLM integration parameters:

        | =Arguments= | =Description= |
        | ``llm`` / ``llm_enabled`` | When ``${True}``, forward detected differences to the configured LLM. Default ``${False}``. |
        | ``llm_override`` | Allow an approving LLM verdict to override baseline comparison failures. Default ``${False}``. |
        | ``llm_prompt`` / ``llm_pdf_prompt`` | Custom prompt for this comparison. Default ``None`` uses the built-in prompt. |

        Result passes only if every requested facet matches.

        Examples:
        | `Compare Pdf Documents`    reference.pdf    candidate.pdf
        | `Compare Pdf Documents`    reference.pdf    candidate.pdf    mask=${CURDIR}${/}mask.json
        | `Compare Pdf Documents`    reference.pdf    candidate.pdf    compare=structure    ignore_ligatures=${True}
        | `Compare Pdf Documents`    reference.pdf    candidate.pdf    compare=structure|metadata    structure_position_tolerance=5.0

        """
        llm_requested = bool(
            kwargs.pop("llm", False)
            or kwargs.pop("llm_enabled", False)
            or kwargs.pop("use_llm", False)
        )
        llm_override_result = bool(kwargs.pop("llm_override", False))
        llm_overrides: Dict[str, Optional[str]] = {}
        llm_general_notes: List[str] = []
        llm_differences: List[Dict[str, Any]] = []
        custom_llm_prompt = kwargs.pop("llm_prompt", None)
        pdf_prompt_override = kwargs.pop("llm_pdf_prompt", None)
        if pdf_prompt_override:
            custom_llm_prompt = pdf_prompt_override
        llm_keys = [
            "llm_provider",
            "llm_models",
            "llm_api_key",
            "llm_base_url",
            "llm_temperature",
            "llm_max_output_tokens",
            "llm_request_timeout",
            "azure_openai_endpoint",
            "azure_openai_deployment",
            "azure_openai_api_version",
            "azure_openai_api_key",
        ]
        for key in llm_keys:
            if key in kwargs:
                value = kwargs.pop(key)
                if value is not None:
                    llm_overrides[key] = value

        mask_value = kwargs.pop('mask', None)
        text_mask_patterns_arg = kwargs.pop('text_mask_patterns', None)
        ignore_ligatures = _as_bool(kwargs.pop('ignore_ligatures', False))
        check_pdf_text = _as_bool(kwargs.pop('check_pdf_text', False))

        # Parse character_replacements from kwargs or use instance default
        char_replacements_arg = kwargs.pop('character_replacements', None)
        char_replacements = self._parse_character_replacements(char_replacements_arg)
        if char_replacements is None:
            char_replacements = self.character_replacements

        llm_general_notes.append(f"check_pdf_text={check_pdf_text}")
        llm_general_notes.append(f"ignore_ligatures={ignore_ligatures}")
        if char_replacements:
            llm_general_notes.append(f"character_replacements={len(char_replacements)} mapping(s)")

        compare_value = kwargs.pop('compare', "all")
        compare_value = compare_value.replace('|', ',')
        compare_tokens = [token.strip().lower() for token in compare_value.split(',') if token.strip()]
        if not compare_tokens:
            compare_tokens = ['all']
        compare_set = set(compare_tokens)
        llm_general_notes.append(f"Compare facets: {sorted(compare_set)}")
        compare_all = 'all' in compare_set

        structure_position_tolerance = _as_float(kwargs.pop('structure_position_tolerance', 15.0), 15.0)
        structure_size_tolerance = _as_float(kwargs.pop('structure_size_tolerance', structure_position_tolerance), structure_position_tolerance)
        structure_relative_tolerance = _as_float(kwargs.pop('structure_relative_tolerance', 0.05), 0.05)
        structure_case_sensitive = _as_bool(kwargs.pop('structure_case_sensitive', True), True)
        structure_strip_font_subset = _as_bool(kwargs.pop('structure_strip_font_subset', True), True)
        structure_collapse_whitespace = _as_bool(kwargs.pop('structure_collapse_whitespace', True), True)
        structure_strip_line_edges = _as_bool(kwargs.pop('structure_strip_line_edges', True), True)
        structure_drop_empty_lines = _as_bool(kwargs.pop('structure_drop_empty_lines', True), True)
        structure_whitespace_replacement = kwargs.pop('structure_whitespace_replacement', " ")
        structure_round_precision = _as_int_or_none(kwargs.pop('structure_round_precision', 3))
        base_dpi = _as_int_or_none(kwargs.pop('dpi', None))
        structure_dpi = _as_int_or_none(kwargs.pop('structure_dpi', None))
        if structure_dpi is None:
            structure_dpi = base_dpi
        if structure_dpi is None:
            structure_dpi = DEFAULT_DPI

        # New parameters for controlling structure comparison behavior
        ignore_page_boundaries = _as_bool(kwargs.pop('ignore_page_boundaries', False), False)
        check_geometry = _as_bool(kwargs.pop('check_geometry', True), True)
        check_block_count = _as_bool(kwargs.pop('check_block_count', True), True)

        # When ignoring page boundaries, disable geometry and block count checks
        if ignore_page_boundaries:
            check_geometry = False
            check_block_count = False

        structure_requested = 'structure' in compare_set
        if structure_requested:
            llm_general_notes.append(
                f"structure_tolerance(position={structure_position_tolerance}, size={structure_size_tolerance}, relative={structure_relative_tolerance})"
            )
            if ignore_page_boundaries:
                llm_general_notes.append("ignore_page_boundaries=True")

        structure_extraction_config = StructureExtractionConfig(
            collapse_whitespace=structure_collapse_whitespace,
            strip_font_subset=structure_strip_font_subset,
            whitespace_replacement=str(structure_whitespace_replacement),
            strip_line_edges=structure_strip_line_edges,
            drop_empty_lines=structure_drop_empty_lines,
            round_precision=structure_round_precision,
            normalize_ligatures=ignore_ligatures,
            character_replacements=char_replacements,
        )
        text_extraction_config = StructureExtractionConfig(
            collapse_whitespace=True,
            strip_font_subset=True,
            whitespace_replacement=" ",
            strip_line_edges=True,
            drop_empty_lines=True,
            round_precision=structure_round_precision,
            normalize_ligatures=ignore_ligatures,
            character_replacements=char_replacements,
        )

        mask_file, mask_payload, text_pattern_values, mask_warnings = self._resolve_mask_arguments(
            mask_value,
            text_mask_patterns_arg,
        )
        for warning in mask_warnings:
            logger.warn(warning)

        compiled_text_patterns, pattern_warnings = self._compile_text_mask_patterns(text_pattern_values)
        for warning in pattern_warnings:
            logger.warn(warning)

        mask_applied = bool(mask_file or mask_payload)
        llm_general_notes.append(f"mask_applied={'yes' if mask_applied else 'no'}")
        llm_general_notes.append(f"text_mask_patterns={'yes' if text_pattern_values else 'no'}")

        reference_document = self._ensure_local_document(reference_document)
        candidate_document = self._ensure_local_document(candidate_document)

        reference_repr = DocumentRepresentation(
            reference_document,
            dpi=structure_dpi,
            ignore_area_file=mask_file,
            ignore_area=mask_payload,
        )
        candidate_repr = DocumentRepresentation(
            candidate_document,
            dpi=structure_dpi,
            ignore_area_file=mask_file,
            ignore_area=mask_payload,
        )

        try:
            differences_detected = False
            reference_snapshot = self._build_document_snapshot(
                reference_repr,
                extraction_config=text_extraction_config,
                text_mask_patterns=compiled_text_patterns,
            )
            candidate_snapshot = self._build_document_snapshot(
                candidate_repr,
                extraction_config=text_extraction_config,
                text_mask_patterns=compiled_text_patterns,
            )

            if reference_snapshot['page_count'] != candidate_snapshot['page_count']:
                raise AssertionError("Documents have different number of pages.")

            metadata_requested = compare_all or 'metadata' in compare_set
            signatures_requested = compare_all or 'signatures' in compare_set
            text_requested = compare_all or 'text' in compare_set
            fonts_requested = compare_all or 'fonts' in compare_set
            images_requested = compare_all or 'images' in compare_set

            def _record_diff(facet: str, description: str, diff_payload: Any):
                nonlocal differences_detected
                differences_detected = True
                print(description)
                pprint(diff_payload, width=200)
                llm_differences.append(
                    {
                        "facet": facet,
                        "description": description,
                        "details": pformat(diff_payload, width=200),
                    }
                )

            if metadata_requested:
                diff = DeepDiff(reference_snapshot['metadata'], candidate_snapshot['metadata'])
                if diff:
                    _record_diff(
                        "metadata",
                        "Metadata differs between reference and candidate.",
                        diff,
                    )

            if signatures_requested:
                diff = DeepDiff(reference_snapshot['sigflags'], candidate_snapshot['sigflags'])
                if diff:
                    _record_diff(
                        "signatures",
                        "PDF signature flags differ.",
                        diff,
                    )

            for ref_page, cand_page in zip(reference_snapshot['pages'], candidate_snapshot['pages']):
                page_number = ref_page['number']

                if compare_all and ref_page.get('rotation') != cand_page.get('rotation'):
                    _record_diff(
                        "rotation",
                        f"Page {page_number} rotation differs.",
                        {
                            "reference": ref_page.get('rotation'),
                            "candidate": cand_page.get('rotation'),
                        },
                    )

                if text_requested:
                    ref_text_lines = sorted(ref_page['text'])
                    cand_text_lines = sorted(cand_page['text'])
                    diff = DeepDiff(ref_text_lines, cand_text_lines)
                    if diff:
                        _record_diff(
                            "text",
                            f"Page {page_number} text content differs.",
                            diff,
                        )

                if fonts_requested:
                    diff = DeepDiff(ref_page['fonts'], cand_page['fonts'])
                    if diff:
                        _record_diff(
                            "fonts",
                            f"Page {page_number} font usage differs.",
                            diff,
                        )

                if images_requested:
                    diff = DeepDiff(ref_page['images'], cand_page['images'])
                    if diff:
                        _record_diff(
                            "images",
                            f"Page {page_number} embedded images differ.",
                            diff,
                        )

                if signatures_requested:
                    diff = DeepDiff(ref_page['signatures'], cand_page['signatures'])
                    if diff:
                        _record_diff(
                            "signatures",
                            f"Page {page_number} form signatures differ.",
                            diff,
                        )

            if structure_requested:
                tolerance = StructureTolerance(
                    position=structure_position_tolerance,
                    size=structure_size_tolerance,
                    relative=structure_relative_tolerance,
                )
                structure_result = self._perform_structure_comparison(
                    reference_document=reference_document,
                    candidate_document=candidate_document,
                    tolerance=tolerance,
                    extraction_config=structure_extraction_config,
                    case_sensitive=structure_case_sensitive,
                    dpi=structure_dpi,
                    reference_representation=reference_repr,
                    candidate_representation=candidate_repr,
                    text_mask_patterns=compiled_text_patterns,
                    ignore_page_boundaries=ignore_page_boundaries,
                    check_geometry=check_geometry,
                    check_block_count=check_block_count,
                )
                if not structure_result.passed:
                    differences_detected = True
                    summary = getattr(structure_result, "summary", None)
                    page_diffs = getattr(structure_result, "page_differences", None)
                    doc_diffs = getattr(structure_result, "document_differences", None)
                    details_parts: List[str] = []
                    if summary:
                        details_parts.extend(str(item) for item in summary)
                    if page_diffs:
                        for page, diffs in page_diffs.items():
                            for diff in diffs:
                                details_parts.append(f"Page {page}: {diff.message}")
                    if doc_diffs:
                        for diff in doc_diffs:
                            details_parts.append(f"Document: {diff.message}")
                    llm_differences.append(
                        {
                            "facet": "structure",
                            "description": "PDF structural comparison failed.",
                            "details": "\n".join(details_parts) if details_parts else "Structure comparison differences detected.",
                        }
                    )

            llm_decision = None
            llm_label_enum = None
            if differences_detected and llm_requested:
                try:
                    assess_pdf_diff_fn, create_binary_content_fn, llm_label_enum = (
                        _load_pdf_llm_runtime()
                    )
                except LLMDependencyError as exc:
                    print(str(exc))
                else:
                    llm_decision = self._handle_llm_for_pdf_differences(
                        reference_document=reference_document,
                        candidate_document=candidate_document,
                        differences=llm_differences,
                        overrides=llm_overrides,
                        notes=llm_general_notes,
                        custom_prompt=custom_llm_prompt,
                        create_binary_content_fn=create_binary_content_fn,
                        assess_pdf_diff_fn=assess_pdf_diff_fn,
                    )
                if llm_decision:
                    decision_value = _coerce_label_value(llm_decision.decision)
                    print(
                        f"LLM decision: {decision_value} "
                        f"(confidence={llm_decision.confidence!r}) - {llm_decision.reason}"
                    )
                    if llm_decision.notes:
                        print(f"LLM notes: {llm_decision.notes}")
                    if llm_override_result and llm_decision.is_positive:
                        print("LLM approved PDF differences. Overriding baseline failure.")
                        differences_detected = False
                    elif _decision_equals_flag(llm_decision.decision, llm_label_enum):
                        print("LLM returned FLAG - keeping original PDF comparison result.")
                    elif not llm_decision.is_positive and llm_override_result:
                        print("LLM rejected PDF differences. Baseline failure will stand.")

            if differences_detected:
                raise AssertionError('The compared PDF Document Data is different.')
        finally:
            reference_repr.close()
            candidate_repr.close()
        # if reference!=candidate:
        #     pprint(DeepDiff(reference, candidate), width=200)
        #     print("Reference Document:")
        #     pprint(reference)
        #     print("Candidate Document:")
        #     pprint(candidate)           
        #     raise AssertionError('The compared PDF Document Data is different.')

    @keyword
    def compare_pdf_structure(self, reference_document, candidate_document, **kwargs):
        """Assert PDF structure/layout matches while tolerating font substitutions.

        The keyword compares per-line text order and bounding boxes between two PDFs.
        Fonts are normalised (subset prefixes removed) and layout is evaluated against
        configurable tolerances. Both ``reference_document`` and ``candidate_document``
        can be file paths or URLs.

        Optional arguments:

            - ``position_tolerance`` (float, default ``15.0``): absolute max delta (points) for line positions.
            - ``size_tolerance`` (float, default equals ``position_tolerance``): absolute delta for width/height.
            - ``relative_tolerance`` (float, default ``0.05``): extra tolerance expressed as percentage of reference value.
            - ``case_sensitive`` (bool, default ``True``): toggle case-sensitive text comparison.
            - ``strip_font_subset`` (bool, default ``True``): remove PDF font subset prefixes.
            - ``collapse_whitespace`` (bool, default ``True``): collapse consecutive whitespace before comparing.
            - ``strip_line_edges`` (bool, default ``True``): trim leading/trailing whitespace.
            - ``drop_empty_lines`` (bool, default ``True``): ignore empty text lines after normalisation.
            - ``whitespace_replacement`` (str, default single space): character inserted when collapsing whitespace.
            - ``round_precision`` (int or ``None``, default ``3``): precision for rounding bounding boxes.
            - ``dpi`` (int, optional): DPI used when building the structure (defaults to library DPI).
            - ``mask``: JSON/dict/list/string mask definition (same formats as ``Compare Pdf Documents``) used to ignore dynamic regions.
            - ``text_mask_patterns``: regex or list of regex strings to skip lines during comparison.
            - ``ignore_ligatures`` (bool, default ``False``): normalise common ligatures (``ﬁ`` → ``fi``) prior to comparison.
            - ``ignore_page_boundaries`` (bool, default ``False``): ignore page breaks and compare text content in reading order across the entire document. When enabled, geometry and block structure are not checked. Useful when font/size changes cause text to reflow across pages.
            - ``check_geometry`` (bool, default ``True``): when ``False``, skip line position/size comparison. Useful for comparing content when layout may differ. Automatically set to ``False`` when ``ignore_page_boundaries`` is ``True``.
            - ``check_block_count`` (bool, default ``True``): when ``False``, skip block count validation per page. Automatically set to ``False`` when ``ignore_page_boundaries`` is ``True``.

        Examples:
        | `Compare Pdf Structure`    reference.pdf    candidate.pdf
        | `Compare Pdf Structure`    reference.pdf    candidate.pdf    position_tolerance=3    relative_tolerance=0.02
        | `Compare Pdf Structure`    reference.pdf    candidate.pdf    mask=${CURDIR}${/}mask.json    text_mask_patterns=\\d{4}-\\d{4}    ignore_ligatures=${True}
        | `Compare Pdf Structure`    reference.pdf    candidate.pdf    ignore_page_boundaries=${True}
        | `Compare Pdf Structure`    reference.pdf    candidate.pdf    check_geometry=${False}    check_block_count=${False}
        | `Run Keyword And Expect Error`    The compared PDF structure is different.    Compare Pdf Structure    reference.pdf    candidate_with_changed_text.pdf

        """
        reference_document = self._ensure_local_document(reference_document)
        candidate_document = self._ensure_local_document(candidate_document)

        position_tolerance = _as_float(kwargs.get('position_tolerance', 15.0), 15.0)
        size_tolerance = _as_float(kwargs.get('size_tolerance', position_tolerance), position_tolerance)
        relative_tolerance = _as_float(kwargs.get('relative_tolerance', 0.05), 0.05)
        case_sensitive = _as_bool(kwargs.get('case_sensitive', True), True)
        strip_font_subset = _as_bool(kwargs.get('strip_font_subset', True), True)
        collapse_whitespace = _as_bool(kwargs.get('collapse_whitespace', True), True)
        strip_line_edges = _as_bool(kwargs.get('strip_line_edges', True), True)
        drop_empty_lines = _as_bool(kwargs.get('drop_empty_lines', True), True)
        whitespace_replacement = kwargs.get('whitespace_replacement', " ")
        round_precision = _as_int_or_none(kwargs.get('round_precision', 3))
        dpi = _as_int_or_none(kwargs.get('dpi'))
        if dpi is None:
            dpi = DEFAULT_DPI

        mask_value = kwargs.get('mask')
        text_mask_patterns_arg = kwargs.get('text_mask_patterns')
        ignore_ligatures = _as_bool(kwargs.get('ignore_ligatures', False), False)

        # Parse character_replacements from kwargs or use instance default
        char_replacements_arg = kwargs.get('character_replacements')
        char_replacements = self._parse_character_replacements(char_replacements_arg)
        if char_replacements is None:
            char_replacements = self.character_replacements

        # New parameters for controlling comparison behavior
        ignore_page_boundaries = _as_bool(kwargs.get('ignore_page_boundaries', False), False)
        check_geometry = _as_bool(kwargs.get('check_geometry', True), True)
        check_block_count = _as_bool(kwargs.get('check_block_count', True), True)

        # When ignoring page boundaries, disable geometry and block count checks
        if ignore_page_boundaries:
            check_geometry = False
            check_block_count = False

        extraction_config = StructureExtractionConfig(
            collapse_whitespace=collapse_whitespace,
            strip_font_subset=strip_font_subset,
            whitespace_replacement=str(whitespace_replacement),
            strip_line_edges=strip_line_edges,
            drop_empty_lines=drop_empty_lines,
            round_precision=round_precision,
            normalize_ligatures=ignore_ligatures,
            character_replacements=char_replacements,
        )
        tolerance = StructureTolerance(
            position=position_tolerance,
            size=size_tolerance,
            relative=relative_tolerance,
        )
        mask_file, mask_payload, text_pattern_values, mask_warnings = self._resolve_mask_arguments(
            mask_value,
            text_mask_patterns_arg,
        )
        for warning in mask_warnings:
            logger.warn(warning)
        compiled_text_patterns, pattern_warnings = self._compile_text_mask_patterns(text_pattern_values)
        for warning in pattern_warnings:
            logger.warn(warning)

        reference_repr = DocumentRepresentation(
            reference_document,
            dpi=dpi,
            ignore_area_file=mask_file,
            ignore_area=mask_payload,
        )
        candidate_repr = DocumentRepresentation(
            candidate_document,
            dpi=dpi,
            ignore_area_file=mask_file,
            ignore_area=mask_payload,
        )
        try:
            result = self._perform_structure_comparison(
                reference_document=reference_document,
                candidate_document=candidate_document,
                tolerance=tolerance,
                extraction_config=extraction_config,
                case_sensitive=case_sensitive,
                dpi=dpi,
                reference_representation=reference_repr,
                candidate_representation=candidate_repr,
                text_mask_patterns=compiled_text_patterns,
                ignore_page_boundaries=ignore_page_boundaries,
                check_geometry=check_geometry,
                check_block_count=check_block_count,
            )
        finally:
            reference_repr.close()
            candidate_repr.close()
        if not result.passed:
            raise AssertionError('The compared PDF structure is different.')

    @keyword
    def check_text_content(self, expected_text_list, candidate_document):
        """*DEPRECATED!!* Use keyword `PDF Should Contain Strings` instead.
        
        Checks if each item provided in the list ``expected_text_list`` appears in the PDF File ``candidate_document``.
        
        ``expected_text_list`` is a list of strings, ``candidate_document`` is the path to a PDF File.
        
        Examples:

        | @{strings}=    Create List    One String    Another String   
        | `Check Text Content`    ${strings}    candidate.pdf   
        
        """
        if is_url(candidate_document):
            candidate_document = download_file_from_url(candidate_document)
        doc = fitz.open(candidate_document)
        missing_text_list = []
        all_texts_were_found = None
        for text_item in expected_text_list:
            text_found_in_page = False
            for page in doc.pages():
                if any(text_item in s for s in page.get_text("text").splitlines()):
                    text_found_in_page = True
                    break
            if text_found_in_page:
                continue
            all_texts_were_found = False
            missing_text_list.append({'text':text_item, 'document':candidate_document})
        if all_texts_were_found is False:
            print(missing_text_list)
            doc = None
            raise AssertionError('Some expected texts were not found in document')
    
    @keyword
    def PDF_should_contain_strings(self, expected_text_list, candidate_document, **kwargs):
        """Checks if each item provided in the list ``expected_text_list`` appears in the PDF File ``candidate_document``.

        ``expected_text_list`` is a list of strings or a single string, ``candidate_document`` is the path or URL to a PDF File.

        Optional arguments:

            - ``character_replacements`` (dict): Maps characters to replacements before comparison.
              Example: ``{'\u00A0': ' '}`` to normalize non-breaking spaces.

        Examples:

        | @{strings}=    Create List    One String    Another String
        | `PDF Should Contain Strings`    ${strings}    candidate.pdf
        | `PDF Should Contain Strings`    One String    candidate.pdf
        | `PDF Should Contain Strings`    ${strings}    candidate.pdf    character_replacements={'\u00A0': ' '}

        """
        # Parse character_replacements from kwargs or use instance default
        char_replacements_arg = kwargs.pop('character_replacements', None)
        char_replacements = self._parse_character_replacements(char_replacements_arg)
        if char_replacements is None:
            char_replacements = self.character_replacements

        if is_url(candidate_document):
            candidate_document = download_file_from_url(candidate_document)
        doc = fitz.open(candidate_document)
        # if expected_text_list is a string, convert it to a list
        if isinstance(expected_text_list, str):
            expected_text_list = [expected_text_list]
        missing_text_list = []
        found_text_list = []
        all_texts_were_found = None
        for text_item in expected_text_list:
            text_found_in_page = False
            for page in doc.pages():
                page_text = page.get_text("text")
                # Apply character replacements if configured
                if char_replacements:
                    page_text = apply_character_replacements(page_text, char_replacements)
                if any(text_item in s for s in page_text.splitlines()):
                    text_found_in_page = True
                    found_text_list.append({'text':text_item, 'document':candidate_document, 'page':page.number+1})
                    break
            if text_found_in_page:
                continue
            all_texts_were_found = False
            missing_text_list.append({'text':text_item, 'document':candidate_document})
        if all_texts_were_found is False:
            print(f"Missing Texts:\n{missing_text_list}")
            print(f"Found Texts:\n{found_text_list}")
            doc = None
            raise AssertionError('Some expected texts were not found in document')
        else:
            doc = None
            print(f"Found Texts:\n{found_text_list}")

    @keyword
    def PDF_should_not_contain_strings(self, expected_text_list, candidate_document, **kwargs):
        """Checks if each item provided in the list ``expected_text_list`` does NOT appear in the PDF File ``candidate_document``.

        ``expected_text_list`` is a list of strings or a single string, ``candidate_document`` is the path or URL to a PDF File.

        Optional arguments:

            - ``character_replacements`` (dict): Maps characters to replacements before comparison.
              Example: ``{'\u00A0': ' '}`` to normalize non-breaking spaces.

        Examples:

        | @{strings}=    Create List    One String    Another String
        | `PDF Should Not Contain Strings`    ${strings}    candidate.pdf
        | `PDF Should Not Contain Strings`    One String    candidate.pdf
        | `PDF Should Not Contain Strings`    ${strings}    candidate.pdf    character_replacements={'\u00A0': ' '}

        """
        # Parse character_replacements from kwargs or use instance default
        char_replacements_arg = kwargs.pop('character_replacements', None)
        char_replacements = self._parse_character_replacements(char_replacements_arg)
        if char_replacements is None:
            char_replacements = self.character_replacements

        if is_url(candidate_document):
            candidate_document = download_file_from_url(candidate_document)
        doc = fitz.open(candidate_document)
        # if expected_text_list is a string, convert it to a list
        if isinstance(expected_text_list, str):
            expected_text_list = [expected_text_list]
        missing_text_list = []
        found_text_list = []
        for text_item in expected_text_list:
            text_item_found = False
            for page in doc.pages():
                page_text = page.get_text("text")
                # Apply character replacements if configured
                if char_replacements:
                    page_text = apply_character_replacements(page_text, char_replacements)
                if any(text_item in s for s in page_text.splitlines()):
                    text_item_found = True
                    found_text_list.append({'text':text_item, 'document':candidate_document, 'page':page.number+1})
                    continue
            if text_item_found == False:
                missing_text_list.append({'text':text_item, 'document':candidate_document})
        if found_text_list:
            print(f"Missing Texts:\n{missing_text_list}")
            print(f"Found Texts:\n{found_text_list}")
            doc = None
            raise AssertionError('Some non-expected texts were found in document')
        else:
            doc = None
            print('None of the non-expected texts were found in document')
            print(f"Missing Texts:\n{missing_text_list}")
    
    @keyword
    def compare_pdf_documents_with_llm(self, *args, llm_override: bool = False, **kwargs):
        """Compare PDF files and consult the configured LLM before failing.

        | =Arguments= | =Description= |
        | ``*args`` | Positional arguments forwarded to ``Compare Pdf Documents`` (reference and candidate). |
        | ``llm_override`` | When ``${True}``, allow an approving LLM verdict to override baseline failures. Default ``${False}``. |
        | ``**kwargs`` | Keyword arguments forwarded to ``Compare Pdf Documents``. ``llm_prompt`` / ``llm_pdf_prompt`` may be used to customise the LLM prompt. |

        Example:
        | `Compare Pdf Documents With LLM`   reference.pdf   candidate.pdf   compare=text   llm_override=${True}
        """
        kwargs["llm_enabled"] = kwargs.get("llm_enabled", True)
        if "llm_override" not in kwargs:
            kwargs["llm_override"] = llm_override
        return self.compare_pdf_documents(*args, **kwargs)

    def _resolve_mask_arguments(
        self,
        mask_value: Optional[Any],
        text_mask_patterns: Optional[Any],
    ) -> Tuple[Optional[str], Optional[Any], List[str], List[str]]:
        mask_file: Optional[str] = None
        mask_payload: Optional[Any] = None
        warnings: List[str] = []

        if mask_value is not None:
            if isinstance(mask_value, (list, dict)):
                mask_payload = mask_value
            elif isinstance(mask_value, str):
                candidate_path = Path(mask_value).expanduser()
                if candidate_path.exists():
                    mask_file = str(candidate_path)
                else:
                    mask_payload = mask_value
            else:
                warnings.append(
                    f"Unsupported mask argument type '{type(mask_value).__name__}'. Expect string, dict or list."
                )

        pattern_values: List[str] = []
        if text_mask_patterns is not None:
            if isinstance(text_mask_patterns, str):
                pattern_values = [text_mask_patterns]
            elif isinstance(text_mask_patterns, (list, tuple, set)):
                pattern_values = [str(value) for value in text_mask_patterns if value is not None]
            else:
                warnings.append(
                    f"Unsupported text_mask_patterns type '{type(text_mask_patterns).__name__}'."
                )

        return mask_file, mask_payload, pattern_values, warnings

    def _compile_text_mask_patterns(
        self,
        pattern_values: List[str],
    ) -> Tuple[List[Pattern[str]], List[str]]:
        compiled: List[Pattern[str]] = []
        warnings: List[str] = []
        for pattern in pattern_values:
            try:
                compiled.append(re.compile(pattern))
            except re.error as exc:
                warnings.append(f"Invalid text mask regex '{pattern}': {exc}")
        return compiled, warnings

    def _build_document_snapshot(
        self,
        representation: DocumentRepresentation,
        *,
        extraction_config: StructureExtractionConfig,
        text_mask_patterns: List[Pattern[str]],
    ) -> Dict[str, Any]:
        pages: List[Dict[str, Any]] = []
        for page in representation.iter_pages(release=False):
            pages.append(
                self._build_page_snapshot(
                    page,
                    extraction_config=extraction_config,
                    text_mask_patterns=text_mask_patterns,
                )
            )
        return {
            "page_count": representation.page_count,
            "metadata": representation.metadata or {},
            "sigflags": representation.sigflags,
            "pages": pages,
        }

    def _build_page_snapshot(
        self,
        page,
        *,
        extraction_config: StructureExtractionConfig,
        text_mask_patterns: List[Pattern[str]],
    ) -> Dict[str, Any]:
        return {
            "number": page.page_number,
            "rotation": page.rotation,
            "mediabox": page.mediabox,
            "fonts": self._normalise_pdf_sequence(page.fonts),
            "images": self._normalise_pdf_sequence(page.images),
            "signatures": self._normalise_pdf_sequence(page.signatures),
            "text": self._extract_page_text_lines(
                page,
                extraction_config=extraction_config,
                text_mask_patterns=text_mask_patterns,
            ),
        }

    def _normalise_pdf_sequence(self, payload: Optional[Iterable[Any]]) -> List[Any]:
        normalised: List[Any] = []
        if not payload:
            return normalised
        for item in payload:
            if isinstance(item, (list, tuple)):
                normalised.append(tuple(item))
            else:
                normalised.append(item)
        return normalised

    def _extract_page_text_lines(
        self,
        page,
        *,
        extraction_config: StructureExtractionConfig,
        text_mask_patterns: List[Pattern[str]],
    ) -> List[str]:
        structure = page.get_pdf_structure(config=extraction_config)
        lines: List[str] = []
        for block in structure.blocks:
            for line in block.lines:
                text = line.text or ""
                if not text:
                    continue
                if text_mask_patterns and self._text_matches_any(text, text_mask_patterns):
                    continue
                lines.append(text)
        return lines

    @staticmethod
    def _text_matches_any(text: str, patterns: List[Pattern[str]]) -> bool:
        return any(pattern.search(text) for pattern in patterns)

    def _perform_structure_comparison(
        self,
        reference_document: str,
        candidate_document: str,
        tolerance: StructureTolerance,
        extraction_config: StructureExtractionConfig,
        *,
        case_sensitive: bool,
        dpi: int,
        reference_representation: Optional[DocumentRepresentation] = None,
        candidate_representation: Optional[DocumentRepresentation] = None,
        text_mask_patterns: Optional[List[Pattern[str]]] = None,
        ignore_page_boundaries: bool = False,
        check_geometry: bool = True,
        check_block_count: bool = True,
    ):
        release_reference = False
        release_candidate = False
        if reference_representation is None:
            reference_representation = DocumentRepresentation(reference_document, dpi=dpi)
            release_reference = True
        if candidate_representation is None:
            candidate_representation = DocumentRepresentation(candidate_document, dpi=dpi)
            release_candidate = True

        try:
            reference_structure = reference_representation.get_pdf_structure(config=extraction_config)
            candidate_structure = candidate_representation.get_pdf_structure(config=extraction_config)

            if text_mask_patterns:
                reference_structure = self._prune_structure_lines(reference_structure, text_mask_patterns)
                candidate_structure = self._prune_structure_lines(candidate_structure, text_mask_patterns)

            if ignore_page_boundaries:
                # Use text-only comparison that ignores page boundaries
                result = compare_document_text_only(
                    reference=reference_structure,
                    candidate=candidate_structure,
                    case_sensitive=case_sensitive,
                )
            else:
                # Use standard page-by-page comparison
                result = compare_document_structures(
                    reference=reference_structure,
                    candidate=candidate_structure,
                    tolerance=tolerance,
                    case_sensitive=case_sensitive,
                    check_geometry=check_geometry,
                    check_block_count=check_block_count,
                )
            self._log_structure_result(result, ignore_page_boundaries=ignore_page_boundaries)
            return result
        finally:
            if release_reference:
                reference_representation.close()
            if release_candidate:
                candidate_representation.close()

    def _prune_structure_lines(
        self,
        structure: DocumentStructure,
        patterns: List[Pattern[str]],
    ) -> DocumentStructure:
        if not patterns:
            return structure

        filtered_pages: List[PageStructure] = []
        for page in structure.pages:
            new_blocks: List[TextBlock] = []
            next_index = 0
            for block in page.blocks:
                new_lines: List[TextLine] = []
                for line in block.lines:
                    text = line.text or ""
                    if self._text_matches_any(text, patterns):
                        continue
                    new_lines.append(
                        TextLine(
                            index=next_index,
                            text=text,
                            bbox=line.bbox,
                            fonts=set(line.fonts),
                            spans=list(line.spans),
                        )
                    )
                    next_index += 1
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

    def _log_structure_result(self, result, ignore_page_boundaries: bool = False):
        """Log comparison results with single summary WARN and detail INFO messages.

        Robot Framework displays WARN messages at the top of log.html. To avoid
        cluttering that section, we emit a single summary warning and log all
        individual differences as INFO (visible only within keyword output).
        """
        if result.passed:
            logger.info("[PDF Structure] Documents match within configured tolerances.")
            return

        # Count total differences
        diff_count = result.difference_count()
        mode = "text-only (ignoring page boundaries)" if ignore_page_boundaries else "structure"

        # Single summary warning (appears at top of log.html)
        logger.warn(f"[PDF Structure] Comparison failed: {diff_count} difference(s) found in {mode} comparison.")

        # Log summary entries as INFO
        if result.summary:
            for entry in result.summary:
                logger.info(f"[PDF Structure] {entry}")

        # Log page differences as INFO
        if result.page_differences:
            for page in sorted(result.page_differences.keys()):
                for diff in result.page_differences[page]:
                    message = f"[PDF Structure] Page {page}: {diff.message}"
                    details = []
                    if diff.reference_index is not None:
                        details.append(f"reference line={diff.reference_index}")
                    if diff.candidate_index is not None:
                        details.append(f"candidate line={diff.candidate_index}")
                    if details:
                        message = f"{message} ({', '.join(details)})"
                    logger.info(message)
                    if diff.deltas:
                        pretty = ", ".join(f"{axis}={value:.3f}" for axis, value in diff.deltas.items())
                        logger.debug(f"[PDF Structure] Page {page} deltas: {pretty}")

        # Log document-level differences as INFO (for text-only mode)
        if result.document_differences:
            for diff in result.document_differences:
                message = f"[PDF Text] {diff.message}"
                details = []
                if diff.ref_index is not None:
                    details.append(f"reference position={diff.ref_index}")
                if diff.cand_index is not None:
                    details.append(f"candidate position={diff.cand_index}")
                if details:
                    message = f"{message} ({', '.join(details)})"
                logger.info(message)

    def _ensure_local_document(self, document):
        return download_file_from_url(document) if is_url(document) else document

    def _handle_llm_for_pdf_differences(
        self,
        reference_document: str,
        candidate_document: str,
        differences: List[Dict[str, Any]],
        overrides: Dict[str, Optional[str]],
        notes: List[str],
        custom_prompt: Optional[str],
        create_binary_content_fn: Optional[Any] = None,
        assess_pdf_diff_fn: Optional[Any] = None,
    ) -> Optional[LLMDecision]:
        overrides = {key: (str(value) if value is not None else None) for key, value in overrides.items()}
        overrides.setdefault("llm_enabled", "true")
        overrides.setdefault("llm_pdf_enabled", "true")

        try:
            settings = load_llm_settings(overrides)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warn(f"Failed to load LLM settings: {exc}")
            return None

        if not settings.pdf_enabled:
            print("LLM PDF evaluation requested but disabled via settings.")
            return None

        if create_binary_content_fn is None or assess_pdf_diff_fn is None:
            try:
                default_assess, default_create, _ = _load_pdf_llm_runtime()
            except LLMDependencyError as exc:
                print(str(exc))
                return None
            if assess_pdf_diff_fn is None:
                assess_pdf_diff_fn = default_assess
            if create_binary_content_fn is None:
                create_binary_content_fn = default_create

        summary_lines = [
            f"Reference document: {reference_document}",
            f"Candidate document: {candidate_document}",
        ]
        attachments = []
        try:
            reference_bytes = Path(reference_document).read_bytes()
            attachments.append(
                create_binary_content_fn(reference_bytes, "application/pdf")
            )
        except FileNotFoundError:
            logger.warn("Reference document not found for LLM attachment: %s", reference_document)
        except LLMDependencyError as exc:
            print(str(exc))
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warn("Failed to attach reference document for LLM: %s", exc)

        try:
            candidate_bytes = Path(candidate_document).read_bytes()
            attachments.append(
                create_binary_content_fn(candidate_bytes, "application/pdf")
            )
        except FileNotFoundError:
            logger.warn("Candidate document not found for LLM attachment: %s", candidate_document)
        except LLMDependencyError as exc:
            print(str(exc))
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warn("Failed to attach candidate document for LLM: %s", exc)

        for item in differences:
            facet = item.get("facet", "unknown")
            description = item.get("description", "")
            detail = item.get("details")
            summary_lines.append(f"{facet}: {description}")
            if detail:
                detail_text = str(detail)
                if len(detail_text) > 2000:
                    detail_text = f"{detail_text[:2000]} ... (truncated)"
                summary_lines.append(detail_text)

        extra_messages = list(notes or [])

        try:
            decision = assess_pdf_diff_fn(
                settings=settings,
                textual_summary="\n".join(summary_lines),
                attachments=tuple(attachments),
                extra_messages=extra_messages,
                system_prompt=custom_prompt,
            )
        except LLMDependencyError as exc:
            print(str(exc))
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warn("LLM PDF evaluation failed: %s", exc)
            return None
        return decision

def is_masked(text, mask):
    if isinstance(mask, str):
        mask = [mask]
    if isinstance(mask, list):
        for single_mask in mask:
            if re.match(single_mask, text):
                return True
    return  False



