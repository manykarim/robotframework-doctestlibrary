from inspect import signature
from pprint import pprint
from deepdiff import DeepDiff
from robot.api import logger
from robot.api.deco import keyword, library
import fitz
import re
from DocTest.config import DEFAULT_DPI
from DocTest.DocumentRepresentation import DocumentRepresentation
from DocTest.Downloader import is_url, download_file_from_url
from DocTest.PdfStructureComparator import StructureTolerance, compare_document_structures
from DocTest.PdfStructureModels import StructureExtractionConfig


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
    
    
    def __init__(self, **kwargs):
        pass
    
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
            - ``structure_round_precision`` (int or ``None``, default ``3``) – rounding precision for coordinates
            - ``structure_dpi`` (int, default: library DPI) – DPI used when rasterising PDFs prior to structure extraction

        Result passes only if every requested facet matches.

        Examples:
        | `Compare Pdf Documents`    reference.pdf    candidate.pdf
        | `Compare Pdf Documents`    reference.pdf    candidate.pdf    compare=text
        | `Compare Pdf Documents`    reference.pdf    candidate.pdf    compare=structure
        | `Compare Pdf Documents`    reference.pdf    candidate.pdf    compare=structure|metadata    structure_position_tolerance=5.0

        """
        mask = kwargs.pop('mask', None)
        check_pdf_text = _as_bool(kwargs.pop('check_pdf_text', False))

        compare_value = kwargs.pop('compare', "all")
        compare_value = compare_value.replace('|', ',')
        compare_tokens = [token.strip().lower() for token in compare_value.split(',') if token.strip()]
        if not compare_tokens:
            compare_tokens = ['all']
        compare_set = set(compare_tokens)

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
        structure_dpi = _as_int_or_none(kwargs.pop('structure_dpi', None))
        if structure_dpi is None:
            structure_dpi = _as_int_or_none(kwargs.get('dpi'))
        if structure_dpi is None:
            structure_dpi = DEFAULT_DPI

        extraction_config = StructureExtractionConfig(
            collapse_whitespace=structure_collapse_whitespace,
            strip_font_subset=structure_strip_font_subset,
            whitespace_replacement=str(structure_whitespace_replacement),
            strip_line_edges=structure_strip_line_edges,
            drop_empty_lines=structure_drop_empty_lines,
            round_precision=structure_round_precision,
        )

        structure_requested = 'structure' in compare_set

        reference_document = self._ensure_local_document(reference_document)
        candidate_document = self._ensure_local_document(candidate_document)
        ref_doc = fitz.open(reference_document)
        cand_doc = fitz.open(candidate_document)
        reference = {}
        reference['pages'] = []
        reference['metadata']=ref_doc.metadata
        reference['page_count']=ref_doc.page_count
        reference['sigflags']=ref_doc.get_sigflags()
        for i, page in enumerate(ref_doc.pages()):
            signature_list = []
            text = [x for x in page.get_text("text").splitlines() if not is_masked(x, mask)]
            for widget in page.widgets():
                if widget.is_signed:
                    signature_list.append(list((widget.field_name, widget.field_label, widget.field_value)))
            reference['pages'].append(dict([('number',i+1), ('fonts', page.get_fonts()), ('images', page.get_images()), ('rotation', page.rotation), ('mediabox', page.mediabox), ('signatures', signature_list),('text', text)]))


        candidate = {}
        candidate['pages'] = []
        candidate['metadata']=cand_doc.metadata
        candidate['page_count']=cand_doc.page_count
        candidate['sigflags']=cand_doc.get_sigflags()
        for i, page in enumerate(cand_doc.pages()):
            signature_list = []
            text = [x for x in page.get_text("text").splitlines() if not is_masked(x, mask)]
            for widget in page.widgets():
                if widget.is_signed:
                    signature_list.append(list((widget.field_name, widget.field_label, widget.field_value)))
            candidate['pages'].append(dict([('number',i+1), ('fonts', page.get_fonts()), ('images', page.get_images()), ('rotation', page.rotation), ('mediabox', page.mediabox), ('signatures', signature_list),('text', text)]))

        differences_detected=False

        if 'metadata' in compare_set or 'all' in compare_set:
            diff = DeepDiff(reference['metadata'], candidate['metadata'])
            if diff != {}:
                differences_detected=True
                print("Different metadata")
                pprint(diff, width=200)
        if 'signatures' in compare_set or 'all' in compare_set:
            diff = DeepDiff(reference['sigflags'], candidate['sigflags'])
            if diff != {}:
                differences_detected=True
                print("Different signature")
                pprint(diff, width=200)
        for ref_page, cand_page in zip(reference['pages'], candidate['pages']):
            diff = DeepDiff(ref_page['rotation'], cand_page['rotation'])
            if diff != {}:
                differences_detected=True
                print("Different rotation")
                pprint(diff, width=200)
            # diff = DeepDiff(ref_page['mediabox'], cand_page['mediabox'])
            # if diff != {}:
            #     differences_detected=True
            #     print("Different mediabox")
            #     pprint(diff, width=200)
            if 'text' in compare_set or 'all' in compare_set:
                diff = DeepDiff(ref_page['text'], cand_page['text'])
                if diff != {}:
                    differences_detected=True
                    print("Different text")
                    pprint(diff, width=200)
            if 'fonts' in compare_set or 'all' in compare_set:
                diff = DeepDiff(ref_page['fonts'], cand_page['fonts'])
                if diff != {}:
                    differences_detected=True
                    print("Different fonts")
                    pprint(diff, width=200)
            if 'images' in compare_set or 'all' in compare_set:
                diff = DeepDiff(ref_page['images'], cand_page['images'])
                if diff != {}:
                    differences_detected=True
                    print("Different images")
                    pprint(diff, width=200)
            if 'signatures' in compare_set or 'all' in compare_set:
                diff = DeepDiff(ref_page['signatures'], cand_page['signatures'])
                if diff != {}:
                    differences_detected=True
                    print("Different signatures")
                    pprint(diff, width=200)

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
                extraction_config=extraction_config,
                case_sensitive=structure_case_sensitive,
                dpi=structure_dpi,
            )
            if not structure_result.passed:
                differences_detected = True

        if differences_detected:
            ref_doc = None
            cand_doc = None             
            raise AssertionError('The compared PDF Document Data is different.')
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

        Examples:
        | `Compare Pdf Structure`    reference.pdf    candidate.pdf
        | `Compare Pdf Structure`    reference.pdf    candidate.pdf    position_tolerance=3    relative_tolerance=0.02
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

        extraction_config = StructureExtractionConfig(
            collapse_whitespace=collapse_whitespace,
            strip_font_subset=strip_font_subset,
            whitespace_replacement=str(whitespace_replacement),
            strip_line_edges=strip_line_edges,
            drop_empty_lines=drop_empty_lines,
            round_precision=round_precision,
        )
        tolerance = StructureTolerance(
            position=position_tolerance,
            size=size_tolerance,
            relative=relative_tolerance,
        )
        result = self._perform_structure_comparison(
            reference_document=reference_document,
            candidate_document=candidate_document,
            tolerance=tolerance,
            extraction_config=extraction_config,
            case_sensitive=case_sensitive,
            dpi=dpi,
        )
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
    def PDF_should_contain_strings(self, expected_text_list, candidate_document):
        """Checks if each item provided in the list ``expected_text_list`` appears in the PDF File ``candidate_document``.
        
        ``expected_text_list`` is a list of strings or a single string, ``candidate_document`` is the path or URL to a PDF File.
        
        Examples:

        | @{strings}=    Create List    One String    Another String   
        | `PDF Should Contain Strings`    ${strings}    candidate.pdf   
        | `PDF Should Contain Strings`    One String    candidate.pdf   
        
        """
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
                if any(text_item in s for s in page.get_text("text").splitlines()):
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
    def PDF_should_not_contain_strings(self, expected_text_list, candidate_document):
        """Checks if each item provided in the list ``expected_text_list`` does NOT appear in the PDF File ``candidate_document``.
        
        ``expected_text_list`` is a list of strings or a single string, ``candidate_document`` is the path or URL to a PDF File.
        
        Examples:

        | @{strings}=    Create List    One String    Another String   
        | `PDF Should Not Contain Strings`    ${strings}    candidate.pdf   
        | `PDF Should Not Contain Strings`    One String    candidate.pdf   
        
        """
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
                if any(text_item in s for s in page.get_text("text").splitlines()):
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
    
    def _perform_structure_comparison(
        self,
        reference_document: str,
        candidate_document: str,
        tolerance: StructureTolerance,
        extraction_config: StructureExtractionConfig,
        *,
        case_sensitive: bool,
        dpi: int,
    ):
        reference_representation = DocumentRepresentation(reference_document, dpi=dpi)
        candidate_representation = DocumentRepresentation(candidate_document, dpi=dpi)
        reference_structure = reference_representation.get_pdf_structure(config=extraction_config)
        candidate_structure = candidate_representation.get_pdf_structure(config=extraction_config)
        result = compare_document_structures(
            reference=reference_structure,
            candidate=candidate_structure,
            tolerance=tolerance,
            case_sensitive=case_sensitive,
        )
        self._log_structure_result(result)
        return result

    def _log_structure_result(self, result):
        if result.summary:
            for entry in result.summary:
                logger.warn(f"[PDF Structure] {entry}")
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
                    logger.warn(message)
                    if diff.deltas:
                        pretty = ", ".join(f"{axis}={value:.3f}" for axis, value in diff.deltas.items())
                        logger.debug(f"[PDF Structure] Page {page} deltas: {pretty}")
        if result.passed:
            logger.info("[PDF Structure] Documents match within configured tolerances.")

    def _ensure_local_document(self, document):
        return download_file_from_url(document) if is_url(document) else document

def is_masked(text, mask):
    if isinstance(mask, str):
        mask = [mask]
    if isinstance(mask, list):
        for single_mask in mask:
            if re.match(single_mask, text):
                return True
    return  False



