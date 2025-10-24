from __future__ import annotations

import re
import textwrap
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
from robot.api.deco import keyword, library

from DocTest.DocumentRepresentation import DocumentRepresentation
from DocTest.Downloader import download_file_from_url, is_url
from DocTest.config import DEFAULT_DPI
from DocTest.llm import load_llm_settings
from DocTest.llm.client import create_binary_content, run_structured_prompt
from DocTest.llm.types import LLMDecision

from .prompts import (
    AREA_EXTRACTION_PROMPT,
    CHAT_SYSTEM_PROMPT,
    COUNT_PROMPT,
    OBJECT_DETECTION_PROMPT,
    TEXT_EXTRACTION_PROMPT,
)
from .renderers import load_document_pages, prepare_image_attachments
from .responses import LLMChatResponse, LLMCountResponse, LLMExtractionResult


def _coerce_override(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value if item is not None)
    return str(value)


def _collect_overrides(**kwargs) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    for key, value in kwargs.items():
        if value is not None:
            mapping[key] = _coerce_override(value)
    return mapping


def _ensure_local(path_or_url: str) -> str:
    if is_url(path_or_url):
        return download_file_from_url(path_or_url)
    return path_or_url


def _summaries_from_text(pages: Iterable[Dict], char_limit: int = 1200) -> List[str]:
    summaries: List[str] = []
    for page in pages:
        text = (page.get("text") or "").strip()
        if not text:
            continue
        snippet = textwrap.shorten(text, width=char_limit, placeholder=" ...")
        summaries.append(f"Page {page['number']} text:\n{snippet}")
    return summaries


@library(scope="GLOBAL", doc_format="ROBOT")
class AIKeywords:
    """Optional LLM-assisted keywords for document analysis. Install with `[ai]` extra."""

    def __init__(self, dpi: int = DEFAULT_DPI):
        self.dpi = dpi

    # ---------------- Text Extraction ---------------- #

    @keyword
    def get_text_with_llm(
        self,
        document: str,
        prompt: Optional[str] = None,
        include_pdf_text: bool = True,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        max_pages: Optional[int] = None,
    ) -> str:
        """Extract textual content from a document using an LLM.

        Args:
            document: Path or URL to the source image/PDF.
            prompt: Optional custom instructions for the LLM.
            include_pdf_text: If ``${True}``, include direct PDF text in the prompt.
            model / provider / temperature / max_output_tokens / request_timeout: Override LLM settings.
            max_pages: Limit how many pages are sent to the model.

        Returns:
            The extracted text as a string.

        Raises:
            AssertionError: When the document cannot be read or produces no pages.

        Example:
        | ${text}=    Get Text With LLM    testdata/invoice.pdf    prompt=Transcribe all text exactly
        | Should Contain    ${text}    INVOICE
        """

        overrides = _collect_overrides(
            llm_enabled="true",
            llm_models=model,
            llm_provider=provider,
            llm_temperature=temperature,
            llm_max_output_tokens=max_output_tokens,
            llm_request_timeout=request_timeout,
        )

        settings = load_llm_settings(overrides)
        _, pages = _load_pages_or_fail(document, self.dpi, max_pages=max_pages)

        attachments = [
            create_binary_content(payload, "image/png")
            for _, payload in prepare_image_attachments(pages)
        ]
        extra = _summaries_from_text(pages) if include_pdf_text else []

        base_prompt = prompt.strip() if prompt else TEXT_EXTRACTION_PROMPT
        result = run_structured_prompt(
            settings=settings,
            purpose="vision",
            prompt=base_prompt,
            attachments=attachments,
            extra_messages=extra,
            output_type=LLMExtractionResult,
        )
        if not isinstance(result, LLMExtractionResult):
            raise AssertionError("LLM returned an unexpected response for text extraction.")
        return result.text

    @keyword
    def get_text_from_area_with_llm(
        self,
        document: str,
        area: Dict,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> str:
        """Extract text from a specified area using an LLM.

        Args:
            document: Path or URL to the source image/PDF.
            area: Mapping describing the region (keys ``x``, ``y``, ``width``, ``height``, optional ``page``/``unit``).
            prompt: Optional custom instructions for the LLM.
            model/provider/temperature/request_timeout: Override LLM settings.

        Returns:
            The extracted text as a string.

        Raises:
            AssertionError: When the document/area cannot be processed.

        Example:
        | &{area}=    Create Dictionary    x=0    y=0    width=400    height=200    page=1
        | ${text}=    Get Text From Area With LLM    testdata/invoice.pdf    ${area}
        | Should Contain    ${text}    Invoice
        """

        overrides = _collect_overrides(
            llm_enabled="true",
            llm_models=model,
            llm_provider=provider,
            llm_temperature=temperature,
            llm_request_timeout=request_timeout,
        )
        settings = load_llm_settings(overrides)

        doc_path, pages = _load_pages_or_fail(document, self.dpi)

        target_page = area.get("page", "all")
        unit = area.get("unit", "px")
        attachments: List = []
        area_text_snippets: List[str] = []

        doc_representation = DocumentRepresentation(doc_path, dpi=self.dpi)
        page_list = list(doc_representation.pages)

        for idx, (page_meta, page_obj) in enumerate(zip(pages, page_list), 1):
            if target_page != "all" and int(target_page) != idx:
                continue
            try:
                x, y, w, h = page_obj._convert_to_pixels(area, unit)
            except Exception as exc:
                raise AssertionError(f"Invalid area specification: {exc}") from exc
            crop = page_meta["image"][y : y + h, x : x + w]
            if crop.size == 0:
                continue
            attachments.append(create_binary_content(cv2.imencode(".png", crop)[1].tobytes(), "image/png"))
            snippet = page_obj._get_text_from_area(area)
            if snippet:
                area_text_snippets.append(f"Page {idx} area text:\n{snippet}")

        if not attachments:
            raise AssertionError("Specified area produced no image data.")

        base_prompt = prompt.strip() if prompt else AREA_EXTRACTION_PROMPT
        result = run_structured_prompt(
            settings=settings,
            purpose="vision",
            prompt=base_prompt,
            attachments=attachments,
            extra_messages=area_text_snippets or None,
            output_type=LLMExtractionResult,
        )
        if not isinstance(result, LLMExtractionResult):
            raise AssertionError("LLM returned an unexpected response for area extraction.")
        return result.text

    # ---------------- Chat ---------------- #

    @keyword
    def chat_with_document(
        self,
        prompt: str,
        documents,
        include_pdf_text: bool = True,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        max_pages: Optional[int] = None,
    ) -> str:
        """Chat with one or more documents using an LLM.

        Args:
            prompt: User question or instruction.
            documents: Single path/URL or list of documents to include.
            include_pdf_text: Add extracted PDF text alongside rendered pages.
            model/provider/temperature/max_output_tokens/request_timeout/max_pages: Override LLM settings.

        Returns:
            The assistant's textual response.

        Raises:
            AssertionError: When documents cannot be read.

        Example:
        | ${answer}=    Chat With Document    prompt=Summarise the document total.    documents=testdata/invoice.pdf
        | Should Contain    ${answer}    EUR
        """

        if not documents:
            raise ValueError("No documents supplied.")
        if isinstance(documents, str):
            doc_list = [documents]
        else:
            doc_list = list(documents)

        overrides = _collect_overrides(
            llm_enabled="true",
            llm_models=model,
            llm_provider=provider,
            llm_temperature=temperature,
            llm_max_output_tokens=max_output_tokens,
            llm_request_timeout=request_timeout,
        )
        settings = load_llm_settings(overrides)

        attachments: List = []
        extra_messages: List[str] = [prompt]

        for doc in doc_list:
            _, pages = _load_pages_or_fail(doc, self.dpi, max_pages=max_pages)
            for _, payload in prepare_image_attachments(pages):
                attachments.append(create_binary_content(payload, "image/png"))
            if include_pdf_text:
                extra_messages.extend(_summaries_from_text(pages))

        result = run_structured_prompt(
            settings=settings,
            purpose="vision",
            prompt=CHAT_SYSTEM_PROMPT,
            attachments=attachments,
            extra_messages=extra_messages,
            output_type=LLMChatResponse,
        )
        if not isinstance(result, LLMChatResponse):
            raise AssertionError("LLM returned an unexpected chat response.")
        return result.response

    # ---------------- Detection & Counting ---------------- #

    @keyword
    def image_should_contain(
        self,
        document: str,
        expected: str,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        """Assert that a document contains the expected visual content.

        Args:
            document: Path or URL.
            expected: Description of objects/text that must appear.
            prompt/model/provider/temperature/request_timeout: Optional overrides.

        Raises:
            AssertionError: When the document cannot be read or the expected content is missing.

        Example:
        | Image Should Contain    testdata/invoice.pdf    Invoice header with company logo
        """

        overrides = _collect_overrides(
            llm_enabled="true",
            llm_models=model,
            llm_provider=provider,
            llm_temperature=temperature,
            llm_request_timeout=request_timeout,
        )
        settings = load_llm_settings(overrides)

        _, pages = _load_pages_or_fail(document, self.dpi)
        attachments = [
            create_binary_content(payload, "image/png")
            for _, payload in prepare_image_attachments(pages)
        ]
        base_prompt = prompt.strip() if prompt else OBJECT_DETECTION_PROMPT
        combined_prompt = f"{base_prompt}\nRequired objects: {expected}"

        decision = run_structured_prompt(
            settings=settings,
            purpose="vision",
            prompt=combined_prompt,
            attachments=attachments,
            output_type=LLMDecision,
        )
        if not isinstance(decision, LLMDecision):
            raise AssertionError(f"LLM returned an unexpected response while checking '{expected}'.")
        if decision.decision == decision.decision.APPROVE:
            return
        if decision.reason:
            print(f"LLM reason: {decision.reason}")
        raise AssertionError(f"Expected object '{expected}' not found in '{document}'.")

    @keyword
    def get_item_count_from_image(
        self,
        document: str,
        item_description: str,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> int:
        """Count items in an image or document using an LLM.

        Args:
            document: Path or URL.
            item_description: Description of the item to count.
            prompt/model/provider/temperature/request_timeout: Optional overrides.

        Returns:
            Integer count (0 when no items detected).

        Raises:
            AssertionError: When the document cannot be read.

        Example:
        | ${count}=    Get Item Count From Image    testdata/invoice.pdf    item_description=number of tables
        | Should Be True    ${count} >= 1
        """

        overrides = _collect_overrides(
            llm_enabled="true",
            llm_models=model,
            llm_provider=provider,
            llm_temperature=temperature,
            llm_request_timeout=request_timeout,
        )
        settings = load_llm_settings(overrides)

        _, pages = _load_pages_or_fail(document, self.dpi)
        attachments = [
            create_binary_content(payload, "image/png")
            for _, payload in prepare_image_attachments(pages)
        ]
        base_prompt = prompt.strip() if prompt else COUNT_PROMPT
        combined_prompt = f"{base_prompt}\nItem to count: {item_description}"

        result = run_structured_prompt(
            settings=settings,
            purpose="vision",
            prompt=combined_prompt,
            attachments=attachments,
            output_type=LLMCountResponse,
        )
        if not isinstance(result, LLMCountResponse):
            raise AssertionError("LLM returned an unexpected response for item count.")
        return int(result.count)
def _load_pages_or_fail(document: str, dpi: int, max_pages: Optional[int] = None):
    path = _ensure_local(document)
    try:
        pages = load_document_pages(path, dpi=dpi, max_pages=max_pages)
    except Exception as exc:
        raise AssertionError(f"Unable to load document '{document}': {exc}") from exc
    if not pages:
        raise AssertionError(f"Document '{document}' produced no renderable pages.")
    return path, pages
