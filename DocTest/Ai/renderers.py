from __future__ import annotations

import cv2
from typing import Dict, Iterable, List, Optional, Tuple

from DocTest.DocumentRepresentation import DocumentRepresentation
from DocTest.config import DEFAULT_DPI


def _encode_image(image) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image to PNG.")
    return encoded.tobytes()


def load_document_pages(
    document_path: str,
    dpi: int = DEFAULT_DPI,
    max_pages: Optional[int] = None,
) -> List[Dict]:
    doc = DocumentRepresentation(document_path, dpi=dpi)
    pages: List[Dict] = []
    for idx, page in enumerate(doc.pages, 1):
        if max_pages and idx > max_pages:
            break
        text_content = ""
        if getattr(page, "pdf_text_data", None):
            text_content = page.pdf_text_data
        elif getattr(page, "text", None):
            text_content = page.text
        else:
            try:
                text_content = page._get_text()
            except Exception:
                text_content = ""

        pages.append(
            {
                "number": idx,
                "image": page.image,
                "text": text_content or "",
                "dpi": page.dpi,
            }
        )
    return pages


def prepare_image_attachments(pages: Iterable[Dict]) -> List[Tuple[int, bytes]]:
    attachments: List[Tuple[int, bytes]] = []
    for page in pages:
        attachments.append((page["number"], _encode_image(page["image"])))
    return attachments
