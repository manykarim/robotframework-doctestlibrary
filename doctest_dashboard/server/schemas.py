"""Request models for the API."""

from typing import List, Optional

from pydantic import BaseModel


class IngestRequest(BaseModel):
    output_xml: str


class RunLabelRequest(BaseModel):
    label: Optional[str] = None


class DecisionRequest(BaseModel):
    actor: Optional[str] = None
    reason: Optional[str] = None


class BatchDecisionRequest(BaseModel):
    ids: List[int]
    actor: Optional[str] = None
    reason: Optional[str] = None


class MaskSaveRequest(BaseModel):
    file: str
    masks: object


class MaskPreviewRequest(BaseModel):
    file: str
    page: int = 1
    masks: object = None
    dpi: Optional[int] = None
    ocr_engine: Optional[str] = None
    force_ocr: bool = False


class RegionBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class RegionTextRequest(BaseModel):
    page_no: int = 1
    region: RegionBox
    force_ocr: bool = False


class RecompareRequest(BaseModel):
    comparison_id: int
    masks: object = None
    settings: Optional[dict] = None


class RecompareBatchRequest(BaseModel):
    masks: object = None
    comparison_ids: Optional[list] = None
    masks_file: Optional[str] = None
    settings: Optional[dict] = None
