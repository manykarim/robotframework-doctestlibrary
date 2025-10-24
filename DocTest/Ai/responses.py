from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class LLMExtractionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    page_summaries: Optional[List[str]] = None
    metadata: Optional[Dict[str, str]] = None


class LLMChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    response: str
    citations: Optional[List[str]] = None
    raw: Optional[str] = None


class LLMCountResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item: str
    count: int
    confidence: Optional[float] = None
    explanation: Optional[str] = None
