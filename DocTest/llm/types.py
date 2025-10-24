from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class LLMDecisionLabel(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    FLAG = "flag"


class LLMDecision(BaseModel):
    """Validated response returned by LLM assessments."""

    model_config = ConfigDict(extra="ignore")

    decision: LLMDecisionLabel
    confidence: Optional[float] = None
    reason: Optional[str] = None
    notes: Optional[str] = None

    @property
    def is_positive(self) -> bool:
        return self.decision == LLMDecisionLabel.APPROVE
