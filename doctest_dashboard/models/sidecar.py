"""Pydantic models for the comparison sidecar, schema v1.

This is the data contract between the core library's ``ResultWriter`` and
the dashboard. The contract test in ``tests/test_sidecar_contract.py`` runs
the real library and validates its output against these models — change
either side only together.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

SUPPORTED_SCHEMA_VERSION = 1


class DiffRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int
    y: int
    width: int
    height: int


class DocumentRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    pages: Optional[int] = None
    dpi: Optional[int] = None


class PageResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    page: Union[int, str]
    status: Literal["PASS", "FAIL"]
    score: Optional[float] = None
    threshold: Optional[float] = None
    diff_regions: List[DiffRegion] = Field(default_factory=list)
    images: Dict[str, str] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
    resolved_masks: List[DiffRegion] = Field(default_factory=list)
    regions_text: List[Dict[str, Any]] = Field(default_factory=list)


class Timing(BaseModel):
    model_config = ConfigDict(extra="allow")

    started: str
    elapsed_ms: int


class ComparisonResult(BaseModel):
    """One comparison sidecar (``doctest_results/{uuid}.json``)."""

    model_config = ConfigDict(extra="allow")

    schema_version: int
    keyword: str
    library: str
    name: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    status: Literal["PASS", "FAIL"]
    reference: DocumentRef
    candidate: DocumentRef
    settings: Dict[str, Any] = Field(default_factory=dict)
    masks: Dict[str, Any] = Field(default_factory=dict)
    pages: List[PageResult] = Field(default_factory=list)
    llm: Optional[Dict[str, Any]] = None
    notes: List[str] = Field(default_factory=list)
    facets: List[Dict[str, Any]] = Field(default_factory=list)
    timing: Timing


def parse_sidecar(data: Dict[str, Any]) -> ComparisonResult:
    """Validate sidecar data, rejecting unknown schema majors explicitly."""
    version = data.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported sidecar schema_version {version!r}; "
            f"this dashboard supports version {SUPPORTED_SCHEMA_VERSION}"
        )
    return ComparisonResult.model_validate(data)
