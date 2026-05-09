from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: str
    client_id: str
    name: str
    file_type: str
    status: str
    checksum: Optional[str] = None
    document_family: Optional[str] = None
    ingestion_job_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    id: str
    client_id: str
    name: str
    file_type: str
    status: str
    document_family: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    client_id: str
    question: str
    session_id: Optional[str] = None
    reasoning_effort: Literal["low", "medium", "high"] = "medium"


class CitationDetail(BaseModel):
    rank: int
    score: Optional[float] = None
    text: str
    document_name: str
    source_file: Optional[str] = None
    page_num: Optional[int] = None
    slide_num: Optional[int] = None
    section_title: Optional[str] = None
    chunk_type: str = "text"
    source_artifact_type: Optional[str] = None
    source_artifact_id: Optional[str] = None
    artifact_bundle_path: Optional[str] = None
    version_label: Optional[str] = None
    version_group: Optional[str] = None
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    citation_label: Optional[str] = None
    authority_score: Optional[float] = None
    asset_refs: List[str] = []
    has_image_assets: bool = False
    figure_type: Optional[str] = None
    chart_type: Optional[str] = None
    table_id: Optional[str] = None


class ConflictDetail(BaseModel):
    conflict_type: str
    summary: Optional[str] = None
    supporting_chunks: Optional[List[Dict[str, Any]]] = None


class QueryResponse(BaseModel):
    answer: str
    reasoning: str = ""
    citations: List[Dict[str, Any]] = []
    conflicts: List[Dict[str, Any]] = []
    query_id: Optional[str] = None
    latency_ms: Optional[int] = None
    source_count: int = 0
    evidence_count: int = 0
    images_used: List[str] = []
    image_evidence_count: int = 0
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    reasoning_effort_applied: bool = False
    session_id: Optional[str] = None
    user_message_id: Optional[str] = None
    assistant_message_id: Optional[str] = None
