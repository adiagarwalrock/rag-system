from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Annotated, Any

from pydantic import BaseModel, Field

# -------------------------------------------------------------------------------------
#
# DATA MODELS
#
# -------------------------------------------------------------------------------------


ZONE_ORDER = {
    "header_zone": 0,
    "main_body": 1,
    "left_column": 2,
    "right_column": 3,
    "sidebar": 4,
    "table_zone": 5,
    "figure_zone": 6,
    "footnote_zone": 7,
    "footer_zone": 8,
}


@dataclass(slots=True)
class PageManifest:
    document_id: str
    page_num: int
    page_width: float
    page_height: float
    screenshot_path: str
    full_page_text: str
    layout_confidence: float
    complexity_score: float
    page_class: str
    ocr_used: bool
    parser_sources: list[str]
    layout_engine: str = "pymupdf_native"
    page_parse_degraded: bool = False
    degraded_stages: list[str] = field(default_factory=list)
    pdf_repair_attempted: bool = False
    pdf_repair_method: str | None = None
    pdf_repair_success: bool = False
    pdf_repair_error: str | None = None
    llm_page_summary: str | None = None
    llm_enriched: bool = False
    llm_page_summary_status: str = "not_run"
    llm_page_summary_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Region:
    region_id: str
    page_num: int
    bbox: list[float]
    region_type: str
    text: str
    reading_order: int
    section_path: str
    parent_region_id: str | None
    adjacent_region_ids: list[str]
    asset_refs: list[str]
    source_parser: str
    confidence: float
    zone: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TableArtifact:
    table_id: str
    page_nums: list[int]
    bbox_list: list[list[float]]
    caption_text: str
    section_path: str
    html_table: str
    json_table: list[list[str]]
    normalized_table_text: str
    header_rows: list[str]
    units: list[str]
    footnotes: list[str]
    continuation_flag: bool
    confidence: float
    llm_enriched: bool = False
    llm_enrichment_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FigureArtifact:
    figure_id: str
    page_num: int
    bbox: list[float]
    figure_type: str
    caption_text: str
    nearby_text: str
    section_path: str
    crop_path: str
    page_screenshot_path: str
    visual_proxy_text: str
    footnotes: list[str]
    confidence: float
    chart_type: str = "unknown"
    chart_title: str = ""
    x_axis_label: str = ""
    y_axis_label: str = ""
    x_categories: list[str] = field(default_factory=list)
    series: list[str] = field(default_factory=list)
    legend_items: list[str] = field(default_factory=list)
    approx_datapoints: list[dict[str, Any]] = field(default_factory=list)
    trend_summary: str = ""
    key_chart_facts: list[str] = field(default_factory=list)
    numeric_extraction_confidence: float | None = None
    chart_parse_status: str = "not_applicable"
    # Exhaustive segment/cell extraction for specific visual types
    pie_segments: list[dict[str, Any]] = field(default_factory=list)
    matrix_cells: list[dict[str, Any]] = field(default_factory=list)
    stat_box_values: list[str] = field(default_factory=list)
    llm_enriched: bool = False
    llm_enrichment_confidence: float | None = None
    llm_caption_model: str | None = None
    llm_caption_version: str | None = None
    llm_caption_prompt_version: str | None = None
    llm_caption_status: str = "skipped"
    llm_caption_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReasoningArtifact:
    reasoning_id: str
    reasoning_type: (
        str  # table_reasoning, chart_reasoning, figure_reasoning, page_reasoning
    )
    page_nums: list[int]
    source_artifact_ids: list[str]
    text: str
    claims: list[str]
    evidence_refs: dict[str, Any]  # bbox refs, region ids, caption refs
    confidence: float
    model: str | None = None
    prompt_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChunkArtifact:
    chunk_id: str
    chunk_type: str
    source_artifact_type: str
    source_artifact_id: str
    page_nums: list[int]
    text: str
    metadata: dict[str, Any]
    asset_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -------------------------------------------------------------------------------------
#
# RESPONSE MODELS
#
# -------------------------------------------------------------------------------------


class ChartDatapointResponse(BaseModel):
    series: str = ""
    x: str = ""
    y: float
    unit: str = ""
    approximate: bool = True


class PieSegmentResponse(BaseModel):
    """One slice of a pie or donut chart."""
    label: str = ""
    value: float = 0.0
    unit: str = "%"


class MatrixCellResponse(BaseModel):
    """One cell from a comparison matrix or cross-tab grid."""
    row_header: str = ""
    col_header: str = ""
    value: str = ""


class ChartCaptionResponse(BaseModel):
    chart_type: str = ""
    chart_title: str = ""
    x_axis_label: str = ""
    y_axis_label: str = ""
    x_categories: list[str] = Field(default_factory=list)
    series: list[str] = Field(default_factory=list)
    legend_items: list[str] = Field(default_factory=list)
    approx_datapoints: list[ChartDatapointResponse] = Field(default_factory=list)
    trend_summary: str = ""
    key_chart_facts: list[str] = Field(default_factory=list)
    numeric_extraction_confidence: float | None = None
    # Exhaustive segment extraction for pie / donut charts
    pie_segments: list[PieSegmentResponse] = Field(default_factory=list)
    # Cell-level extraction for comparison matrices and grids
    matrix_cells: list[MatrixCellResponse] = Field(default_factory=list)
    # Labeled values from quick-facts boxes / infographic stat tiles
    stat_box_values: list[str] = Field(default_factory=list)


class ArtifactEnrichmentResponse(BaseModel):
    summary_points: list[str] = Field(default_factory=list)


class PageScreenshotResponse(BaseModel):
    layout_description: str = ""
    # Each entry is a "Label: value" pair (e.g. "Properties: 179", "Dividend Yield: 4.7%")
    labeled_values: list[str] = Field(default_factory=list)
    numeric_values: list[str] = Field(default_factory=list)
    chart_descriptions: list[str] = Field(default_factory=list)
    table_summaries: list[str] = Field(default_factory=list)
    map_or_diagram_annotations: list[str] = Field(default_factory=list)
    key_takeaways: list[str] = Field(default_factory=list)
    # Exhaustive pie slice listing: "label: value%"
    pie_chart_segments: list[str] = Field(default_factory=list)
    # Matrix cell listing: "row | col | value"
    matrix_cell_values: list[str] = Field(default_factory=list)


ReasoningClaimText = Annotated[str, Field(min_length=1, max_length=180)]
ReasoningEvidenceRefText = Annotated[str, Field(min_length=1, max_length=120)]


class ReasoningStructuredResponse(BaseModel):
    key_insights: list[ReasoningClaimText] = Field(default_factory=list, max_length=5)
    metric_comparisons: list[ReasoningClaimText] = Field(
        default_factory=list,
        max_length=4,
    )
    trend_statement: Annotated[str, Field(max_length=180)] = ""
    caveats: list[ReasoningClaimText] = Field(default_factory=list, max_length=3)
    evidence_refs: list[ReasoningEvidenceRefText] = Field(
        default_factory=list,
        max_length=8,
    )


@dataclass(frozen=True, slots=True)
class ReasoningInferenceResult:
    payload: dict[str, Any]
    used_structured_output: bool
