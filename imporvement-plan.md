# Plan: RAG Structured Data Improvements — Chart/Table/Diagram Understanding + Eval Fixes

## Context

This is a financial document RAG system targeting 100k-document scale with chart and graph-heavy content (investor decks, quarterly reports, property presentations). Eval run `enterprise_rag_eval_answers_test4_oai.jsonl` exposed 5 concrete failure clusters. This plan addresses all of them plus replaces the document parser with Docling for better baseline structured data understanding.

**Failure clusters from eval (in priority order):**

1. **Answer CoT leakage** (Q24, Q46 — urgent): `<thinking>` tags bleed into final answer
2. **Version cross-contamination** (Q23): BXP Q4 deck and investor day deck answers mixed
3. **Slide/KPI structured retrieval** (Q16, 21, 22, 29, 32, 33): PPTX KPI slides untagged, map slides missing `asset_refs`
4. **Chart legend/visual field accuracy** (Q22, 29, 33): Right slide family found, wrong visual field returned
5. **Parser upgrade**: LiteParse → Docling for better table cell structure and figure detection

---

## Phase 0 — Metadata Enrichment (Document-Level + Chunk-Level)

**Why this comes first:** Company/ticker metadata is the structural fix for cross-company contamination (Q23, Q54, Q58) and wrong-field substitution (Q1, Q30, Q86). All other phases become more precise when metadata is richer.

### Current metadata gap

`_build_document_metadata()` in `app/services/ingest_service.py` (line 405) only produces 7 fields: `document_id`, `client_id`, `client_name`, `document_name`, `file_name`, `file_type`, `ingestion_job_id`. There is no `company`, `ticker`, `sector`, `document_type`, or `reporting_period`.

Qdrant has 6 indexed payload fields (`PAYLOAD_INDEXES` in `app/indexing/vector_store.py` line 25): `client_id`, `document_id`, `file_name`, `document_version_group`, `page_num`, `slide_num`. No company-level index exists.

### Step 0a — Document-level entity extraction

Add a new module `app/ingestion/metadata_extractor.py`:

```python
import re
from pathlib import Path

# Common REIT ticker patterns from filename (BXP, DLR, PSA, O, VICI, EGP, SPG, AMT...)
_TICKER_PATTERNS = [
    re.compile(r"^([A-Z]{2,5})[-_\s]", re.IGNORECASE),       # "BXP_Q4_..." → BXP
    re.compile(r"[-_\s]([A-Z]{2,5})[-_\s]", re.IGNORECASE),  # "reit_BXP_2025" → BXP
]

_DOC_TYPE_SIGNALS = {
    "investor presentation": ["investor", "presentation", "deck"],
    "fact sheet": ["fact sheet", "fact_sheet", "factsheet", "quick facts"],
    "earnings release": ["earnings", "supplement", "release", "8-k"],
    "annual report": ["annual report", "10-k", "10k"],
    "quarterly report": ["10-q", "10q", "quarterly"],
}

_SECTOR_SIGNALS = {
    "office": ["office", "workspace"],
    "data center": ["data center", "datacenter", "digital realty", "equinix"],
    "industrial": ["industrial", "logistics", "warehouse"],
    "retail": ["mall", "simon", "retail", "shopping"],
    "net lease": ["net lease", "realty income", "agree"],
    "self storage": ["storage", "public storage", "extra space"],
    "diversified": ["diversified", "vici", "gaming"],
}

def extract_document_metadata(filename: str, content_preview: str = "") -> dict:
    """
    Extract company, ticker, document_type, sector, and reporting_period
    from filename and up to 500 chars of content.
    These fields flow into every chunk's Qdrant payload.
    """
    stem = Path(filename).stem.upper()
    combined = f"{stem} {content_preview[:300]}".lower()

    ticker = _extract_ticker(stem)
    doc_type = _extract_doc_type(combined)
    sector = _extract_sector(combined)

    return {
        "company_ticker": ticker or "",
        "document_type": doc_type or "unknown",
        "sector": sector or "unknown",
    }


def extract_chunk_metric_types(text: str) -> list[str]:
    """
    Detect which financial KPI types appear in a chunk.
    Stored as metric_types: list[str] in chunk metadata.
    Used by reranker to boost metric-specific chunks.
    """
    lowered = text.lower()
    _METRIC_PATTERNS = {
        "walt": ["walt", "weighted average lease term"],
        "occupancy": ["occupancy", "occupied", "leased %", "leased pct"],
        "noi": ["noi", "net operating income"],
        "abr": ["abr", "annualized base rent", "base rent"],
        "sqft": ["sq ft", "square feet", "sf ", "msf"],
        "dividend": ["dividend yield", "dividend"],
        "cap_rate": ["cap rate", "capitalization rate"],
        "ebitda": ["ebitda", "ebitdare"],
        "ffo": ["ffo", "funds from operations"],
        "enterprise_value": ["enterprise value", "ev "],
        "debt": ["net debt", "total debt", "leverage"],
        "capacity": ["mw ", "megawatt", "gw ", "gigawatt"],
    }
    return [
        metric
        for metric, patterns in _METRIC_PATTERNS.items()
        if any(p in lowered for p in patterns)
    ]
```

### Step 0b — Inject at ingestion time

**In `_build_document_metadata()` (`ingest_service.py` line 405):**

```python
from app.ingestion.metadata_extractor import extract_document_metadata

entity_meta = extract_document_metadata(
    filename=self.filename,
    content_preview=self._raw_content_preview(),  # first 300 chars of extracted text
)
document_metadata.update(entity_meta)
```

**In `_apply_document_metadata()` (`ingest_service.py` line 458), per-node loop:**

```python
from app.ingestion.metadata_extractor import extract_chunk_metric_types

for node in nodes:
    node.metadata["metric_types"] = extract_chunk_metric_types(node.get_content())
```

### Step 0c — Add Qdrant indexed fields for efficient company-scoped retrieval

In `app/indexing/vector_store.py` line 25 (`PAYLOAD_INDEXES`):

```python
PAYLOAD_INDEXES = [
    ("client_id",              KEYWORD),
    ("document_id",            KEYWORD),
    ("file_name",              KEYWORD),
    ("document_version_group", KEYWORD),
    ("company_ticker",         KEYWORD),   # NEW — enables company-scoped filtering
    ("document_type",          KEYWORD),   # NEW
    ("page_num",               INTEGER),
    ("slide_num",              INTEGER),
]
```

Also add `"company_ticker"` and `"metric_types"` to `NON_SEMANTIC_EMBED_METADATA_KEYS` and `NON_SEMANTIC_LLM_METADATA_KEYS` in `app/services/ingest_metadata.py` to prevent them from polluting the embedding vector.

### Step 0d — Use company_ticker in version scoping (replaces filename-prefix approach)

In `app/retrieval/reranker.py:_apply_version_consistency_adjustment()`:

```python
# Key by (company_ticker, document_type) instead of raw document_name
# This prevents BXP Q4 deck and BXP Investor Day from being treated as versions of each other
# (different document_type: "quarterly" vs "investor presentation")
doc_key = (
    meta.get("company_ticker") or meta.get("document_name", ""),
    meta.get("document_type", ""),
    meta.get("version_label", ""),
)
```

### Step 0e — Use metric_types in reranker for KPI field precision

In `app/retrieval/reranker.py:_structural_adjustment()`:

```python
# Boost chunks whose metric_types intersect with query's detected KPIs
query_metrics = _detect_query_metrics(question)  # same patterns as extract_chunk_metric_types
chunk_metrics = set(node.metadata.get("metric_types") or [])
if query_metrics & chunk_metrics:
    adjustment += 0.07  # Direct metric type match — prefer this chunk over generic numeric chunks
```

---

## Phase 1 — URGENT: Fix Answer Post-Processing (CoT Leakage + Truncation)

**File:** `app/retrieval/synthesizer.py`
**Functions:** `_split_reasoning_from_text()` (lines 353-377), `_effective_max_output_tokens()` (lines 50-56)

### Problem

`_split_reasoning_from_text()` uses `re.search(r"<thinking>(.*?)</thinking>", ...)`. When the model's token budget exhausts mid-tag (e.g., `<thinking>... [truncated]`, no `</thinking>`), neither regex matches and raw CoT is returned verbatim. This is Q24 and Q46.

There is also no post-extraction scrub for CoT artifacts that leak into `TextBlock`s ("Let me think...", "Step 1:", numbered reasoning lists).

### Fix

**Step 1a — Robust tag extraction with ordered fallbacks:**

```python
# Replaces _split_reasoning_from_text() in synthesizer.py
_THINKING_OPEN = re.compile(r"<thinking>", re.IGNORECASE)
_THINKING_CLOSE = re.compile(r"</thinking>", re.IGNORECASE)
_ANSWER_OPEN = re.compile(r"<answer>", re.IGNORECASE)
_ANSWER_CLOSE = re.compile(r"</answer>", re.IGNORECASE)
_COT_ARTIFACT_PATTERN = re.compile(
    r"^(let me (think|analyze|break|consider)|step \d+[:.]|"
    r"first[,:]|to answer this|thinking:|analysis:)",
    re.IGNORECASE | re.MULTILINE,
)

def _split_reasoning_from_text(text: str) -> tuple[str, str]:
    """
    Extract (reasoning, answer) with ordered fallbacks for malformed/truncated tags.
    """
    text = text.strip()

    # Strategy 1: Both tags well-formed
    thinking_m = re.search(r"<thinking>(.*?)</thinking>", text, re.IGNORECASE | re.DOTALL)
    answer_m = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if thinking_m and answer_m:
        return thinking_m.group(1).strip(), _scrub_cot_artifacts(answer_m.group(1).strip())

    # Strategy 2: <answer> tag found, no valid <thinking> (model skipped reasoning)
    if answer_m:
        return "", _scrub_cot_artifacts(answer_m.group(1).strip())

    # Strategy 3: <thinking> opened and closed, no <answer> — use text after </thinking>
    if thinking_m:
        after_thinking = text[thinking_m.end():].strip()
        return thinking_m.group(1).strip(), _scrub_cot_artifacts(after_thinking)

    # Strategy 4: <thinking> opened but NOT closed (token budget exhausted mid-tag)
    # Discard everything from <thinking> onward — return any text BEFORE it
    open_m = _THINKING_OPEN.search(text)
    if open_m:
        before = text[: open_m.start()].strip()
        if before:
            return "", _scrub_cot_artifacts(before)
        # Nothing usable before thinking tag — answer is empty (signal truncation)
        return text[open_m.end():].strip(), ""

    # Strategy 5: No tags at all — return scrubbed raw text
    return "", _scrub_cot_artifacts(text)


def _scrub_cot_artifacts(text: str) -> str:
    """Remove leading CoT artifact lines from text blocks."""
    lines = text.splitlines()
    clean = []
    for line in lines:
        if _COT_ARTIFACT_PATTERN.match(line.strip()):
            continue
        clean.append(line)
    return "\n".join(clean).strip()
```

**Step 1b — Completeness check before returning answer:**

```python
# In GroundedAnswerResult construction (synthesizer.py, around line 199)
_TRUNCATION_SIGNALS = re.compile(r"(\.{3}$|\[trunc|\s\w{1,4}$)", re.IGNORECASE)

def _validate_answer_completeness(answer: str) -> str:
    """Flag obviously truncated answers; do not suppress, just log."""
    if not answer:
        return answer
    stripped = answer.rstrip()
    if _TRUNCATION_SIGNALS.search(stripped):
        logger.warning("Answer may be truncated: ends with %r", stripped[-30:])
    return answer
```

**Step 1c — Token budget: extend for single-chart/map questions:**

```python
# In _effective_max_output_tokens (synthesizer.py, lines 50-56)
# Add visual-heavy query signals to the complex-query bump
_VISUAL_SIGNALS = ("map", "chart", "graph", "figure", "diagram", "pie", "bar", "trend")

def _effective_max_output_tokens(question: str, base: int) -> int:
    visual_hit = any(t in question.lower() for t in _VISUAL_SIGNALS)
    complex_hit = sum(1 for s in _COMPLEX_QUERY_SIGNALS if s in question.lower()) >= 2
    if visual_hit or complex_hit:
        return max(base, 2000)
    return base
```

---

## Phase 2 — Version Pinning and Document Scoping

**Files:** `app/ingestion/version_resolver.py`, `app/retrieval/reranker.py`

### Problem

- `is_current` is almost always `False` (set only on keyword "final"/"latest")
- `_VERSION_PENALTY = 0.03` is too weak; stale chunks rank 0.72 vs current 0.72+0.03 = trivial difference
- Two different decks from the same company (BXP Q4 deck, BXP Investor Day) share `version_group` if filenames share prefix — they are treated as versions of each other

### Fix

**Step 2a — Auto-set `is_current` for the highest-rank document in each version group:**

`version_resolver.py` runs per-document. The fix is in post-processing after all documents for a client are resolved. In `app/services/ingest_service.py` (wherever `resolve_version()` is called), after resolving all documents for a `client_id`, apply:

```python
def _mark_current_by_version_rank(resolved_versions: list[dict]) -> None:
    """For each version_group, mark the highest version_rank as is_current=True."""
    by_group: dict[str, list[dict]] = {}
    for v in resolved_versions:
        group = v.get("version_group", "")
        by_group.setdefault(group, []).append(v)

    for group, versions in by_group.items():
        if len(versions) <= 1:
            continue
        best = max(versions, key=lambda v: v.get("version_rank", 0))
        if not best.get("is_current"):
            best["is_current"] = True
```

**Step 2b — Strengthen version penalty and add document-scoped version grouping:**

```python
# In reranker.py
_VERSION_PENALTY = 0.08   # was 0.03 — make stale-version suppression meaningful

# In _apply_version_consistency_adjustment: key by (document_name, version_label)
# instead of just document_id to prevent cross-company grouping errors
def _apply_version_consistency_adjustment(nodes, question):
    if _is_comparison_query(question):
        return nodes  # preserve multi-version for explicit compare queries

    # Group by (document_name, version_group) — never mix documents
    from collections import defaultdict
    group_scores: dict[tuple, float] = defaultdict(float)
    node_groups: dict[str, tuple] = {}

    for ns in nodes:
        meta = ns.node.metadata
        doc_name = str(meta.get("document_name", meta.get("document_id", "")))
        version_label = str(meta.get("version_label", ""))
        group_key = (doc_name, version_label)
        group_scores[group_key] += ns.score or 0
        node_groups[ns.node.node_id] = group_key

    # Find dominant version per document_name (not globally)
    doc_dominant: dict[str, tuple] = {}
    for (doc_name, ver_label), total_score in group_scores.items():
        if doc_name not in doc_dominant or total_score > group_scores[doc_dominant[doc_name]]:
            doc_dominant[doc_name] = (doc_name, ver_label)

    for ns in nodes:
        node_group = node_groups.get(ns.node.node_id)
        if node_group is None:
            continue
        doc_name = node_group[0]
        if node_group != doc_dominant.get(doc_name):
            ns.score = (ns.score or 0) - _VERSION_PENALTY

    return nodes
```

**Step 2c — Scope structured evidence injection to same document_id:**

In `evidence_selector.py:_ensure_structured_evidence()`, when a query is not explicitly comparative, restrict the injected candidate to have the same `document_id` as the highest-ranked node already in `selected`:

```python
# In _ensure_structured_evidence, add before the candidate loop:
anchor_doc_id = None
if selected and not _is_comparison_or_conflict_query(question):
    anchor_doc_id = selected[0].node.metadata.get("document_id")

# In the candidate matching loop:
if anchor_doc_id:
    cand_doc_id = candidate.node.metadata.get("document_id")
    if cand_doc_id and cand_doc_id != anchor_doc_id:
        continue  # don't inject from a different document
```

---

## Phase 3 — Slide/KPI Structured Retrieval

**Files:** `app/ingestion/parser.py`, `app/retrieval/evidence_selector.py`

### Problem A — PPTX slides missing `slide_purpose` tag

The legacy parser (`_parse_legacy`) stores each slide as one flat text chunk with no `slide_purpose` field. The PDF pipeline's `_classify_slide_purpose()` logic in `chunk_builder.py` (e.g., `overview_stats` for high numeric-density short slides) is never applied to PPTX files.

### Fix A — Tag PPTX slides with `slide_purpose` in `_parse_legacy()`

```python
# In parser.py, add to the add_item() inner function:
from app.ingestion.pdf_pipeline.helpers import numeric_density as _numeric_density

def _classify_slide_purpose_simple(text: str) -> str:
    """Lightweight slide_purpose classifier for legacy PPTX parsing."""
    nd = _numeric_density(text)
    length = len(text)
    if length < 600 and nd > 0.08:
        return "overview_stats"
    if nd > 0.06:
        return "data_slide"
    return "content"

# In add_item():
slide_purpose = (
    _classify_slide_purpose_simple(text) if unit_type == "slide" else None
)
meta = {
    **document_metadata,
    meta_key: page_or_slide,
    "source_file": path.name,
    "parser_name": PARSER_NAME,
    "parser_version": LEGACY_PARSER_VERSION,
}
if slide_purpose:
    meta["slide_purpose"] = slide_purpose
```

### Problem B — KPI/stats slides not injected as structured evidence

`_ensure_structured_evidence()` only handles tables and charts. A slide tagged `slide_purpose == "overview_stats"` never gets force-injected even when the query is a headline KPI question.

### Fix B — Add `_is_stats_slide_node()` and inject stats slides

```python
# In evidence_selector.py
def _is_stats_slide_node(node: NodeWithScore) -> bool:
    meta = node.node.metadata
    return meta.get("slide_purpose") in {"overview_stats", "data_slide"}

def _wants_stats_slide_evidence(question: str) -> bool:
    lowered = question.lower()
    return any(
        term in lowered
        for term in (
            "properties", "leased", "walt", "abr", "occupancy",
            "enterprise value", "total", "portfolio", "sq ft",
            "square feet", "assets under", "aum", "noi", "ebitda",
        )
    )
```

Add this injection in `_ensure_structured_evidence()` alongside table/chart injection (same append-or-replace pattern).

### Problem C — Map slides missing `asset_refs`

Q32 and Q33 (world capacity map, US footprint map) fail because `_ensure_image_evidence()` requires `_node_has_image_assets(node)` — which checks `asset_refs`. Map slides from the legacy PPTX parser have no `asset_refs` because screenshots are never generated for PPTX.

### Fix C — Generate slide screenshots for PPTX during ingestion

In `_parse_legacy()` for PPTX files, use `python-pptx` (already a likely transitive dependency) to render slide thumbnails, or use LibreOffice headless conversion + PyMuPDF to render screenshots per slide, storing them as `asset_refs`:

```python
# In _parse_legacy, for ".pptx":
# After reading slide text, attempt to save slide thumbnail
slide_screenshot_path = _try_save_slide_screenshot(file_path, i, dest_folder="data/slides")
if slide_screenshot_path:
    meta["asset_refs"] = [slide_screenshot_path]
```

Where `_try_save_slide_screenshot()` uses `python-pptx`'s slide.shapes to render via `PIL` or delegates to a configurable method, failing silently if unavailable.

---

## Phase 4 — Chart Legend and Visual Field Accuracy

**Files:** `app/core/prompts.py`, `app/ingestion/pdf_pipeline/models.py`, `app/ingestion/pdf_pipeline/artifact_builders.py`

### Problem

The LLM chart captioning prompt (`CHART_REASONING_PROMPT`) and `ChartCaptionResponse` model capture `series` as a flat list of strings but don't explicitly extract legend labels, color-to-series mappings, or per-segment/per-bar labels. Queries like "NOI mix by market" need the legend categories, not just the chart title.

### Fix A — Enrich `ChartCaptionResponse` with legend and segment fields

The `ChartCaptionResponse` model (`models.py`) already has `pie_segments`, `matrix_cells`, and `stat_box_values`. Add `legend_items`:

```python
# In models.py, ChartCaptionResponse:
legend_items: list[str] = Field(default_factory=list)
# Each item: "Label: value" or "Label (color_hint)" for bar/line legend entries
```

### Fix B — Update `build_chart_caption_prompt()` to request legend extraction

In `app/core/prompts.py`, add to `CHART_REASONING_PROMPT` (the multimodal captioning instructions):

```
- "legend_items": list of every legend label with its associated value or percentage if readable
  (e.g., "Boston: 34%", "San Francisco: 18%", "NYC: 15%").
  For bar/line charts: list series labels. For pie/donut: list segmnt label + value.
  If chart has no legend, return [].
```

### Fix C — Create `chart_legend_chunk` for direct legend retrieval

In `chunk_builder.py:_figure_to_chunks()`, after the existing `chart_data_points` chunk, emit a `chart_legend_chunk` when `legend_items` is non-empty:

```python
if is_chart_like and figure.legend_items:
    legend_text = (
        f"Chart legend — {figure.chart_title or figure.caption_text}:\n"
        + "\n".join(figure.legend_items)
    )
    chunks.append(ChunkArtifact(
        chunk_id=str(uuid.uuid4()),
        chunk_type="chart_legend_chunk",
        source_artifact_type="figure",
        source_artifact_id=figure.figure_id,
        page_nums=[figure.page_num],
        text=legend_text,
        metadata=metadata,
        asset_refs=assets,
    ))
```

Add `"chart_legend_chunk"` to `_is_chart_like_node` in `evidence_selector.py` and the 2400-char excerpt budget list in `prompt_builder.py`.

---

## Phase 5 — Docling Parser Swap (LiteParse Fully Removed)

**Status:** Partially complete. `adapters.py` and `config.py` already updated. Two artifact_builders changes and full liteparse removal remain.

**Why:** DocLayNet model gives better table cell structure and accurate figure bboxes vs LiteParse + PyMuPDF heuristics. LiteParse also requires a Node.js npm global install (`@llamaindex/liteparse`) — a fragile runtime dependency. Removing it entirely is cleaner than keeping it as a fallback.

### What's already done (previous session)

- `docling` installed via `uv add docling`
- `ENABLE_DOCLING_PARSER: bool = True` added to `app/core/config.py`
- `_get_docling_converter()` singleton, `_docling_table_to_rows()`, `_bbox_bottomleft_to_topleft()`, `extract_docling_pages()` all added to `adapters.py`
- `DefaultPDFExtractionStage.extract()` updated to call `extract_docling_pages()` when flag is set, merge `docling_by_page` into `pymupdf_pages`
- Coordinate system mismatch fixed: `bbox.to_top_left_origin(page_height)` converts Docling BOTTOMLEFT → PyMuPDF TOPLEFT

### Step 5a — Complete liteparse removal from `adapters.py`

**Remove:**

- `from liteparse import LiteParse` import (line 9)

- `extract_liteparse_pages()` function (lines 264–332)
- The `else` branch in `DefaultPDFExtractionStage.extract()` that called `extract_liteparse_pages()` (lines 219–224) — always call `extract_docling_pages()` now
- `ENABLE_DOCLING_PARSER` config branch (always Docling, no flag check needed in the extract method)

**Rename:**

- `build_liteparse_fallback_pages()` → `build_pymupdf_text_pages()` — it's pure PyMuPDF (no liteparse), just needs the name fixed
- Update the call inside `extract_docling_pages()` at line 101 to use new name

**Rename the contract field `liteparse_pages` → `text_pages` across all consumers:**

| File | Location |
|------|----------|
| `app/ingestion/pdf_pipeline/contracts.py` | `ExtractionResult.liteparse_pages` field; `PageStructureStage.build()` param; `ChunkBuildStage.build()` param |
| `app/ingestion/pdf_pipeline/pipeline.py` | `extraction.liteparse_pages` (2 references) |
| `app/ingestion/pdf_pipeline/page_structure.py` | `liteparse_pages` param name + all internal uses; `_blocks_from_liteparse_items()` → `_blocks_from_text_items()` |
| `app/ingestion/pdf_pipeline/chunk_builder.py` | `liteparse_pages` param + `liteparse_pages.json` dump filename → `text_pages.json` |
| `app/ingestion/pdf_pipeline/adapters.py` | Return variable names inside `extract_docling_pages()` |

### Step 5b — Update `build_figure_artifacts()` in `artifact_builders.py`

In `build_figure_artifacts()` (line 612), prepend Docling figure candidates before PyMuPDF candidates so `_dedupe_candidates()` (IoU > 0.8) keeps the more precise Docling bboxes:

```python
candidates: list[dict[str, Any]] = []

# Docling picture candidates first — DocLayNet bboxes win deduplication
for fig_cand in page.get("docling_figure_candidates") or []:
    candidates.append(fig_cand)  # already has {"bbox": [...], "kind": "docling_picture"}

# PyMuPDF image refs
for image in page.get("image_refs") or []:
    bbox = image.get("bbox")
    if bbox:
        candidates.append({"bbox": to_float_bbox(bbox), "kind": "image"})

# PyMuPDF vector-dense regions
vector_count = int(page.get("vector_count") or 0)
vector_union = union_bbox(page.get("vector_bboxes") or [])
if vector_count >= 20 and vector_union:
    candidates.append({"bbox": vector_union, "kind": "vector_dense"})
```

Docling candidates get confidence 0.90 (vs 0.83 for captioned non-Docling) — update `FigureArtifact` confidence assignment:

```python
is_docling = candidate.get("kind") == "docling_picture"
artifacts.append(FigureArtifact(
    ...
    confidence=0.90 if is_docling else (0.83 if caption else 0.72),
))
```

### Step 5c — Add `"docling_picture"` to `_classify_figure()` in `artifact_builders.py`

Add before the final `return "unknown"` at line 1087:

```python
if candidate_type == "docling_picture":
    return "figure"
```

### Step 5d — Remove liteparse from `pyproject.toml` and `setup.sh`

- `pyproject.toml`: remove `"liteparse>=1.2.1"` from dependencies
- `setup.sh`: remove the `npm install -g @llamaindex/liteparse` block and associated echo messages

---

## Phase 6 — Table Aggregation Summaries + Temporal Scope

**File:** `app/ingestion/pdf_pipeline/chunk_builder.py` (both), `app/retrieval/reranker.py` (temporal)

**Table aggregation summary:** New `table_aggregation_summary` chunk that precomputes column min/max/avg/total for numeric columns. Fixes "highest Q2 revenue" queries. Add `_build_table_aggregation_summary()` in `_table_to_chunks()`.

**Temporal scope metadata:** Extract Q1-Q4 / FY year labels from table header rows via regex. Store in chunk metadata `temporal_scope: {quarters, years}`. Apply +0.06 match / -0.03 mismatch adjustment in `reranker.py:_temporal_adjustment()`.

Both are ingestion-time only (pure Python, no LLM calls), zero query-time cost.

---

## Phase 7 — KPI Field Disambiguation + Page Citation Fix

### Problem A — Wrong field from the right slide (Q1: 10.1yr WALT vs 7.6yr; Q3: $32.5M vs $32.0M)

Q1 returns 10.1 years (WALT from a lease-term detail slide) instead of 7.6 years (WALT from the Quick Facts KPI card on page 3). Both are numerically valid for WALT — the retriever found the right document but not the right slide. The KPI Quick Facts card chunk ranks below a more verbose lease-discussion chunk.

The fix is in the `GROUNDED_ANSWER_DEVELOPER_PROMPT` — add an explicit instruction to prefer the KPI card / overview-stats chunk over narrative discussion when the query asks for a specific headline metric, and to use only the value from the cited source (not nearby similar fields):

```
Rule 9: When answering a request for a specific headline metric (e.g., WALT, occupancy %, ABR,
sq ft), prefer the evidence chunk tagged slide_purpose=overview_stats or chunk_type=table_aggregation_summary
over body text chunks. If two chunks give different values for the same metric, cite both and note
which is the primary KPI card value.
```

Also add a `slide_purpose` boost in `reranker.py:_structural_adjustment`: if `slide_purpose == "overview_stats"` and `wants_numeric`, add +0.07 (currently only `"overview_stats"` with factual lookup adds +0.06 — expand to all numeric-intent queries).

### Problem B — Page citation grounding wrong (Q86: says page 37, correct is page 6)

`citation_builder.py` builds citations from `page_num` metadata on the chunk. The incorrect page number suggests the chunk for the BXP occupancy sensitivity statement was retrieved from a different page or a continuation chunk from the wrong page.

**File:** `app/retrieval/citation_builder.py`

Check that `page_num` in the citation is taken from the actual matched chunk's metadata, not from a parent or neighboring chunk. Add a deduplification step: if two citations point to the same `document_id` and `source_artifact_id` but different `page_num` values, prefer the lower page number (the primary page where the artifact starts).

```python
# In citation_builder.py, in _dedupe_citations() or equivalent:
def _resolve_citation_page(citation: dict) -> int:
    """Use minimum page_num when a chunk spans multiple pages."""
    page_nums = citation.get("page_nums", [citation.get("page_num", 0)])
    if isinstance(page_nums, list) and page_nums:
        return min(page_nums)
    return int(page_nums or 0)
```

---

## Phase 8 — Compound Chart+Numeric Boost + Cross-Reference Resolution

**Compound boost** (`reranker.py:_structural_adjustment`): Add +0.08 compound signal when both chart intent AND numeric intent detected simultaneously, only on chart chunks with non-empty `approx_datapoints` or `trend_summary`.

**Cross-reference resolution** (`retriever.py`): After reranking, scan top-5 nodes for "Table 3" / "Figure 2" text references and auto-inject the referenced artifact chunk. Bounded at max 2 injections, O(1) relative to corpus.

---

## Files Modified

| File | Change |
|------|--------|
| `app/retrieval/synthesizer.py` | Rewrite `_split_reasoning_from_text()` with 5-strategy fallback; add `_scrub_cot_artifacts()`; extend token budget for visual queries |
| `app/ingestion/version_resolver.py` | No direct change; caller adds `_mark_current_by_version_rank()` post-processing |
| `app/services/ingest_service.py` | Add `_mark_current_by_version_rank()` after version resolution |
| `app/retrieval/reranker.py` | Fix `_apply_version_consistency_adjustment()` key scoping; raise `_VERSION_PENALTY` 0.03→0.08; add compound chart+numeric boost; add temporal scope signals |
| `app/retrieval/evidence_selector.py` | Add `_is_stats_slide_node()`, `_wants_stats_slide_evidence()`, inject stats slides in `_ensure_structured_evidence()`; add document_id scoping to structured evidence injection; add `"table_aggregation_summary"`, `"chart_legend_chunk"` to predicates |
| `app/ingestion/parser.py` | Add `_classify_slide_purpose_simple()` and `slide_purpose` tagging in `_parse_legacy()`; attempt slide screenshots for PPTX `asset_refs` |
| `app/core/prompts.py` | Add `legend_items` instruction to `CHART_REASONING_PROMPT` / `build_chart_caption_prompt()` |
| `app/ingestion/pdf_pipeline/models.py` | Add `legend_items: list[str]` to `ChartCaptionResponse` and `FigureArtifact` |
| `app/ingestion/pdf_pipeline/artifact_builders.py` | Propagate `legend_items` in `_apply_llm_chart_fields()`; accept Docling figure candidates in `build_figure_artifacts()`; accept Docling table candidates in `_table_candidates_for_page()` |
| `app/ingestion/pdf_pipeline/chunk_builder.py` | Add `_build_table_aggregation_summary()`, `_extract_table_temporal_scope()`; emit `table_aggregation_summary` and `chart_legend_chunk` chunk types |
| `app/ingestion/pdf_pipeline/adapters.py` | Add `extract_docling_pages()`, `_docling_text_items()`, `_render_page_screenshot()`, `_docling_table_candidates()`, `_docling_figure_candidates()`, `_get_docling_converter()` |
| `app/ingestion/pdf_pipeline/pipeline.py` | Swap to `extract_docling_pages()` behind `ENABLE_DOCLING_PARSER` flag; pass Docling table/figure candidates to artifact stage |
| `app/retrieval/retriever.py` | Add `_resolve_cross_references()` helper |
| `app/retrieval/prompt_builder.py` | Add `"table_aggregation_summary"`, `"chart_legend_chunk"` to 2400-char excerpt budget list |
| `app/core/config.py` | Add `ENABLE_DOCLING_PARSER: bool = True` |
| `pyproject.toml` / `setup.sh` | Add `docling`; remove `@llamaindex/liteparse` npm dep |

---

## Scale Considerations (100k Financial Documents)

- Phases 1-3 are query-time changes — no ingestion re-run needed
- Phases 4-7 require re-ingesting documents but are all ingestion-time computation (no extra LLM calls)
- Docling singleton converter (`_DOCLING_CONVERTER`) loads DocLayNet model once per worker — stays warm across all documents
- Docling CPU mode runs without GPU; GPU workers give ~5× throughput for bulk re-ingestion
- `_VERSION_PENALTY` raise (0.03→0.08) affects only the reranker — no ingestion impact

---

## Verification

### Regression tests

**Phase 1 (CoT fix):**

```python
def test_split_reasoning_missing_close_tag():
    text = "<thinking>step 1: analyze\nstep 2: conclude"  # no </thinking>
    reasoning, answer = _split_reasoning_from_text(text)
    assert "<thinking>" not in answer
    assert "thinking" not in answer.lower()

def test_split_reasoning_both_tags():
    text = "<thinking>reasoning</thinking><answer>42%</answer>"
    _, answer = _split_reasoning_from_text(text)

    assert answer == "42%"

def test_scrub_cot_artifacts():
    text = "Let me think through this.\nThe answer is 42%."
    assert _scrub_cot_artifacts(text) == "The answer is 42%."
```

**Phase 2 (version scoping):**

```python

def test_version_penalty_increased():
    assert reranker._VERSION_PENALTY == 0.08

def test_structured_evidence_same_document_only():
    # table from doc_A should not inject when query anchors to doc_B
    ...
```

**Phase 3 (slide tagging):**

```python
def test_pptx_slide_purpose_tagged():
    # A numeric-dense short slide gets slide_purpose="overview_stats"
    ...
```

### End-to-end eval targets (from `enterprise_rag_eval_answers_test4_oai.jsonl`)

**Baseline:** 67/92 correct (72.8% strict), 74/92 lenient (80.4%). Target: ≥85% strict after these fixes.

| Q# | Current failure | Fix phase | Pass condition |
|----|----------------|-----------|----------------|
| Q1 | Returns 10.1yr WALT; misses 52.6M sq ft, 4.7% yield from Quick Facts slide | Ph3 + Ph7 | Returns 7.6yr from overview_stats slide; includes sq ft + yield |
| Q3 | NOI $32.5M instead of $32.0M | Ph5 (Docling table precision) | Returns $32.0M |
| Q16 | Incomplete peer NOI/FCF comparison | Ph3 + Ph4 | Full peer matrix retrieved |
| Q19, Q20 | Wrong field from right document | Ph7 (KPI boost) | Correct metric from overview_stats chunk |
| Q21 | Table truncated at row 2 | Ph6 (aggregation) + Ph3 | Full table rows returned |
| Q22 | NOI-by-market pie chart miss | Ph4 (legend chunk) | Legend items in evidence |
| Q23 | BXP + Digital Realty figures mixed | Ph2 (version scoping) | Single-company answer |
| Q24, Q46 | `<thinking>` leaks into answer | Ph1 (CoT fix) | Zero `<thinking>` in output |
| Q29 | Co-tenancy matrix column missing | Ph4 (legend/matrix) | Matrix cells in evidence |
| Q30 | Wrong field substitution | Ph7 (KPI prompt rule) | Correct slot value cited |
| Q32, Q33 | Map slides not retrieved | Ph3 (PPTX screenshots) | Map chunk with asset_refs in evidence |
| Q35 | Partial table/infographic | Ph5 (Docling) + Ph4 | Complete structured data |
| Q54, Q58 | Cross-deck contamination | Ph2 (document scoping) | Single-deck answer |
| Q80 | Infographic data miss | Ph4 (chart legend) | Key stat_box values retrieved |
| Q86 | Page 37 cited instead of page 6 | Ph7 (citation min-page) | Correct page number in citation |

**Partial (target full credit):** Q21, Q23, Q34, Q36, Q37, Q48, Q91 — mostly table completeness and version precision.
