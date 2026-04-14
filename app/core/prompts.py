from __future__ import annotations

from llama_index.core.prompts import PromptTemplate

QUERY_EXPANSION_PROMPT = PromptTemplate(
    "Rewrite the user question into at most {max_rewrites} short retrieval queries "
    "for enterprise document RAG. Preserve concrete product names, dates, versions, "
    "and numeric terms. Do not answer the question.\n\n"
    "Return a structured object with `rewrites` containing only rewritten queries.\n\n"
    "Question: {question}"
)


def build_chart_caption_prompt(
    *,
    page_num: int,
    page_class: str,
    caption_text: str,
    nearby_text: str,
    current_proxy: str,
) -> str:
    return "\n".join(
        [
            "You are extracting chart structure from a financial report.",
            "Return strict JSON with keys:",
            "chart_type, chart_title, x_axis_label, y_axis_label, x_categories, series, approx_datapoints, trend_summary, key_chart_facts, numeric_extraction_confidence.",
            "approx_datapoints must be an array of objects with shape:",
            '{"series": "...", "x": "...", "y": <number>, "unit": "...", "approximate": true}',
            "Use approximate values when exact values are not readable.",
            "Do not include markdown. Output JSON only.",
            f"Page: {page_num}",
            f"Page class: {page_class}",
            f"Caption: {caption_text}",
            f"Nearby text: {nearby_text}",
            f"Current proxy: {current_proxy}",
        ]
    )


def build_artifact_enrichment_prompt(
    *,
    page_num: int,
    table_summaries: list[tuple[str, str]],
    figure_summaries: list[tuple[str, str]],
) -> str:
    lines = [
        "Summarize these artifacts for retrieval.",
        "Output 4-6 concise bullet points with concrete metrics, units, and interpretation cues.",
        "Do not speculate.",
        f"Page: {page_num}",
    ]

    if table_summaries:
        lines.append("Tables:")
        for caption, content in table_summaries:
            lines.append(f"- Caption: {caption}")
            lines.append(f"- Content: {content[:1200]}")

    if figure_summaries:
        lines.append("Figures:")
        for caption, nearby_text in figure_summaries:
            lines.append(f"- Caption: {caption}")
            lines.append(f"- Nearby: {nearby_text[:1200]}")

    return "\n".join(lines)


PAGE_SCREENSHOT_PROMPT = """\
You are a data analyst reviewing a full document page/slide screenshot.
Describe everything visible: charts, tables, maps, diagrams, annotations, and key numbers.
Output a structured plain-text summary with:
- Page layout description (what elements are present and how they relate)
- All numeric values, metrics, and units visible
- Chart/graph descriptions including axes, trends, and approximate data points
- Table contents summarized with key rows and columns
- Map/diagram annotations and geographic/spatial data
- Key takeaways and relationships between visual elements
Be exhaustive. Do not omit any numbers, labels, or annotations visible on the page.
"""


REASONING_PROMPT_VERSION = "reasoning_v1"

TABLE_REASONING_PROMPT = """\
You are a financial data analyst. Analyze the following table artifact and produce grounded insights.

Table caption: {caption}
Section: {section_path}
Page: {page_num}
Units: {units}
Content (first 1500 chars):
{content}

Output a JSON object with:
- "key_insights": list of 3-5 specific factual insights with concrete numbers
- "metric_comparisons": list of comparisons between rows/columns (e.g. "X grew 15% vs Y")
- "trend_statement": one sentence describing the overall trend
- "caveats": list of data quality warnings or assumptions
- "evidence_refs": list of specific cell references or row labels supporting each claim

Be precise. Every claim must reference specific data from the table. Do not speculate.
"""

CHART_REASONING_PROMPT = """\
You are a financial data analyst. Analyze the following chart/figure artifact and produce grounded insights.

Chart type: {chart_type}
Title: {chart_title}
Caption: {caption}
Section: {section_path}
Page: {page_num}
X axis: {x_axis}
Y axis: {y_axis}
Series: {series}
Trend: {trend_summary}
Key facts: {key_facts}
Approximate datapoints (sample): {datapoints}

Output a JSON object with:
- "key_insights": list of 3-5 specific factual insights with concrete numbers
- "metric_comparisons": list of comparisons between series/categories
- "trend_statement": one sentence describing the overall trend
- "caveats": list of data quality warnings (e.g. approximate values)
- "evidence_refs": list of specific series names, axis labels, or datapoints supporting each claim

Be precise. Every claim must cite specific data from the chart. Do not speculate.
"""

PAGE_REASONING_PROMPT = """\
You are a financial data analyst. Analyze the relationships between artifacts on this page.

Page: {page_num}
Page class: {page_class}
Artifacts present:
{artifact_summaries}

LLM page summary (if available):
{llm_page_summary}

Output a JSON object with:
- "key_insights": list of 3-5 cross-artifact insights (how table data relates to chart trends)
- "metric_comparisons": list of consistency checks between artifacts
- "trend_statement": one sentence summarizing the page overall message
- "caveats": any inconsistencies or gaps between artifacts
- "evidence_refs": specific artifact IDs and data points supporting each claim

Be precise. Every claim must reference specific artifacts and their data. Do not speculate.
"""


def build_grounded_answer_prompt(
    *,
    question: str,
    image_attachment_count: int,
    evidence_block: str,
    conflict_block: str,
) -> str:
    return (
        "You are a retrieval-grounded assistant for sensitive enterprise documents.\n"
        "Answer using only the provided evidence snippets and attached images.\n"
        "Rules:\n"
        "1) If evidence is insufficient or contradictory, say so explicitly.\n"
        "2) For each factual claim, cite at least one source index like [1].\n"
        "3) Do not cite sources that are not in the evidence list.\n"
        "4) Prefer the most current/effective version unless the question asks for comparison.\n"
        "5) If conflict hints are empty, avoid absolute claims such as "
        "'no conflicts exist'; state only what was or was not detected in "
        "the retrieved evidence.\n"
        "6) Use attached images when they help resolve chart/table/map questions.\n"
        "7) Return output using this exact format:\n"
        "<thinking>\n"
        "your step-by-step reasoning grounded in source indices\n"
        "</thinking>\n"
        "<answer>\n"
        "final answer with inline citations like [1], [2]\n"
        "</answer>\n\n"
        f"Question:\n{question}\n\n"
        f"Attached image count: {image_attachment_count}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"Conflict hints:\n{conflict_block}\n\n"
        "Answer:"
    )
