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
    def _truncate(value: str, max_chars: int) -> str:
        text = value.strip()
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 16].rstrip()} ...[truncated]"

    caption = _truncate(caption_text, 320)
    nearby = _truncate(nearby_text, 900)
    proxy = _truncate(current_proxy, 700)

    return "\n".join(
        [
            "You are extracting chart structure from a financial report.",
            "Extract all fields in the provided response schema.",
            "Capture explicit chart labels and values whenever legible.",
            "Use approximate values when exact values are not readable.",
            "Do not speculate beyond visible chart evidence.",
            f"Page: {page_num}",
            f"Page class: {page_class}",
            f"Caption: {caption}",
            f"Nearby text: {nearby}",
            f"Current proxy: {proxy}",
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
        "Return concise evidence-grounded summary points with concrete metrics and units.",
        "Each point should be independently useful for retrieval and answering.",
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
Fill the provided structured fields with:
- Page layout description (what elements are present and how they relate)
- Numeric values, metrics, and units visible
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

Fill the provided structured fields:
- "key_insights": list of 3-5 specific factual insights with concrete numbers
- "metric_comparisons": list of comparisons between rows/columns (e.g. "X grew 15% vs Y")
- "trend_statement": one sentence describing the overall trend
- "caveats": list of data quality warnings or assumptions
- "evidence_refs": JSON array of specific cell references or row labels supporting each claim

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

Fill the provided structured fields:
- "key_insights": list of 3-5 specific factual insights with concrete numbers
- "metric_comparisons": list of comparisons between series/categories
- "trend_statement": one sentence describing the overall trend
- "caveats": list of data quality warnings (e.g. approximate values)
- "evidence_refs": JSON array of specific series names, axis labels, or datapoints supporting each claim

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

Fill the provided structured fields:
- "key_insights": list of 3-5 cross-artifact insights (how table data relates to chart trends)
- "metric_comparisons": list of consistency checks between artifacts
- "trend_statement": one sentence summarizing the page overall message
- "caveats": any inconsistencies or gaps between artifacts
- "evidence_refs": JSON array of specific artifact IDs and data points supporting each claim

Be precise. Every claim must reference specific artifacts and their data. Do not speculate.
"""


GROUNDED_ANSWER_DEVELOPER_PROMPT = (
    "You are a retrieval-grounded assistant for sensitive enterprise documents.\n"
    "Use only RETRIEVAL_EVIDENCE for factual claims. "
    "SESSION_SUMMARY and historical context can resolve references only.\n"
    "Rules:\n"
    "1) If evidence is insufficient or contradictory, say so explicitly.\n"
    "2) For each factual claim, cite at least one source index like [1].\n"
    "3) Do not cite sources that are not in RETRIEVAL_EVIDENCE.\n"
    "4) Prefer the most current/effective version unless asked to compare.\n"
    "5) If conflict hints are empty, avoid absolute claims about no conflicts.\n"
    "6) When images are attached, use them for chart/table/map interpretation.\n"
    "7) Return this exact format:\n"
    "<thinking>\n"
    "step-by-step grounded reasoning with source indices\n"
    "</thinking>\n"
    "<answer>\n"
    "final answer with inline citations like [1], [2]\n"
    "</answer>"
)

SESSION_SUMMARY_DEVELOPER_PROMPT = (
    "You maintain a concise running conversation summary for enterprise chat.\n"
    "Keep entities, decisions, constraints, and unresolved asks.\n"
    "Output plain text only, max 8 short lines, no markdown."
)


def build_grounded_answer_prompt(
    *,
    question: str,
    image_attachment_count: int,
    evidence_block: str,
    conflict_block: str,
    conversation_context_block: str,
) -> str:
    """
    Backward-compatible combined prompt builder.

    New synthesis path uses static developer prompt + structured user content.
    """
    return (
        f"CURRENT_QUERY:\n{question}\n\n"
        f"ATTACHED_IMAGE_COUNT:\n{image_attachment_count}\n\n"
        f"CONVERSATION_CONTEXT:\n{conversation_context_block}\n\n"
        f"RETRIEVAL_EVIDENCE:\n{evidence_block}\n\n"
        f"CONFLICT_HINTS:\n{conflict_block}\n"
    )


_GENERIC_STRUCTURED_PROMPT = PromptTemplate("{user_prompt}")
