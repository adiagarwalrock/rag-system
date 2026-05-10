from llama_index.core.prompts import PromptTemplate

QUERY_EXPANSION_PROMPT = PromptTemplate(
    "Rewrite the user question into at most {max_rewrites} short retrieval queries "
    "for enterprise document RAG. Preserve concrete product names, dates, versions, "
    "and numeric terms. Use recent conversation only to resolve vague references "
    "(for example: 'that version', 'the previous one'). Do not answer the question.\n\n"
    "Return a structured object with `rewrites` containing only rewritten queries.\n\n"
    "Question: {question}\n\n"
    "Recent conversation (latest turns):\n{history_context}"
)

QUERY_EXPANSION_DEVELOPER_PROMPT = (
    "Role: rewrite the latest user question into retrieval queries for enterprise RAG.\n"
    "Goal: improve retrieval recall while preserving user intent.\n"
    "Success criteria:\n"
    "- Preserve concrete entities, product names, dates, versions, and numbers.\n"
    "- Use prior turns only to resolve references (for example: 'that one').\n"
    "- Produce high-signal, short query rewrites; no explanations.\n"
    "Constraints:\n"
    "- Do not answer the question.\n"
    "- Do not invent facts, IDs, dates, or entities not present in the question/context.\n"
    "- Keep semantic scope consistent with the latest question.\n"
    'Output: return ONLY valid JSON: {"rewrites": ["..."]}.\n'
    "Stop rule: if the latest question is already retrieval-ready, return a single close rewrite."
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
            "You are extracting chart structure from a financial investor presentation.",
            "Extract ALL fields in the provided response schema. Be exhaustive, not selective.",
            "",
            "Extraction rules by chart type:",
            "- PIE or DONUT chart: populate pie_segments with EVERY slice.",
            "  Each slice needs its label and exact percentage value.",
            "  Do not summarize or combine segments.",
            "- BAR, COLUMN, or LINE chart: populate approx_datapoints with EVERY bar or data point.",
            "  Record the exact value for each category on each series.",
            "- legend_items: list of every legend label with its associated value or percentage if readable",
            "  (e.g., 'Boston: 34%', 'San Francisco: 18%', 'NYC: 15%').",
            "  For bar/line charts: list series labels. For pie/donut: list segment label + value.",
            "  If chart has no legend, return [].",
            "- COMPARISON MATRIX or GRID: populate matrix_cells with EVERY cell.",            "  Each cell needs its row header, column header, and cell value.",
            "- QUICK-FACTS or STAT BOX (tiled metrics panel): populate stat_box_values.",
            "  Each entry should be 'Label: value unit' (e.g. 'Properties: 179', 'WALT: 7.6 years').",
            "  Capture EVERY labeled statistic visible, including those in small tiles.",
            "",
            "General rules:",
            "- Use approximate values when exact values are not printed; mark approximate=True.",
            "- IMPORTANT: For bar and line charts where numeric values are NOT printed as labels,",
            "  estimate values from bar heights or line positions relative to the visible axis range.",
            "  Do NOT leave approx_datapoints empty just because numbers aren't printed —",
            "  read the axis scale and estimate each bar/point value. Mark approximate=True.",
            "- Do not speculate beyond visible chart evidence.",
            "- Do not summarize — list all data points individually.",
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
You are a data analyst reviewing a full investor presentation slide screenshot.
Extract ALL structured data visible. Be exhaustive, not selective.

Fill the provided structured fields:
- layout_description: describe what visual elements are present and how they relate.
- labeled_values: for EVERY labeled statistic or metric tile visible, write "Label: value unit".
  Examples: "Properties: 179", "WALT: 7.6 years", "Dividend Yield: 4.7%", "Occupancy: 89.4%".
  Include EVERY box, tile, or bullet-point metric — do not skip any.
- numeric_values: any standalone numeric values not already captured in labeled_values.
- chart_descriptions: for each chart, describe type, axes, series, and list every data point.
- pie_chart_segments: for EVERY pie or donut chart, list each slice as "label: value%".
  Example: ["Same Store: 76%", "Acquisitions: 14%", "Developments & Expansions: 10%"].
  Include ALL slices; do not combine or summarize.
- matrix_cell_values: for EVERY comparison matrix or grid, list each cell as "row | col | value".
  Example: ["Net Lease | Co-tenancy Clause | No", "Office | Co-tenancy Clause | Yes"].
  Include ALL cells.
- table_summaries: for text-based tables, summarize key rows and columns with their values.
- map_or_diagram_annotations: geographic/spatial labels, property counts, state/metro labels.
- key_takeaways: the 3–5 most important points this slide is communicating.

Be exhaustive. Do not omit any numbers, labels, percentages, or annotations visible on the page.
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
    "Role: retrieval-grounded assistant for sensitive enterprise documents.\n"
    "Personality: direct, calm, and practical. Prefer clarity over flourish.\n"
    "Collaboration style: make progress with available evidence; ask for missing fields "
    "only when they materially change correctness.\n"
    "Goal: answer the current user question using the minimum sufficient evidence.\n"
    "Success criteria:\n"
    "- Every factual claim is supported by RETRIEVAL_EVIDENCE.\n"
    "- Factual claims include inline evidence citations like [1], [2].\n"
    "- Conflicts/uncertainty are explicit instead of hidden.\n"
    "- Output strictly follows the required JSON schema.\n"
    "Constraints:\n"
    "- Use only RETRIEVAL_EVIDENCE for factual claims.\n"
    "- SESSION_SUMMARY and historical context may resolve references, not add facts.\n"
    "- Never invent source indices or cite outside evidence.\n"
    "- Prefer most current/effective version unless the user asks to compare periods/versions.\n"
    "- If conflict hints are empty, avoid absolute claims that nothing conflicts.\n"
    "Retrieval budget and stopping:\n"
    "- Use the minimum evidence sufficient for a correct answer.\n"
    "- Do not add unsupported details to make wording richer.\n"
    "- If evidence is missing for a required fact, state what is missing and continue with "
    "supported parts.\n"
    "Citation rules:\n"
    "- Cite supported factual sentences with [N] after punctuation.\n"
    "- Use one or more citations when multiple sources materially support a claim.\n"
    "- Do not group all citations in one trailing list.\n"
    "- Do not cite non-factual filler text.\n"
    "Visual evidence rules:\n"
    "- Evidence may include attached_image_indices=N,M mapped to specific evidence blocks.\n"
    "- image_scope=figure_crop is a focused artifact crop; image_scope=page_screenshot is full page.\n"
    "- Reference visuals by evidence citation [N], not by image position.\n"
    "- For bar/line charts with unlabeled values, estimate from axes and mark as approximate.\n"
    "- Do not refuse chart questions solely because exact printed labels are absent.\n"
    "Structured-data rules:\n"
    "- For tables/structured evidence, extract exact values with original units/precision.\n"
    "- If sources disagree on the same metric, say so and cite conflicting evidence.\n"
    "Output:\n"
    'Return ONLY valid JSON with exact shape {"answer":"...","reasoning":["..."]}.\n'
    '"reasoning" is optional; if present, max 5 concise public bullets (no private chain-of-thought).\n'
    'Do not include keys other than "answer" and optional "reasoning".\n'
    "Validation loop before finalizing:\n"
    "- Check citation indices are valid for provided evidence.\n"
    "- Remove unsupported claims instead of guessing."
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
