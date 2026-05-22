import re
import time
from pathlib import Path

import httpx
from reducto import Reducto

from app.core.config import settings


def run(pdf_path: Path) -> tuple[str, float]:
    """Parse a document with Reducto; return (markdown, elapsed_seconds)."""

    client = Reducto(api_key=settings.REDUCTO_API_KEY)
    start = time.perf_counter()

    # Step 1: upload
    print(f"  [reducto] uploading {pdf_path.name} ...")
    with open(pdf_path, "rb") as f:
        upload = client.upload(file=(pdf_path.name, f, "application/pdf"))

    # Step 2: parse
    print(f"  [reducto] parsing (agentic_plus chart + table + text agents) ...")
    response = client.parse.run(
        input=upload.file_id,
        async_={"priority": True},
        # ── FORMATTING ────────────────────────────────────────────────────
        formatting={
            # TABLE OUTPUT FORMAT
            # "dynamic"  — md for simple tables, html for complex  ← API default
            # "md"       — always markdown pipe tables             ← our default (required for RAG embedding)
            # "html"     — always HTML <table>
            # "json"     — structured JSON array
            # "jsonbbox" — JSON with per-cell bounding box coordinates
            # "csv"      — CSV (flattens nested structure)
            # NOTE: "md" cannot represent merged cells — the base parser emits complex tables as
            # single <br />-joined rows (8k–13k chars). The agentic table agent always re-extracts
            # these as clean flat pipe tables. The collapsed base-parser rows are stripped in
            # post-processing below (any pipe row > 500 chars).
            "table_output_format": "md",
            "merge_tables": True,  # stitch debt maturity / portfolio tables split across pages — mirrors LlamaParse merge_continued_tables
            "add_page_markers": True,  # insert [[START OF PAGE n]] / [[END OF PAGE n]] at each page boundary — enables RAG page-level chunking and citation
            "include": [
                "hyperlinks"
            ],  # convert embedded links to markdown [text](url) — mirrors annotate_links
        },
        # ── ENHANCE — VLM-powered accuracy improvements ───────────────────
        enhance={
            "summarize_figures": True,  # VLM generates text descriptions for figures — must remain True: figure agentic scope requires it to activate (per docs)
            "intelligent_ordering": True,  # VLM corrects reading order across multi-column slide layouts
            "agentic": [
                # TABLE AGENT — VLM corrects table structure and alignment
                {
                    "scope": "table",
                    # "mode": "default",  # "default" = always run (all tables receive full enrichment) ← current default
                    #                     # "auto"    = skip tables the standard pipeline handles well (lower latency)
                    # NOTE: on May 29 2026, "default" is renamed to "max"; "auto" becomes the new default.
                    # Switch to mode: "auto" after that date unless full enrichment on every table is needed.
                    "prompt": (
                        "This is a REIT financial presentation. Follow these instructions in order:\n"
                        "1. FINANCIAL TABLES: Reconstruct with full fidelity — all column headers, row labels, "
                        "merged cells, units, currencies, bold subtotals, and footnotes exactly as written. "
                        "Normalize dates to ISO format (YYYY-MM-DD, YYYY-MM, YYYY, or YYYY-Q#) and keep the "
                        "original label alongside in the same cell.\n"
                        "2. VALUE TYPES: Label each column or value as actual, estimate, guidance, pro forma, "
                        "or target where indicated. Key metrics to preserve exactly: NOI, FFO, AFFO, NAV, "
                        "Cap rate, Occupancy, ABR, WALT, Net debt/EBITDA, leasing spreads, guidance ranges.\n"
                        "3. STRATEGY AND FRAMEWORK SLIDES with multiple pillars, columns of text, or "
                        "side-by-side comparisons: Reconstruct as a single pipe table. Use the slide's "
                        "column or pillar headers as table column headers. Each pillar or bullet group "
                        "becomes one row. Do not split into separate prose sections.\n"
                        "4. Do NOT re-extract standalone KPI icon tiles, metric summary boxes, charts, "
                        "graphs, or any visual element handled by the figure agent.\n"
                        "5. Do not flatten rows into prose. Do not hallucinate. Include all footnotes."
                    ),
                },
                # FIGURE AGENT — VLM interprets charts and figures
                {
                    "scope": "figure",
                    "advanced_chart_agent": True,  # highest-accuracy chart extraction model
                    "prompt": (
                        "This is a REIT financial presentation. Follow these instructions in order:\n"
                        "1. CHARTS AND GRAPHS (bar, line, scatter, waterfall): Output a single markdown "
                        "pipe table per chart. Use axis labels and series names as column headers. Include "
                        "every data point with exact values, units, and ISO dates alongside original labels. "
                        "Annotate CAGR labels, trend arrows, and callout boxes as extra rows or a footnote "
                        "row at the bottom. Mark visually estimated values with '(approx)'.\n"
                        "2. KPI TILES AND SUMMARY METRIC BOXES: Output exactly one pipe table per tile "
                        "using columns | Metric | Value | Unit | Period |. Never emit a bare "
                        "'metric | value | unit | period' row — pipe table format only. If the same "
                        "metrics appear in a structured data table elsewhere on the same slide, skip the "
                        "KPI tile entirely — do not re-extract it.\n"
                        "3. FLOW DIAGRAMS, STRATEGY FRAMEWORKS, LIFECYCLE DIAGRAMS, PROCESS MAPS: "
                        "Extract every component, stage, pillar, or step as a row in a structured pipe "
                        "table. Choose column headers that match the diagram's logic — for example: "
                        "| Stage | Description | Representative Properties | for lifecycle diagrams; "
                        "| Pillar | Mechanism | Examples | for strategy frameworks; "
                        "| Step | Action | Outcome | for process flows. "
                        "Never output prose bullet lists or 'Key Entities:' sections for diagrams — "
                        "the pipe table is the complete and final output.\n"
                        "4. GEOGRAPHIC MAPS AND PROPERTY MAPS: Extract all labeled entities as a pipe "
                        "table with columns | Entity | Owner / Affiliation | Location | Notes |. "
                        "No prose description.\n"
                        "5. DECORATIVE PHOTOGRAPHS (casino exteriors, resort interiors, aerial cityscapes, "
                        "people): Output exactly one line: '[Photograph: {one-line subject description}]'. "
                        "Nothing else — no bullet lists, no Key Entities sections.\n"
                        "6. ICONS, LOGOS, ARROWS, DECORATIVE GRAPHICS, AND SMALL ILLUSTRATIONS with no "
                        "extractable data: Output nothing. Do not write '[Photograph: icon...]' or any "
                        "description of the visual.\n"
                        "7. NEVER follow a table with 'Key Entities:', bullet lists, or prose paragraphs "
                        "describing the same visual. The pipe table is always the complete output.\n"
                        "8. Use actual newline characters between rows. Every row starts with '|'. "
                        "Never use backslash-n inside a table.\n"
                        "9. Key metrics: NOI, FFO, AFFO, NAV, Cap rate, Occupancy, ABR, WALT, "
                        "Net debt/EBITDA, leasing spreads, guidance ranges.\n"
                        "10. Do not invent values. Do not emit image placeholders."
                    ),
                    # "return_overlays": False,     # return overlay images for visual QA verification
                },
                # TEXT AGENT — VLM corrects handwriting / faded form text
                # Disabled: no value on clean digital PDFs; significant latency cost on dense docs.
                # Re-enable only for scanned/handwritten documents.
                # {
                #     "scope": "text",
                #     "prompt": (
                #         "This is a financial REIT investor presentation. "
                #         "Preserve all currency symbols ($, €, £), percentages (%), and basis points (bps) exactly as written. "
                #         "Do not abbreviate or reformat numeric values — keep original units (thousands, millions, billions, per share). "
                #         "Preserve section headers, footnotes, and source notes verbatim. "
                #         "Do not hallucinate or infer values not explicitly present in the text."
                #     ),
                # },
            ],
        },
        # ── RETRIEVAL — output filtering and chunking ─────────────────────
        retrieval={
            # FILTER BLOCKS — strip repeating slide headers, footers, page numbers, and cover-page title blocks
            # that inflate the duplicate line rate and pollute RAG retrieval.
            # Valid values: "Header" | "Footer" | "Title" | "Section Header" |
            #               "Page Number" | "List Item" | "Figure" | "Table" |
            #               "Key Value" | "Text" | "Comment" | "Signature"
            # NOTE: "Title" filters ALL title-classified blocks including cover logos AND slide titles.
            # Slide section names are also captured as "Section Header" blocks so document navigation
            # is preserved. If legitimate slide titles disappear after a parse run, remove "Title".
            # "filter_blocks": ["Header", "Footer", "Page Number", "Title"],
            #
            # CHUNKING — page_sections splits by page first, then by sections within each page.
            # Provides reliable page-boundary context in chunk metadata as a complement to
            # add_page_markers (which inserts [[START OF PAGE n]] inline but may be absent in agentic+ mode).
            # chunk_mode options:
            # "disabled"     — single output blob (previous default)
            # "variable"     — character-length chunks with visual boundary awareness
            # "section"      — split at section headers
            # "page"         — one chunk per page
            # "block"        — one chunk per content block
            # "page_sections"— split by page and section combined  ← current
            "chunking": {
                "chunk_mode": "page_sections",
                # "chunk_size": None,   # approx characters per chunk (None = 250–1500 auto range)
                # "chunk_overlap": 0,   # character overlap between adjacent chunks
            },
        },
        # ── SETTINGS — global parsing controls ───────────────────────────
        # settings={
        #     # EXTRACTION MODE
        #     # "hybrid" — OCR + embedded PDF text  ← default, best for digital PDFs
        #     # "ocr"    — OCR only (use for scanned docs with no embedded text layer)
        #     "extraction_mode": "hybrid",
        #
        #     # OCR SYSTEM
        #     # "standard" — best multilingual OCR  ← default
        #     # "legacy"   — germanic-language OCR only (older system)
        #     "ocr_system": "standard",
        #
        #     # PAGE RANGE — 1-indexed, inclusive
        #     # "page_range": {"start": 1, "end": None},
        #
        #     "return_images": [],        # image types to return: ["figure", "table", "page"]
        #     "return_ocr_data": False,   # include word/line-level OCR data in response
        #     "force_url_result": False,  # always return result as presigned URL (consistent for large docs)
        #     "persist_results": False,   # keep results indefinitely (default TTL = 7 days)
        #     "timeout": None,            # job timeout in seconds
        #     "document_password": None,  # decryption password for password-protected PDFs
        #     "embed_pdf_metadata": False,# embed OCR metadata back into the returned PDF
        #     # "force_file_extension": None,  # override detected file type e.g. ".png"
        # },
        # ── SPREADSHEET OPTIONS (XLSX / CSV inputs only) ─────────────────
        # spreadsheet={
        #     # CLUSTERING — table boundary detection mode
        #     # "accurate" — most powerful model, 5× the cost  ← default
        #     # "fast"     — faster, lower cost
        #     # "disabled" — no clustering
        #     "clustering": "accurate",
        #
        #     "split_large_tables": {
        #         "enabled": True,
        #         "size": 50,    # rows per chunk; or {"row": N, "column": N} for independent control
        #     },
        #     "include": [],     # extra data: ["cell_colors", "formula", "dropdowns"]
        #     "exclude": [],     # skip: ["hidden_sheets", "hidden_rows", "hidden_cols",
        #                        #        "styling", "spreadsheet_images"]
        # },
    )

    elapsed = time.perf_counter() - start

    print(f"  [reducto] done — result type: {response.result.type}")
    if response.result.type == "url":
        # Large document: result returned as URL — fetch it
        chunks = httpx.get(response.result.url, timeout=60).json().get("chunks", [])
    else:
        chunks = response.result.chunks

    parts = []
    current_page = None
    for c in chunks:
        content = c.content if hasattr(c, "content") else c.get("content", "")
        if not content:
            continue
        try:
            blocks = c.blocks if hasattr(c, "blocks") else c.get("blocks", [])
            if blocks:
                first_block = blocks[0]
                bbox = (
                    first_block.bbox
                    if hasattr(first_block, "bbox")
                    else first_block.get("bbox", {})
                )
                page = bbox.page if hasattr(bbox, "page") else bbox.get("page")
            else:
                page = None
        except (AttributeError, IndexError, KeyError):
            page = None

        if page is not None and page != current_page:
            if current_page is not None:
                parts.append(f"[[END OF PAGE {current_page}]]")
            parts.append(f"[[START OF PAGE {page}]]")
            current_page = page

        parts.append(content)

    if current_page is not None:
        parts.append(f"[[END OF PAGE {current_page}]]")

    markdown = "\n\n".join(parts)

    # ── POST-PROCESSING ───────────────────────────────────────────────────

    # 1. Strip base-parser collapsed rows: under table_output_format="md", the base parser
    #    cannot represent merged cells and emits complex tables as single >500-char pipe rows
    #    with all cell content joined via <br />. The agentic table agent always re-extracts
    #    these as clean flat pipe tables immediately after. The collapsed rows are redundant
    #    and harmful for RAG embedding — strip them along with their |-| separator lines.
    lines = markdown.splitlines()
    clean_lines = []
    skip_next_separator = False
    for line in lines:
        if line.startswith("|") and len(line) > 500:
            skip_next_separator = True  # drop the |-| separator row that follows
            continue
        if skip_next_separator and re.match(r"^\|[-|]+\|?$", line):
            skip_next_separator = False
            continue
        skip_next_separator = False
        clean_lines.append(line)
    markdown = "\n".join(clean_lines)

    # 2. Normalize escaped \n sequences that the figure agent VLM occasionally emits
    #    instead of actual newlines inside table cells (model serialization artifact).
    markdown = markdown.replace("\\n", "\n")

    return markdown, elapsed
