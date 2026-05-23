import time
from pathlib import Path


def run(pdf_path: Path) -> tuple[str, float]:
    """Parse a document with LlamaParse; return (markdown, elapsed_seconds)."""
    from llama_cloud import LlamaCloud

    from app.core.config import settings

    client = LlamaCloud(api_key=settings.LLAMAPARSE_API_KEY)
    start = time.perf_counter()

    # Step 1: upload the file
    print(f"  [llamaparse] uploading {pdf_path.name} ...")
    file = client.files.create(file=str(pdf_path), purpose="parse")

    # Step 2: parse (SDK blocks until the job finishes)
    print(f"  [llamaparse] parsing (agentic_plus, blocking) ...")
    result = client.parsing.parse(
        file_id=file.id,
        # ── TIER ─────────────────────────────────────────────────────────
        # "fast"           — rule-based, cheapest, NO markdown output
        # "cost_effective" — balanced speed/quality, supports custom_prompt
        # "agentic"        — AI-powered, handles complex layouts  ← default
        # "agentic_plus"   — highest accuracy, ~4.5× cost of agentic
        tier="agentic_plus",
        # ── VERSION — pin for reproducibility, or use "latest" ───────────
        # "latest" | "2026-05-13" | "2026-05-11" | "2026-04-09" | "2025-12-11"
        version="latest",
        # ── EXPAND — which result fields to return ────────────────────────
        # "markdown"               — per-page markdown            ← always include
        # "text"                   — per-page plain text
        # "items"                  — structured layout tree with bounding boxes
        # "metadata"               — per-page confidence scores, cost_optimized flag
        # "images_content_metadata"— image list with presigned download URLs
        # "xlsx_content_metadata"  — tables as downloadable XLSX file
        expand=["markdown"],
        # ── PAGE SELECTION ────────────────────────────────────────────────
        # page_ranges={
        #     "max_pages": None,        # int — cap total pages processed from page 1
        #     "target_pages": None,     # str — e.g. "1,3,5-10" (1-based, inclusive)
        # },
        # ── CROP BOX — strip page margins (ratios 0.0–1.0 of page size) ──
        # crop_box={
        #     "top": 0.0,
        #     "bottom": 0.0,
        #     "left": 0.0,
        #     "right": 0.0,
        # },
        # ── CACHE ─────────────────────────────────────────────────────────
        # disable_cache=False,   # True = force re-parse, bypass result cache
        # ── INPUT OPTIONS ─────────────────────────────────────────────────
        # input_options={
        #     # PDF has no sub-fields currently.
        #     # "pdf": {},
        #
        #     # Presentation options — relevant if parsing PPTX files:
        #     # "presentation": {
        #     #     "out_of_bounds_content": False,  # extract content outside slide area
        #     #     "skip_embedded_data": False,     # skip chart data tables in slides
        #     # },
        #
        #     # Spreadsheet options — relevant if parsing XLSX/CSV files:
        #     # "spreadsheet": {
        #     #     "detect_sub_tables_in_sheets": False,         # find multiple tables per sheet
        #     #     "force_formula_computation_in_sheets": False, # compute formulas, not formula text
        #     #     "include_hidden_sheets": False,               # also parse hidden sheets
        #     # },
        #
        #     # HTML options — relevant if parsing .html files:
        #     # "html": {
        #     #     "make_all_elements_visible": False,   # override CSS display/visibility
        #     #     "remove_navigation_elements": False,  # strip nav bars, sidebars
        #     #     "remove_fixed_elements": False,       # strip sticky headers/footers
        #     # },
        # },
        # ── OUTPUT OPTIONS ────────────────────────────────────────────────
        output_options={
            "markdown": {
                "tables": {
                    "output_tables_as_markdown": True,  # False = HTML <table> tags
                    "merge_continued_tables": True,  # stitch tables split across pages
                    "compact_markdown_tables": True,  # remove whitespace padding in cells
                    # "markdown_table_multiline_separator": " ",  # join multi-line cells; e.g. " " or "<br>"
                },
                "annotate_links": True,  # include [text](url) link destinations
                "inline_images": True,  # transcribe figures/charts into md instead of ![...](img) refs
            },
            "spatial_text": {
                "do_not_unroll_columns": True,  # keep multi-column layout intact
                "preserve_very_small_text": True,  # capture footnotes, fine print, debt covenants, NAV assumptions
                # "preserve_layout_alignment_across_pages": True,  # auto-on for agentic tier
            },
            # "extract_printed_page_number": False,   # capture "Page X of Y" printed labels
            # "images_to_save": [],    # ["screenshot", "embedded", "layout"]
            #                          # screenshot = full-page renders
            #                          # embedded   = images inside the document
            #                          # layout     = figure/diagram crops from layout detection
            # "tables_as_spreadsheet": {
            #     "enable": False,           # export each table as a sheet in an XLSX file
            #     "guess_sheet_name": False, # auto-name sheets from surrounding table context
            # },
            # "additional_outputs": [],  # extra artifacts saved alongside markdown:
            #                            # "stripped_md"              — formatting-stripped md per page (for search indexing)
            #                            # "concatenated_stripped_txt" — all pages as one plain-text blob (for embedding)
            #                            # "word_bbox"                 — word-level bounding boxes JSONL (for answer grounding)
        },
        # ── PROCESSING OPTIONS ────────────────────────────────────────────
        processing_options={
            # CHART PARSING
            # "efficient"    — fast rule-based chart reading
            # "agentic"      — AI-powered chart data extraction
            # "agentic_plus" — highest accuracy chart extraction  ← our default
            "specialized_chart_parsing": "agentic_plus",
            "aggressive_table_extraction": True,  # detect borderless/implicit tables (may add false positives)
            # "disable_heuristics": False,         # turn off outlined-table detection and adaptive long-table handling
            # COST OPTIMIZER — incompatible with auto_mode_configuration; keep disabled.
            # Re-enable only if auto_mode_configuration is removed.
            # "cost_optimizer": {
            #     "enable": True,
            # },
            # IGNORE — skip noisy content types
            # "ignore": {
            #     "ignore_diagonal_text": False,  # skip CONFIDENTIAL/DRAFT diagonal watermarks
            #     "ignore_hidden_text": False,    # skip invisible/white-on-white text layers
            #     "ignore_text_in_image": False,  # skip OCR on embedded raster images
            # },
            # OCR LANGUAGES — order matters; first language = primary
            "ocr_parameters": {
                "languages": ["en"],  # e.g. ["en", "fr", "de"]
            },
            # AUTO MODE CONFIGURATION — per-page conditional rules
            # Apply different tiers, prompts, and settings per page based on page content.
            # Each entry = one rule with triggers + a parsing_conf to apply when triggered.
            # Available triggers (set trigger_mode to "and"/"or"):
            #   "table_in_page": True,                        # page contains at least one table
            #   "image_in_page": True,                        # page contains non-screenshot images
            #   "full_page_image_in_page": True,              # page is a scanned image
            #   "page_contains_at_least_n_charts": 1,
            #   "page_contains_at_least_n_tables": 1,
            #   "page_contains_at_least_n_numbers": 10,
            #   "page_contains_at_least_n_percent_numbers": 3,
            #   "page_contains_at_least_n_words": 50,
            #   "page_contains_at_most_n_words": 200,
            #   "page_longer_than_n_chars": 500,
            #   "page_shorter_than_n_chars": 100,
            #   "regexp_in_page": r"\$[\d,]+",               # regex match against page text
            #   "text_in_page": "CONFIDENTIAL",              # substring match
            #   "trigger_mode": "or",  # "and" = all conditions must match, "or" = any
            # Available parsing_conf keys:
            #   "tier", "version", "specialized_chart_parsing", "aggressive_table_extraction",
            #   "custom_prompt", "high_res_ocr", "ignore", "spatial_text", "crop_box"
            "auto_mode_configuration": [
                {
                    # Trigger on any page with at least one detected chart
                    "page_contains_at_least_n_charts": 1,
                    "parsing_conf": {
                        "tier": "agentic_plus",
                        "version": "latest",
                        "specialized_chart_parsing": "agentic_plus",
                        "custom_prompt": (
                            "This page contains charts. For every chart, output a markdown table: "
                            "axis labels and series names as column headers, all data points with exact values and units. "
                            "For KPI tiles and summary boxes, output: metric | value | unit | period. "
                            "Key REIT metrics: NOI, FFO, AFFO, NAV, Cap rate, Occupancy, ABR, WALT, Net debt/EBITDA, leasing spreads, guidance ranges. "
                            "Append '(approx)' for values estimated from the visual. Never emit image placeholders."
                        ),
                    },
                },
            ],
        },
        # ── AGENTIC OPTIONS (cost_effective / agentic / agentic_plus only) ──
        agentic_options={
            "custom_prompt": (
                "You are a specialized REIT document parser. "
                "Extract all financial metrics, tables, charts, and visual data with precise attention to:\n\n"
                "1. DATE NORMALIZATION: Convert all dates to ISO format (YYYY-MM-DD, YYYY-MM, YYYY, or YYYY-Q# for quarters). Preserve original labels alongside normalized dates.\n\n"
                "2. TABLE PRESERVATION: Extract tables with full structure including headers, row labels, units, currencies, and footnotes. Do not flatten tables into prose.\n\n"
                "3. VISUAL EXTRACTION: Convert all charts, graphs, maps, KPI tiles, and diagrams into structured data. For charts with visible values, extract exact numbers. For approximate values from visual estimation, mark is_approximate as true.\n\n"
                "4. REIT METRICS: Pay special attention to: NOI, Same-store NOI, FFO, Core FFO, AFFO, NAV, Cap rate, Occupancy, Leasing spreads, Rent growth, ABR, WALT, Debt maturity, Net debt/EBITDA, Interest coverage, Development pipeline, Property count, GLA/square footage, Tenant concentration, Sector exposure, Geographic exposure, Dividend metrics, Guidance ranges.\n\n"
                "5. VALUE TYPES: Distinguish between actuals, estimates, guidance, pro forma, and targets. Preserve units (thousands, millions, billions, per share, percentage, basis points, square feet).\n\n"
                "6. DO NOT HALLUCINATE: Only extract values explicitly present in the document. Mark uncertain extractions appropriately."
            ),
        },
        # ── PROCESSING CONTROL ────────────────────────────────────────────
        # processing_control={
        #     "timeouts": {
        #         "base_in_seconds": 300,               # base job timeout in seconds (max 1800)
        #         "extra_time_per_page_in_seconds": 10, # added per page (max 300)
        #         # total timeout = base + (extra_per_page × page_count)
        #     },
        #     "job_failure_conditions": {
        #         "allowed_page_failure_ratio": 0.05,           # max ratio of failed pages (0–1)
        #         "fail_on_buggy_font": False,                  # fail if problematic font detected
        #         "fail_on_image_extraction_error": False,      # fail on image extraction errors
        #         "fail_on_image_ocr_error": False,             # fail on OCR errors
        #         "fail_on_markdown_reconstruction_error": False, # fail if markdown can't be built
        #     },
        # },
    )

    elapsed = time.perf_counter() - start

    pages = result.markdown.pages if result.markdown else []
    print(f"  [llamaparse] done — {len(pages)} pages extracted")
    parts = []
    for p in pages:
        if not (hasattr(p, "markdown") and p.markdown):
            continue
        page_num = p.page_number if hasattr(p, "page_number") else None
        if page_num is not None:
            parts.append(f"[[START OF PAGE {page_num}]]")
        parts.append(p.markdown)
        if page_num is not None:
            parts.append(f"[[END OF PAGE {page_num}]]")
    markdown = "\n\n".join(parts)
    return markdown, elapsed
