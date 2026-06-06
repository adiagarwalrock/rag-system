"""
Parser registry — single source of truth for available parsers and their config requirements.

Adding a new parser:
  1. Define a PARSER_<NAME> constant and add it to VALID_PARSERS.
  2. Add an entry in get_available_parsers() with its availability logic.
  3. Add the routing branch in parse_document() in __init__.py.
"""

from __future__ import annotations

from app.core.config import settings as _default_settings

PARSER_AUTO = "auto"
PARSER_REDUCTO = "reducto"
PARSER_LLAMA = "llamaparse"
PARSER_LAYOUT = "layout"
PARSER_LEGACY = "legacy"
PARSER_DOCLING = "docling"

VALID_PARSERS: frozenset[str] = frozenset(
    {PARSER_AUTO, PARSER_REDUCTO, PARSER_LLAMA, PARSER_LAYOUT, PARSER_LEGACY, PARSER_DOCLING}
)

_PARSER_METADATA = [
    {
        "id": PARSER_AUTO,
        "label": "Auto (recommended)",
        "description": "Tries parsers in priority order: Reducto → LlamaParse → Layout-aware PDF → Legacy.",
    },
    {
        "id": PARSER_REDUCTO,
        "label": "Reducto",
        "description": "Cloud-based high-fidelity PDF parser. Requires REDUCTO_API_KEY.",
    },
    {
        "id": PARSER_LLAMA,
        "label": "LlamaParse",
        "description": "LlamaIndex cloud parser with markdown output. Requires LLAMA_CLOUD_API_KEY.",
    },
    {
        "id": PARSER_LAYOUT,
        "label": "Layout-aware PDF",
        "description": "Local layout-aware pipeline with table/chart extraction. PDF files only.",
    },
    {
        "id": PARSER_LEGACY,
        "label": "Legacy",
        "description": "LlamaIndex built-in readers. Broadest file-type support; no layout analysis.",
    },
    {
        "id": PARSER_DOCLING,
        "label": "Docling",
        "description": "Local layout-aware parser using Docling. Supports tables, figures, and section hierarchy. PDF only.",
    },
]


def is_available(parser: str, settings_obj=None) -> bool:
    """Return True if the named parser can be invoked given current config."""
    cfg = settings_obj if settings_obj is not None else _default_settings
    if parser == PARSER_AUTO:
        return True
    if parser == PARSER_REDUCTO:
        return bool(cfg.ENABLE_EXTERNAL_PARSER and cfg.REDUCTO_API_KEY)
    if parser == PARSER_LLAMA:
        return bool(cfg.ENABLE_EXTERNAL_PARSER and cfg.LLAMAPARSE_API_KEY)
    if parser == PARSER_LAYOUT:
        return bool(cfg.ENABLE_LAYOUT_AWARE_PDF)
    if parser == PARSER_LEGACY:
        return True
    if parser == PARSER_DOCLING:
        return bool(cfg.ENABLE_DOCLING_PARSER)
    return False


def get_available_parsers(settings_obj=None) -> list[dict]:
    """Return all parsers with their current availability status."""
    cfg = settings_obj if settings_obj is not None else _default_settings
    return [
        {**meta, "available": is_available(meta["id"], cfg)}
        for meta in _PARSER_METADATA
    ]


def validate_parser_preference(preference: str | None) -> None:
    """Raise ValueError if preference is not a recognised parser ID."""
    if preference is None:
        return
    if preference not in VALID_PARSERS:
        valid = ", ".join(sorted(VALID_PARSERS))
        raise ValueError(
            f"Unknown parser '{preference}'. Valid options: {valid}."
        )
