import re
from pathlib import Path

# Common REIT ticker patterns from filename (BXP, DLR, PSA, O, VICI, EGP, SPG, AMT...)
_TICKER_PATTERNS = [
    re.compile(r"^([A-Z]{2,5})[-_\s]", re.IGNORECASE),       # "BXP_Q4_..." -> BXP
    re.compile(r"[-_\s]([A-Z]{2,5})[-_\s]", re.IGNORECASE),  # "reit_BXP_2025" -> BXP
]

_DOC_TYPE_SIGNALS = {
    "investor presentation": ["investor", "presentation", "deck"],
    "fact sheet": ["fact sheet", "fact_sheet", "factsheet", "quick facts"],
    "earnings release": ["earnings", "supplement", "release", "8-k"],
    "annual report": ["annual report", "10-k", "10k"],
    "quarterly report": ["10-q", "10q", "quarterly"],
}

_EXCLUDED_TICKERS = {"REIT", "Q1", "Q2", "Q3", "Q4", "FY", "FINAL", "DRAFT"}

_SECTOR_SIGNALS = {
    "office": ["office", "workspace", "bxp", "boston properties", "kilroy"],
    "data center": ["data center", "datacenter", "digital realty", "equinix", "dlr"],
    "industrial": ["industrial", "logistics", "warehouse", "prologis", "pld"],
    "retail": ["mall", "simon", "retail", "shopping", "spg"],
    "net lease": ["net lease", "realty income", "agree"],
    "self storage": ["storage", "public storage", "extra space", "psa"],
    "diversified": ["diversified", "vici", "gaming"],
}

METRIC_PATTERNS = {
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


def _extract_ticker(stem: str) -> str | None:
    for pattern in _TICKER_PATTERNS:
        for match in pattern.finditer(stem):
            ticker = match.group(1).upper()
            if ticker not in _EXCLUDED_TICKERS:
                return ticker
    return None


def _extract_doc_type(combined: str) -> str | None:
    for doc_type, signals in _DOC_TYPE_SIGNALS.items():
        if any(signal in combined for signal in signals):
            return doc_type
    return None


def _extract_sector(combined: str) -> str | None:
    for sector, signals in _SECTOR_SIGNALS.items():
        if any(signal in combined for signal in signals):
            return sector
    return None


def extract_chunk_metric_types(text: str) -> list[str]:
    """
    Detect which financial KPI types appear in a chunk.
    Stored as metric_types: list[str] in chunk metadata.
    Used by reranker to boost metric-specific chunks.
    """
    lowered = text.lower()
    return [
        metric
        for metric, patterns in METRIC_PATTERNS.items()
        if any(p in lowered for p in patterns)
    ]
