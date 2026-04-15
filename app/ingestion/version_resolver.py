"""
Version resolver: detects version hints from filenames, document titles,
and content to support version-aware retrieval and conflict detection.
"""

import logging
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Common version patterns in filenames
VERSION_PATTERNS = [
    r"[vV](\d+(?:\.\d+)*)",  # v1, v2.1, V3
    r"[_\-\s](\d{4})[_\-\s]",  # _2024_, -2023-
    r"(?:Q[1-4])\s*(\d{4})",  # Q1 2024
    r"(\d{4})[\s_\-]?(?:Q[1-4])",  # 2024 Q1, 2024-Q2
    r"(?:FY|fy)\s*(\d{2,4})",  # FY2024, FY24
    r"(?:rev|revision|version)\s*(\d+)",  # revision 3
    r"(\d{4})[\-_](\d{2})[\-_](\d{2})",  # 2024-01-15 date
    r"(?:draft|final|updated|revised)",  # status labels
]

DATE_PATTERNS = [
    (r"(\d{4})[\-/](\d{1,2})[\-/](\d{1,2})", "%Y-%m-%d"),
    (r"(\d{1,2})[\-/](\d{1,2})[\-/](\d{4})", "%m-%d-%Y"),
    (
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}",
        None,
    ),
]

QUARTER_MAP = {"Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)}
MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def resolve_version(filename: str, content_preview: str = "") -> dict:
    """Attempt to resolve version information from filename and content.

    Args:
        filename: Name of the file being processed.
        content_preview: Optional preview of the text content to assist resolution.

    Returns:
        Dict containing version labels, date limits, and confidence.
    """
    result = {
        "version_label": None,
        "version_group": None,
        "version_rank": 0,
        "published_at": None,
        "effective_from": None,
        "effective_to": None,
        "is_current": False,
        "confidence_score": 0.0,
    }

    combined = f"{filename} {content_preview[:500]}"
    lower = combined.lower()

    # 1. Version number (v1, v2, etc.)
    if v_match := re.search(r"[vV](\d+(?:\.\d+)*)", filename):
        result["version_label"] = f"v{v_match.group(1)}"
        try:
            result["version_rank"] = int(v_match.group(1).split(".")[0])
        except ValueError:
            pass
        result["confidence_score"] = 0.8

    # 2. Quarter/Year (Q1 2024)
    if q_match := re.search(r"(Q[1-4])\s*(\d{4})", combined, re.IGNORECASE):
        quarter, year = q_match.group(1).upper(), int(q_match.group(2))
        start_m, end_m = QUARTER_MAP[quarter]
        result.update(
            {
                "version_label": result["version_label"] or f"{quarter} {year}",
                "version_rank": year * 10 + int(quarter[1]),
                "effective_from": datetime(year, start_m, 1),
                "effective_to": datetime(year, end_m, 28),
                "confidence_score": max(result["confidence_score"], 0.7),
            }
        )
    elif m_match := re.search(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+(\d{4})\b",
        combined,
        re.IGNORECASE,
    ):
        month_token = m_match.group(1).lower()
        year = int(m_match.group(2))
        month = MONTH_MAP.get(month_token)
        if month and 2000 <= year <= 2030:
            display_month = m_match.group(1).strip()
            result.update(
                {
                    "version_label": result["version_label"]
                    or f"{display_month} {year}",
                    "version_rank": year * 100 + month,
                    "effective_from": datetime(year, month, 1),
                    "effective_to": datetime(year, month, 28),
                    "confidence_score": max(result["confidence_score"], 0.65),
                }
            )

    # 3. Year-only patterns
    elif y_match := (
        re.search(r"(?:FY|fy)\s*(\d{4})", combined)
        or re.search(r"[_\-\s](\d{4})[_\-\s]", filename)
    ):
        year = int(y_match.group(1))
        if 2000 <= year <= 2030:
            result.update(
                {
                    "version_label": result["version_label"] or str(year),
                    "version_rank": year,
                    "effective_from": datetime(year, 1, 1),
                    "effective_to": datetime(year, 12, 31),
                    "confidence_score": max(result["confidence_score"], 0.5),
                }
            )

    # 4. Date in filename
    if d_match := re.search(r"(\d{4})[\-_](\d{2})[\-_](\d{2})", filename):
        try:
            result["published_at"] = datetime(
                *(int(d_match.group(i)) for i in (1, 2, 3))
            )
            result["confidence_score"] = max(result["confidence_score"], 0.6)
        except ValueError:
            pass

    # 5. Status signals
    if any(kw in lower for kw in ["final", "latest", "current", "updated"]):
        result["is_current"] = True
        result["confidence_score"] = max(result["confidence_score"], 0.6)
    elif any(
        kw in lower for kw in ["draft", "old", "archived", "previous", "superseded"]
    ):
        result["is_current"] = False

    # 6. Version group
    base = filename
    for pat in [
        r"[vV]\d+(?:\.\d+)*",
        r"Q[1-4]\s*\d{4}",
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{4}",
        r"\d{4}[\-_]\d{2}[\-_]\d{2}",
    ]:
        base = re.sub(pat, "", base, flags=re.IGNORECASE)

    base = re.sub(r"[_\-\s]+", "_", base).strip("_.")
    base = re.sub(r"\.[^.]+$", "", base)  # remove extension
    if base:
        result["version_group"] = base.lower()

    return result
