"""
Version resolver: detects version hints from filenames, document titles,
and content to support version-aware retrieval and conflict detection.
"""

import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

CONTENT_PREVIEW_LIMIT = 500
YEAR_MIN = 2000
YEAR_MAX = 2030
QUARTER_RANK_MULTIPLIER = 10
MONTH_RANK_MULTIPLIER = 100
FIRST_DAY_OF_PERIOD = 1
LAST_DAY_OF_PERIOD = 28

VERSION_NUMBER_PATTERN = r"[vV](\d+(?:\.\d+)*)"
QUARTER_YEAR_PATTERN = r"(Q[1-4])\s*(\d{4})"
MONTH_YEAR_PATTERN = (
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+(\d{4})\b"
)
SHORT_MONTH_YEAR_PATTERN = (
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[\-_ ]+(\d{2,4})\b"
)
FISCAL_YEAR_PATTERN = r"(?:FY|fy)\s*(\d{4})"
YEAR_TOKEN_PATTERN = r"[_\-\s](\d{4})[_\-\s]"
FILENAME_DATE_PATTERN = r"(\d{4})[\-_](\d{2})[\-_](\d{2})"
DOTTED_DATE_PATTERN = r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b"
WHITESPACE_SEPARATOR_PATTERN = r"[_\-\s]+"
FILE_EXTENSION_PATTERN = r"\.[^.]+$"

CURRENT_STATUS_KEYWORDS = ("final", "latest", "current", "updated")
STALE_STATUS_KEYWORDS = ("draft", "old", "archived", "previous", "superseded")

# Common version patterns in filenames
VERSION_PATTERNS = [
    VERSION_NUMBER_PATTERN,  # v1, v2.1, V3
    YEAR_TOKEN_PATTERN,  # _2024_, -2023-
    QUARTER_YEAR_PATTERN,  # Q1 2024
    r"(\d{4})[\s_\-]?(?:Q[1-4])",  # 2024 Q1, 2024-Q2
    r"(?:FY|fy)\s*(\d{2,4})",  # FY2024, FY24
    r"(?:rev|revision|version)\s*(\d+)",  # revision 3
    FILENAME_DATE_PATTERN,  # 2024-01-15 date
    DOTTED_DATE_PATTERN,  # 3.20.2026 date
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
MONTH_DISPLAY = {
    "jan": "January",
    "feb": "February",
    "mar": "March",
    "apr": "April",
    "may": "May",
    "jun": "June",
    "jul": "July",
    "aug": "August",
    "sep": "September",
    "sept": "September",
    "oct": "October",
    "nov": "November",
    "dec": "December",
}


_INVESTOR_DAY_TOKENS = ("investor day", "investor event", "analyst day", "annual meeting")
_MERGER_TOKENS = ("merger", "acquisition", "transaction", "business combination")
_QUARTERLY_UPDATE_TOKENS = (
    "quarterly", "company update", "earnings update",
)
# Matches q1/q2/q3/q4 surrounded by word boundaries (handles hyphens and spaces)
_QUARTER_PATTERN = re.compile(r"\bq[1-4]\b")


def _infer_document_type(combined_lower: str) -> str | None:
    """Classify document as investor-day, merger-presentation, quarterly-update, or None."""
    if any(t in combined_lower for t in _INVESTOR_DAY_TOKENS):
        return "investor-day"
    if any(t in combined_lower for t in _MERGER_TOKENS):
        return "merger-presentation"
    if any(t in combined_lower for t in _QUARTERLY_UPDATE_TOKENS) or _QUARTER_PATTERN.search(
        combined_lower
    ):
        return "quarterly-update"
    return None


def _new_version_result() -> dict:
    return {
        "version_label": None,
        "version_group": None,
        "version_rank": 0,
        "published_at": None,
        "effective_from": None,
        "effective_to": None,
        "is_current": False,
        "confidence_score": 0.0,
        "document_type": None,
    }


def _apply_version_number(filename: str, result: dict) -> None:
    if not (v_match := re.search(VERSION_NUMBER_PATTERN, filename)):
        return

    result["version_label"] = f"v{v_match.group(1)}"
    try:
        result["version_rank"] = int(v_match.group(1).split(".")[0])
    except ValueError:
        pass
    result["confidence_score"] = 0.8


def _apply_quarter_year(combined: str, result: dict) -> bool:
    if not (q_match := re.search(QUARTER_YEAR_PATTERN, combined, re.IGNORECASE)):
        return False

    quarter, year = q_match.group(1).upper(), int(q_match.group(2))
    start_m, end_m = QUARTER_MAP[quarter]
    result.update(
        {
            "version_label": result["version_label"] or f"{quarter} {year}",
            "version_rank": year * QUARTER_RANK_MULTIPLIER + int(quarter[1]),
            "effective_from": datetime(year, start_m, FIRST_DAY_OF_PERIOD),
            "effective_to": datetime(year, end_m, LAST_DAY_OF_PERIOD),
            "confidence_score": max(result["confidence_score"], 0.7),
        }
    )
    return True


def _apply_month_year(combined: str, result: dict) -> bool:
    if not (m_match := re.search(MONTH_YEAR_PATTERN, combined, re.IGNORECASE)):
        return False

    month_token = m_match.group(1).lower()
    year = int(m_match.group(2))
    month = MONTH_MAP.get(month_token)
    if not month or not (YEAR_MIN <= year <= YEAR_MAX):
        return False

    display_month = m_match.group(1).strip()
    result.update(
        {
            "version_label": result["version_label"] or f"{display_month} {year}",
            "version_rank": year * MONTH_RANK_MULTIPLIER + month,
            "effective_from": datetime(year, month, FIRST_DAY_OF_PERIOD),
            "effective_to": datetime(year, month, LAST_DAY_OF_PERIOD),
            "confidence_score": max(result["confidence_score"], 0.65),
        }
    )
    return True


def _apply_short_month_year(combined: str, result: dict) -> bool:
    if not (m_match := re.search(SHORT_MONTH_YEAR_PATTERN, combined, re.IGNORECASE)):
        return False

    month_token = m_match.group(1).lower()
    year = _normalize_year(int(m_match.group(2)))
    month = MONTH_MAP.get(month_token)
    if not month or not (YEAR_MIN <= year <= YEAR_MAX):
        return False

    display_month = MONTH_DISPLAY.get(month_token, m_match.group(1).strip())
    result.update(
        {
            "version_label": result["version_label"] or f"{display_month} {year}",
            "version_rank": year * MONTH_RANK_MULTIPLIER + month,
            "effective_from": datetime(year, month, FIRST_DAY_OF_PERIOD),
            "effective_to": datetime(year, month, LAST_DAY_OF_PERIOD),
            "confidence_score": max(result["confidence_score"], 0.65),
        }
    )
    return True


def _normalize_year(year: int) -> int:
    if year < 100:
        return 2000 + year
    return year


def _apply_year_only(filename: str, combined: str, result: dict) -> None:
    y_match = re.search(FISCAL_YEAR_PATTERN, combined) or re.search(
        YEAR_TOKEN_PATTERN, filename
    )
    if not y_match:
        return

    year = int(y_match.group(1))
    if not (YEAR_MIN <= year <= YEAR_MAX):
        return

    result.update(
        {
            "version_label": result["version_label"] or str(year),
            "version_rank": year,
            "effective_from": datetime(year, 1, FIRST_DAY_OF_PERIOD),
            "effective_to": datetime(year, 12, 31),
            "confidence_score": max(result["confidence_score"], 0.5),
        }
    )


def _apply_filename_date(filename: str, result: dict) -> None:
    if d_match := re.search(FILENAME_DATE_PATTERN, filename):
        try:
            result["published_at"] = datetime(
                *(int(d_match.group(i)) for i in (1, 2, 3))
            )
            result["confidence_score"] = max(result["confidence_score"], 0.6)
        except ValueError:
            pass
        return

    if d_match := re.search(DOTTED_DATE_PATTERN, filename):
        try:
            month = int(d_match.group(1))
            day = int(d_match.group(2))
            year = _normalize_year(int(d_match.group(3)))
            result["published_at"] = datetime(year, month, day)
            result["confidence_score"] = max(result["confidence_score"], 0.6)
        except ValueError:
            pass


def _apply_status_signals(lower_combined: str, result: dict) -> None:
    if any(keyword in lower_combined for keyword in CURRENT_STATUS_KEYWORDS):
        result["is_current"] = True
        result["confidence_score"] = max(result["confidence_score"], 0.6)
        return
    if any(keyword in lower_combined for keyword in STALE_STATUS_KEYWORDS):
        result["is_current"] = False


def _build_version_group(filename: str) -> str:
    base = filename
    for pattern in [
        VERSION_NUMBER_PATTERN,
        QUARTER_YEAR_PATTERN,
        MONTH_YEAR_PATTERN,
        SHORT_MONTH_YEAR_PATTERN,
        FILENAME_DATE_PATTERN,
        DOTTED_DATE_PATTERN,
    ]:
        base = re.sub(pattern, "", base, flags=re.IGNORECASE)

    base = re.sub(WHITESPACE_SEPARATOR_PATTERN, "_", base).strip("_.")
    return re.sub(FILE_EXTENSION_PATTERN, "", base).lower()


def resolve_version(filename: str, content_preview: str = "") -> dict:
    """Attempt to resolve version information from filename and content.

    Args:
        filename: Name of the file being processed.
        content_preview: Optional preview of the text content to assist resolution.

    Returns:
        Dict containing version labels, date limits, and confidence.
    """
    result = _new_version_result()

    combined = f"{filename} {content_preview[:CONTENT_PREVIEW_LIMIT]}"
    lower = combined.lower()

    _apply_version_number(filename, result)
    has_quarter = _apply_quarter_year(combined, result)
    if not has_quarter:
        has_month = _apply_month_year(combined, result)
        if not has_month:
            has_short_month = _apply_short_month_year(combined, result)
            if not has_short_month:
                _apply_year_only(filename, combined, result)

    _apply_filename_date(filename, result)
    _apply_status_signals(lower, result)
    version_group = _build_version_group(filename)
    if version_group:
        result["version_group"] = version_group

    result["document_type"] = _infer_document_type(lower)

    return result
