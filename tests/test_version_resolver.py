from datetime import datetime

from app.ingestion.version_resolver import resolve_version


def test_resolve_version_detects_semver_and_current_signals():
    result = resolve_version("policy_v2.1_final.pdf")

    assert result["version_label"] == "v2.1"
    assert result["version_rank"] == 2
    assert result["is_current"] is True
    assert result["confidence_score"] >= 0.8
    assert result["version_group"] == "policy_final"


def test_resolve_version_detects_quarter_and_effective_window():
    result = resolve_version("roadmap_Q2 2024.pptx")

    assert result["version_label"] == "Q2 2024"
    assert result["version_rank"] == 20242
    assert result["effective_from"] == datetime(2024, 4, 1)
    assert result["effective_to"] == datetime(2024, 6, 28)


def test_resolve_version_detects_fiscal_year_patterns():
    result = resolve_version("budget_FY2023.pdf")

    assert result["version_label"] == "2023"
    assert result["version_rank"] == 2023
    assert result["effective_from"] == datetime(2023, 1, 1)
    assert result["effective_to"] == datetime(2023, 12, 31)


def test_resolve_version_extracts_published_date_from_filename():
    result = resolve_version("policy_2024-01-15.pdf")

    assert result["published_at"] == datetime(2024, 1, 15)
    assert result["confidence_score"] >= 0.6


def test_resolve_version_normalizes_version_group_name():
    result = resolve_version("Revenue-Report_Q1 2025_v3.pdf")

    assert result["version_group"].startswith("revenue_report")


def test_resolve_version_keeps_draft_as_not_current():
    result = resolve_version("plan_draft_v1.docx")

    assert result["version_label"] == "v1"
    assert result["is_current"] is False
