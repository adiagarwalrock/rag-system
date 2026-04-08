from app.ingestion.parser import (
    _has_chart_signals,
    _has_numeric_data,
    _has_table_signals,
    parse_document,
)


def test_has_table_signals_detects_multi_line_table_patterns():
    text = "col1|col2|col3\n1|2|3\nplain text"
    assert _has_table_signals(text) is True


def test_has_table_signals_ignores_single_line_delimiters():
    text = "just one line with a|b|c"
    assert _has_table_signals(text) is False


def test_has_chart_signals_detects_common_chart_keywords():
    assert _has_chart_signals("Figure 3 shows quarterly growth.") is True
    assert _has_chart_signals("No visual references here") is False


def test_has_numeric_data_requires_at_least_three_numbers():
    assert _has_numeric_data("values: 10, 20, and 30") is True
    assert _has_numeric_data("values: 10 and 20") is False


def test_parse_document_fallback_text_path_emits_unit_metadata(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "Revenue was 10 in Q1, 20 in Q2, and 30 in Q3.", encoding="utf-8"
    )

    metadata = {
        "document_id": "doc-123",
        "client_id": "client-1",
        "client_name": "Acme",
    }

    docs, units = parse_document(str(file_path), metadata)

    assert len(docs) == 1
    assert len(units) == 1
    assert docs[0].metadata["page_num"] == 1
    assert docs[0].metadata["source_file"] == "sample.txt"
    assert units[0]["document_id"] == "doc-123"
    assert units[0]["unit_type"] == "section"
    assert units[0]["page_num"] == 1
    assert units[0]["contains_numeric_data"] is True
