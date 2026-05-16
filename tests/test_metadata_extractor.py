import pytest
from app.ingestion.metadata_extractor import extract_document_metadata, extract_chunk_metric_types

def test_extract_document_metadata_ticker_from_filename():
    filename = "BXP_Q4_2024_Presentation.pdf"
    meta = extract_document_metadata(filename)
    assert meta["company_ticker"] == "BXP"
    assert meta["document_type"] == "investor presentation"
    assert meta["sector"] == "office"

def test_extract_document_metadata_ticker_mid_filename():
    filename = "reit_DLR_investor_day.pptx"
    meta = extract_document_metadata(filename)
    assert meta["company_ticker"] == "DLR"
    assert meta["document_type"] == "investor presentation"
    assert meta["sector"] == "data center"

def test_extract_document_metadata_unknown():
    filename = "random_file.pdf"
    meta = extract_document_metadata(filename)
    assert meta["company_ticker"] == ""
    assert meta["document_type"] == "unknown"
    assert meta["sector"] == "unknown"

def test_extract_chunk_metric_types():
    text = "The weighted average lease term (WALT) is 7.6 years and occupancy is 92.5%."
    metrics = extract_chunk_metric_types(text)
    assert "walt" in metrics
    assert "occupancy" in metrics
    assert "noi" not in metrics

def test_extract_chunk_metric_types_numeric_aliases():
    text = "Net Operating Income grew by 5% while ABR remained stable."
    metrics = extract_chunk_metric_types(text)
    assert "noi" in metrics
    assert "abr" in metrics
