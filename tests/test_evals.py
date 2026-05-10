from evaluation import (
    IngestionEvalCase,
    RetrievalEvalCase,
    build_quality_payload,
    evaluate_ingestion_cases,
    evaluate_retrieval_cases,
)


def test_retrieval_eval_scores_expected_hit_and_mrr():
    summary = evaluate_retrieval_cases(
        [
            RetrievalEvalCase(
                query="compare v2 and v1",
                relevant_ids=["chunk-a"],
                retrieved=[
                    {
                        "vector_node_id": "chunk-a",
                        "citation_label": "Doc A",
                        "version_label": "v2",
                    },
                    {
                        "vector_node_id": "chunk-b",
                        "citation_label": "Doc B",
                        "version_label": "v1",
                    },
                ],
                expected_version_label="v2",
                expect_conflict=False,
                expected_citation_labels=["Doc A"],
            )
        ]
    )

    assert summary.case_count == 1
    assert summary.mean_hit_rate_at_5 == 1.0
    assert summary.mean_recall_at_5 == 1.0
    assert summary.mean_mrr == 1.0
    assert summary.mean_version_attribution == 1.0
    assert summary.mean_citation_precision == 0.5


def test_retrieval_eval_penalizes_citation_overreporting():
    retrieved = [
        {"vector_node_id": f"chunk-{idx}", "citation_label": f"Source {idx}"}
        for idx in range(10)
    ]
    # Only one expected citation label should be considered precise.
    retrieved[0]["citation_label"] = "Policy v2 p.1"

    summary = evaluate_retrieval_cases(
        [
            RetrievalEvalCase(
                query="What changed in latest policy?",
                relevant_ids=["chunk-0"],
                retrieved=retrieved,
                expected_version_label="v2",
                expected_citation_labels=["Policy v2 p.1"],
            )
        ]
    )

    assert summary.case_count == 1
    assert summary.mean_citation_precision == 0.1


def test_ingestion_eval_measures_metadata_completeness():
    summary = evaluate_ingestion_cases(
        [
            IngestionEvalCase(
                document_id="doc-1",
                parse_success=True,
                chunk_count=4,
                vector_node_count=4,
                required_metadata_fields=["document_id", "client_id", "citation_label"],
                metadata_by_chunk=[
                    {
                        "document_id": "doc-1",
                        "client_id": "client-1",
                        "citation_label": "Doc 1 p.1",
                    },
                    {
                        "document_id": "doc-1",
                        "client_id": "client-1",
                        "citation_label": "Doc 1 p.2",
                    },
                ],
            )
        ]
    )

    assert summary.case_count == 1
    assert summary.parse_success_rate == 1.0
    assert summary.qdrant_index_success_rate == 1.0
    assert summary.avg_metadata_completeness == 1.0


def test_quality_payload_is_json_friendly():
    retrieval = evaluate_retrieval_cases([])
    ingestion = evaluate_ingestion_cases([])
    payload = build_quality_payload(retrieval, ingestion, extra={"mode": "dry_run"})

    assert payload["extra"]["mode"] == "dry_run"
    assert payload["retrieval"]["case_count"] == 0
    assert payload["ingestion"]["case_count"] == 0
