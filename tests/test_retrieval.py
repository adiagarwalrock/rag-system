import app.retrieval.retriever as retriever_module
from types import SimpleNamespace

from llama_index.core.base.llms.types import TextBlock, ThinkingBlock
from llama_index.core.schema import NodeWithScore, TextNode

from app.retrieval.citation_builder import build_citations
from app.retrieval.query_expansion import should_expand_query
from app.retrieval.retriever import (
    VecteraRetriever,
    _build_conversation_context_block,
    _build_retrieval_diagnostics,
    _collect_image_evidence_paths,
    _extract_answer_and_reasoning_from_chat,
    _fuse_node_batches,
    _split_reasoning_from_text,
)


def _node(node_id: str, score: float, metadata: dict | None = None) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            id_=node_id, text=f"source text for {node_id}", metadata=metadata or {}
        ),
        score=score,
    )


def test_query_expansion_routes_broad_and_version_questions_only():
    assert should_expand_query("Compare the current version against the older version")
    assert should_expand_query("Summarize trends across the uploaded reports")
    assert should_expand_query("Show me the chart for quarterly revenue")
    assert should_expand_query("How did customer count change over time?")
    assert should_expand_query(
        {
            "current_question": "What about that one?",
            "recent_turns": [
                {"role": "user", "content": "Compare v1 and v2 policy changes."},
                {
                    "role": "assistant",
                    "content": "v1 had 30 days, v2 had 45 days retention.",
                },
            ],
        }
    )
    assert not should_expand_query("What is the renewal deadline?")


def test_retrieve_with_mode_passes_recent_turns_in_expansion_question(monkeypatch):
    retriever = VecteraRetriever(
        "client-1",
        top_k=5,
        conversation_context={
            "recent_turns": [
                {"role": "user", "content": "Show v1 policy"},
                {"role": "assistant", "content": "v1 has 30 days"},
            ]
        },
    )

    captured: dict = {}

    def _fake_should_expand_query(question_payload):
        captured["payload"] = question_payload
        return False

    class FakeBaseRetriever:
        def retrieve(self, _question):
            return []

    monkeypatch.setattr(
        retriever_module.vector_store_manager,
        "get_retriever",
        lambda **kwargs: FakeBaseRetriever(),
    )
    monkeypatch.setattr(
        retriever_module, "should_expand_query", _fake_should_expand_query
    )

    nodes, expanded = retriever._retrieve_with_mode("What about that?", hybrid=True)

    assert nodes == []
    assert expanded is False
    assert captured["payload"]["current_question"] == "What about that?"
    assert captured["payload"]["recent_turns"][0]["role"] == "user"
    assert captured["payload"]["recent_turns"][1]["role"] == "assistant"


def test_retrieve_with_mode_uses_history_aware_query_variants(monkeypatch):
    retriever = VecteraRetriever(
        "client-1",
        top_k=5,
        conversation_context={
            "recent_turns": [
                {"role": "user", "content": "Compare policy v1 and v2."},
                {"role": "assistant", "content": "v2 changed retention days."},
            ]
        },
    )

    retrieval_calls: list[str] = []
    captured: dict = {}

    class FakeBaseRetriever:
        def retrieve(self, question):
            retrieval_calls.append(question)
            index = len(retrieval_calls)
            return [
                _node(
                    f"node-{index}",
                    0.9,
                    {"document_id": f"doc-{index}", "citation_label": f"Doc {index}"},
                )
            ]

    def _fake_build_query_variants(question_payload, max_rewrites=2):
        captured["payload"] = question_payload
        captured["max_rewrites"] = max_rewrites
        return [
            "What about that?",
            "Compare policy v2 retention against v1",
        ]

    monkeypatch.setattr(
        retriever_module.vector_store_manager,
        "get_retriever",
        lambda **kwargs: FakeBaseRetriever(),
    )
    monkeypatch.setattr(retriever_module, "should_expand_query", lambda _payload: True)
    monkeypatch.setattr(
        retriever_module, "build_query_variants", _fake_build_query_variants
    )

    nodes, expanded = retriever._retrieve_with_mode("What about that?", hybrid=True)

    assert expanded is True
    assert len(nodes) == 2
    assert retrieval_calls == [
        "What about that?",
        "Compare policy v2 retention against v1",
    ]
    assert captured["payload"]["current_question"] == "What about that?"
    assert captured["payload"]["recent_turns"][0]["role"] == "user"
    assert captured["max_rewrites"] == 2


def test_fuse_node_batches_dedupes_and_boosts_repeated_nodes():
    first = _node("same", 0.4)
    second = _node("other", 0.7)
    duplicate = _node("same", 0.5)

    fused = _fuse_node_batches([[first, second], [duplicate]])

    assert [node.node.node_id for node in fused].count("same") == 1
    assert fused[0].node.node_id == "same"
    assert fused[0].score > second.score


def test_latest_query_prefers_current_version_nodes():
    retriever = VecteraRetriever("client-1", top_k=5)

    current = _node(
        "current",
        0.5,
        {
            "version_rank": "12",
            "is_current": "true",
            "version_label": "v2",
            "document_version_group": "policy",
            "document_id": "doc-v2",
        },
    )
    old = _node(
        "old",
        0.5,
        {
            "version_rank": "1",
            "is_current": "false",
            "version_label": "v1",
            "document_version_group": "policy",
            "document_id": "doc-v1",
        },
    )

    ranked = retriever._rank_nodes("What is the latest policy version?", [old, current])
    assert ranked[0].node.node_id == "current"


def test_latest_query_promotes_all_chunks_from_newest_version_group():
    retriever = VecteraRetriever("client-1", top_k=5)

    old_exact = _node(
        "dec-customers",
        0.52,
        {
            "version_rank": "202512",
            "is_current": "false",
            "version_label": "December 2025",
            "document_version_group": "digital-realty-investor-presentation",
            "document_id": "doc-dec",
        },
    )
    newest_other = _node(
        "mar-other",
        0.9,
        {
            "version_rank": "202603",
            "is_current": "false",
            "version_label": "March 2026",
            "document_version_group": "digital-realty-investor-presentation",
            "document_id": "doc-mar",
        },
    )
    newest_exact = _node(
        "mar-customers",
        0.35,
        {
            "version_rank": "202603",
            "is_current": "false",
            "version_label": "March 2026",
            "document_version_group": "digital-realty-investor-presentation",
            "document_id": "doc-mar",
        },
    )
    old_exact.node.text = "5,000+ Customers"
    newest_other.node.text = "Customer type (% by ARR). Top customers by revenue."
    newest_exact.node.text = "5,500+ Customers"

    ranked = retriever._rank_nodes(
        "How many customers does Digital Realty have?",
        [old_exact, newest_other, newest_exact],
    )
    ranked_ids = [node.node.node_id for node in ranked]

    assert ranked_ids.index("mar-customers") < ranked_ids.index("dec-customers")


def test_citations_are_bounded_subset_of_ranked_candidates():
    retriever = VecteraRetriever("client-1", top_k=10)
    ranked = [
        _node(
            f"node-{i}",
            1.0 - (i * 0.01),
            {
                "version_label": "v2",
                "document_id": f"doc-{i}",
                "citation_label": f"Doc {i}",
            },
        )
        for i in range(10)
    ]

    evidence_nodes = retriever._select_evidence_nodes(
        "What is the renewal deadline?", ranked
    )
    citations = build_citations(evidence_nodes)

    assert len(ranked) == 10
    assert len(citations) <= retriever.evidence_limit

    ranked_ids = {node.node.node_id for node in ranked}
    citation_ids = {citation["vector_node_id"] for citation in citations}
    assert citation_ids <= ranked_ids


def test_comparison_queries_select_multiple_versions_for_citations():
    retriever = VecteraRetriever("client-1", top_k=10)
    nodes = [
        _node(
            "v2-primary",
            0.92,
            {
                "version_rank": "12",
                "is_current": "true",
                "version_label": "v2",
                "document_version_group": "policy",
                "document_id": "doc-v2",
            },
        ),
        _node(
            "v2-secondary",
            0.88,
            {
                "version_rank": "12",
                "is_current": "true",
                "version_label": "v2",
                "document_version_group": "policy",
                "document_id": "doc-v2-2",
            },
        ),
        _node(
            "v1-primary",
            0.84,
            {
                "version_rank": "10",
                "is_current": "false",
                "version_label": "v1",
                "document_version_group": "policy",
                "document_id": "doc-v1",
            },
        ),
    ]

    ranked = retriever._rank_nodes("Compare changes between v1 and v2", nodes)
    evidence_nodes = retriever._select_evidence_nodes(
        "Compare changes between v1 and v2", ranked
    )
    citations = build_citations(evidence_nodes)
    versions = {citation.get("version_label") for citation in citations}

    assert "v1" in versions
    assert "v2" in versions


def test_comparison_query_deprioritizes_reasoning_chunks_for_factual_delta_questions():
    retriever = VecteraRetriever("client-1", top_k=10)
    ranked = [
        _node(
            "reasoning-v2",
            0.95,
            {
                "chunk_type": "reasoning_chart",
                "version_label": "v2",
                "document_id": "doc-v2",
                "document_version_group": "policy",
            },
        ),
        _node(
            "reasoning-v1",
            0.93,
            {
                "chunk_type": "reasoning_chart",
                "version_label": "v1",
                "document_id": "doc-v1",
                "document_version_group": "policy",
            },
        ),
        _node(
            "factual-v2",
            0.9,
            {
                "chunk_type": "body_text",
                "version_label": "v2",
                "document_id": "doc-v2",
                "document_version_group": "policy",
            },
        ),
        _node(
            "factual-v1",
            0.88,
            {
                "chunk_type": "body_text",
                "version_label": "v1",
                "document_id": "doc-v1",
                "document_version_group": "policy",
            },
        ),
    ]

    evidence_nodes = retriever._select_evidence_nodes(
        "How much did customer count change between v1 and v2?",
        ranked,
    )

    selected_ids = [node.node.node_id for node in evidence_nodes]
    assert selected_ids[:2] == ["factual-v2", "factual-v1"]


def test_retrieval_diagnostics_reports_reasoning_and_document_diversity():
    ranked = [
        _node(
            "reasoning-1",
            0.8,
            {"chunk_type": "reasoning_chart", "document_id": "doc-1"},
        ),
        _node("body-1", 0.7, {"chunk_type": "body_text", "document_id": "doc-1"}),
        _node("body-2", 0.6, {"chunk_type": "body_text", "document_id": "doc-2"}),
    ]
    evidence = [ranked[1], ranked[2]]

    diagnostics = _build_retrieval_diagnostics(ranked, evidence)

    assert diagnostics["ranked_document_count"] == 2
    assert diagnostics["evidence_document_count"] == 2
    assert diagnostics["ranked_reasoning_count"] == 1
    assert diagnostics["evidence_reasoning_count"] == 0
    assert diagnostics["ranked_image_chunk_count"] == 0
    assert diagnostics["evidence_image_chunk_count"] == 0


def test_conflict_query_diversifies_evidence_when_versions_are_missing():
    retriever = VecteraRetriever("client-1", top_k=10)
    ranked = [
        _node(
            "doc-a-primary",
            0.95,
            {
                "document_id": "doc-a",
                "document_version_group": "group-a",
                "contains_numeric_data": True,
            },
        ),
        _node(
            "doc-a-secondary",
            0.92,
            {
                "document_id": "doc-a",
                "document_version_group": "group-a",
                "contains_numeric_data": True,
            },
        ),
        _node(
            "doc-b-primary",
            0.9,
            {
                "document_id": "doc-b",
                "document_version_group": "group-b",
                "contains_numeric_data": True,
            },
        ),
    ]

    evidence_nodes = retriever._select_evidence_nodes(
        "Are there conflicting data points across documents?", ranked
    )
    evidence_doc_ids = {
        node.node.metadata.get("document_id") for node in evidence_nodes
    }

    assert "doc-a" in evidence_doc_ids
    assert "doc-b" in evidence_doc_ids


def test_table_query_ensures_table_evidence_when_available():
    retriever = VecteraRetriever("client-1", top_k=10)
    ranked = [
        _node(
            f"text-{i}",
            1.0 - (i * 0.01),
            {"chunk_type": "body_text", "document_id": f"doc-{i}"},
        )
        for i in range(8)
    ]
    ranked.append(
        _node(
            "table-1",
            0.45,
            {
                "chunk_type": "full_table",
                "document_id": "doc-table",
                "table_detected": True,
            },
        )
    )

    evidence = retriever._select_evidence_nodes(
        "Using the top-20 clients table, who are the top three?",
        ranked,
    )
    table_nodes = [
        node
        for node in evidence
        if node.node.metadata.get("chunk_type")
        in {"full_table", "table_segment", "table_summary_text"}
    ]

    assert table_nodes


def test_time_anchored_query_does_not_force_latest_version_bias():
    retriever = VecteraRetriever("client-1", top_k=5)
    old = _node(
        "old-2025",
        0.95,
        {
            "version_rank": "1",
            "is_current": "false",
            "version_label": "2025-q4",
            "document_id": "doc-old",
        },
    )
    current = _node(
        "current",
        0.85,
        {
            "version_rank": "12",
            "is_current": "true",
            "version_label": "2026-q1",
            "document_id": "doc-current",
        },
    )

    ranked = retriever._rank_nodes("As of 2025, what was occupancy?", [current, old])
    assert ranked[0].node.node_id == "old-2025"


def test_build_citations_includes_asset_refs_and_visual_metadata():
    citations = build_citations(
        [
            _node(
                "figure-1",
                0.8,
                {
                    "chunk_type": "figure_artifact",
                    "asset_refs": ["/tmp/fake-chart.png"],
                    "figure_type": "chart",
                    "chart_type": "line",
                    "table_id": None,
                },
            )
        ]
    )

    assert len(citations) == 1
    assert citations[0]["asset_refs"] == ["/tmp/fake-chart.png"]
    assert citations[0]["has_image_assets"] is True
    assert citations[0]["figure_type"] == "chart"
    assert citations[0]["chart_type"] == "line"


def test_build_citations_dedupes_asset_refs_across_citations():
    citations = build_citations(
        [
            _node(
                "figure-1",
                0.8,
                {
                    "chunk_type": "figure_artifact",
                    "asset_refs": ["/tmp/shared-image.png"],
                },
            ),
            _node(
                "figure-2",
                0.7,
                {
                    "chunk_type": "chart_context",
                    "asset_refs": ["/tmp/shared-image.png"],
                },
            ),
        ]
    )

    assert citations[0]["asset_refs"] == ["/tmp/shared-image.png"]
    assert citations[1]["asset_refs"] == []


def test_collect_image_evidence_paths_resolves_relative_and_dedupes(tmp_path):
    absolute_image = tmp_path / "absolute.png"
    absolute_image.write_bytes(b"fake")

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    relative_image = bundle_dir / "relative.png"
    relative_image.write_bytes(b"fake")
    non_image = bundle_dir / "note.txt"
    non_image.write_text("not an image", encoding="utf-8")

    citations = [
        {
            "asset_refs": [str(absolute_image), str(absolute_image)],
            "artifact_bundle_path": None,
        },
        {
            "asset_refs": ["relative.png", "note.txt"],
            "artifact_bundle_path": str(bundle_dir),
        },
    ]

    paths = _collect_image_evidence_paths(citations, max_images=10)

    # Same image bytes should be de-duplicated even when paths differ.
    assert paths == [str(absolute_image.resolve())]


def test_collect_image_evidence_paths_includes_unique_image_content(tmp_path):
    first_image = tmp_path / "first.png"
    first_image.write_bytes(b"first")
    second_image = tmp_path / "second.png"
    second_image.write_bytes(b"second")

    citations = [
        {"asset_refs": [str(first_image)], "artifact_bundle_path": None},
        {"asset_refs": [str(second_image)], "artifact_bundle_path": None},
    ]

    paths = _collect_image_evidence_paths(citations, max_images=10)

    assert paths == [str(first_image.resolve()), str(second_image.resolve())]


def test_collect_image_evidence_paths_respects_max_images(tmp_path):
    first_image = tmp_path / "first.png"
    first_image.write_bytes(b"first")
    second_image = tmp_path / "second.png"
    second_image.write_bytes(b"second")
    third_image = tmp_path / "third.png"
    third_image.write_bytes(b"third")

    citations = [
        {"asset_refs": [str(first_image)], "artifact_bundle_path": None},
        {"asset_refs": [str(second_image)], "artifact_bundle_path": None},
        {"asset_refs": [str(third_image)], "artifact_bundle_path": None},
    ]

    paths = _collect_image_evidence_paths(citations, max_images=2)

    assert paths == [str(first_image.resolve()), str(second_image.resolve())]


def test_visual_query_injects_image_evidence_when_top_evidence_has_no_images():
    retriever = VecteraRetriever("client-1", top_k=10)
    ranked = [
        _node(
            f"text-{i}",
            1.0 - (i * 0.01),
            {
                "chunk_type": "body_text",
                "document_id": f"doc-{i}",
            },
        )
        for i in range(7)
    ]
    ranked.extend(
        [
            _node(
                "image-1",
                0.5,
                {
                    "chunk_type": "figure_artifact",
                    "document_id": "img-doc-1",
                    "asset_refs": ["/tmp/figure-1.png"],
                },
            ),
            _node(
                "image-2",
                0.49,
                {
                    "chunk_type": "chart_context",
                    "document_id": "img-doc-2",
                    "asset_refs": ["/tmp/figure-2.png"],
                },
            ),
        ]
    )

    evidence = retriever._select_evidence_nodes(
        "What does the chart image show?", ranked
    )
    image_evidence = [node for node in evidence if node.node.metadata.get("asset_refs")]

    # 9 input nodes < evidence_limit, so all should be selected
    assert len(evidence) == len(ranked)
    assert len(image_evidence) >= 2


def test_split_reasoning_from_text_extracts_thinking_and_answer():
    raw = (
        "<thinking>Check Source [1] and [2], compare figures, reconcile conflicts.</thinking>\n"
        "<answer>Revenue rises from 10 to 14 across versions [1][2].</answer>"
    )

    answer, reasoning = _split_reasoning_from_text(raw)

    assert answer == "Revenue rises from 10 to 14 across versions [1][2]."
    assert reasoning is not None
    assert "compare figures" in reasoning


def test_build_conversation_context_block_returns_no_context_marker():
    context_block = _build_conversation_context_block({})
    assert context_block == "NO_PRIOR_CONVERSATION_CONTEXT"


def test_build_conversation_context_block_formats_and_truncates_sections():
    long_turn = "x" * 340
    long_user = "u" * 250
    long_assistant = "a" * 290
    context = {
        "session_summary": " Summary line  ",
        "recent_turns": [
            {"role": "user", "content": f"  first   question  {long_turn}  "},
            {"role": "assistant", "content": " second    answer "},
        ],
        "cross_session_pairs": [
            {
                "user_text": f" prior user {long_user} ",
                "assistant_text": f" prior answer {long_assistant} ",
                "score": 0.81234,
            }
        ],
    }

    context_block = _build_conversation_context_block(context)

    assert "SESSION_SUMMARY:\nSummary line" in context_block
    assert "CURRENT_SESSION_RECENT_TURNS:" in context_block
    assert "CROSS_SESSION_RELEVANT_QA:" in context_block
    assert "similarity=0.812" in context_block
    assert "..." in context_block


def test_synthesize_answer_prefers_responses_path(monkeypatch):
    class _FakeBudgeter:
        def __init__(self, model: str):
            self.model = model

        def build_budgeted_sections(self, **_kwargs):
            class _Metrics:
                model = "gpt-test"
                total_input_tokens = 100
                input_budget_tokens = 200
                history_tokens = 10
                summary_tokens = 5
                cross_session_tokens = 15
                evidence_tokens = 60
                conflict_tokens = 10

            return (
                [
                    {"role": "developer", "content": "dev"},
                    {"role": "user", "content": "ctx"},
                ],
                "ctx",
                _Metrics(),
            )

    captured: dict[str, object] = {}

    def _fake_invoke_llm_chat(**kwargs):
        captured["input_messages"] = kwargs.get("input_messages")
        return {"id": "resp-1"}

    monkeypatch.setattr(retriever_module, "ResponsesInputBudgeter", _FakeBudgeter)
    monkeypatch.setattr(
        retriever_module,
        "invoke_llm_chat",
        _fake_invoke_llm_chat,
    )
    monkeypatch.setattr(
        retriever_module,
        "extract_chat_response_text",
        lambda _response: "<answer>Grounded response.</answer>",
    )

    retriever = VecteraRetriever("client-1", top_k=5)
    result = retriever._synthesize_answer(
        question="What changed?",
        citations=[
            {
                "citation_label": "Doc 1",
                "document_name": "Doc 1",
                "version_label": "v2",
                "chunk_type": "body_text",
                "text": "Policy changed on section 2.",
            }
        ],
        conflicts=[],
    )

    assert captured["input_messages"] is not None
    assert result["answer"] == "Grounded response."
    assert result["reasoning"] is None
    assert result["images_used"] == []


def test_extract_answer_and_reasoning_from_chat_uses_blocks():
    response = SimpleNamespace(
        message=SimpleNamespace(
            blocks=[
                ThinkingBlock(content="Reason through source comparison."),
                TextBlock(text="Final grounded answer."),
            ],
            content="",
        )
    )

    answer, reasoning = _extract_answer_and_reasoning_from_chat(response)

    assert answer == "Final grounded answer."
    assert reasoning == "Reason through source comparison."


def test_extract_answer_and_reasoning_from_chat_falls_back_to_message_content():
    response = SimpleNamespace(
        message=SimpleNamespace(
            blocks=[],
            content="<thinking>Hidden reasoning</thinking><answer>Tagged answer</answer>",
        )
    )

    answer, reasoning = _extract_answer_and_reasoning_from_chat(response)

    assert answer == "Tagged answer"
    assert reasoning == "Hidden reasoning"
