from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

import app.retrieval.query_decomposition as decomposition_module
import app.retrieval.retriever as retriever_module
from app.core.config import settings
from app.retrieval.retriever import VecteraRetriever
from app.retrieval.v1_router import (
    RetrievalStrategy,
    RoutingDecision,
    V1RetrievalRouter,
)


def _node(node_id: str, score: float = 0.9) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(id_=node_id, text=f"source text for {node_id}"),
        score=score,
    )


class _BaseRetriever:
    def __init__(self):
        self.queries: list[str] = []

    def retrieve(self, question: str) -> list[NodeWithScore]:
        self.queries.append(question)
        return [_node(question)]


def _enable_router(retriever: VecteraRetriever, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ENABLE_V1_QUERY_ROUTER", True)
    retriever._v1_query_router_active = True


@pytest.mark.parametrize(
    ("index", "question", "expected"),
    [
        (0, "What was BXP occupancy in Q4 2025?", RetrievalStrategy.DIRECT),
        (1, "What about that chart?", RetrievalStrategy.EXPANDED),
        (
            2,
            "Compare NOI and FFO for Company A and Company B",
            RetrievalStrategy.DECOMPOSED,
        ),
    ],
)
def test_llamaindex_selector_result_maps_to_typed_strategy(
    index,
    question,
    expected,
):
    selector = SimpleNamespace(
        select=lambda _choices, _question: SimpleNamespace(
            ind=index,
            reason="Selected for the query shape.",
        )
    )

    decision = V1RetrievalRouter(selector).route(question)

    assert decision == RoutingDecision(
        strategy=expected,
        reason="Selected for the query shape.",
    )


def test_decompose_query_uses_v1_model_and_preserves_standalone_queries(monkeypatch):
    captured: dict = {}

    def _fake_invoke(**kwargs):
        captured.update(kwargs)
        return {"id": "response-1"}

    monkeypatch.setattr(decomposition_module, "invoke_llm_chat", _fake_invoke)
    monkeypatch.setattr(
        decomposition_module,
        "extract_chat_response_text",
        lambda _response: (
            '{"queries": ["BXP NOI FY2025", "BXP FFO FY2025", '
            '"BXP NOI FY2025", "VICI FFO FY2025"]}'
        ),
    )
    monkeypatch.setattr(settings, "V1_QUERY_ROUTER_MODEL", "test-router-model")

    queries = decomposition_module.decompose_query(
        "Compare BXP and VICI NOI and FFO in FY2025"
    )

    assert queries == ["BXP NOI FY2025", "BXP FFO FY2025", "VICI FFO FY2025"]
    assert captured["model"] == "test-router-model"
    assert (
        "Compare BXP and VICI NOI and FFO in FY2025"
        in captured["input_messages"][1]["content"]
    )


def test_v1_direct_route_runs_one_query(monkeypatch):
    retriever = VecteraRetriever("client-1", top_k=5)
    _enable_router(retriever, monkeypatch)
    base_retriever = _BaseRetriever()
    monkeypatch.setattr(
        retriever_module,
        "get_v1_retrieval_router",
        lambda: SimpleNamespace(
            route=lambda _question: RoutingDecision(
                RetrievalStrategy.DIRECT,
                "Precise single-metric lookup.",
            )
        ),
    )
    monkeypatch.setattr(
        retriever_module.vector_store_manager,
        "get_retriever",
        lambda **_kwargs: base_retriever,
    )

    nodes, expanded = retriever._retrieve_with_mode(
        "What was BXP occupancy in Q4 2025?",
        hybrid=True,
    )

    assert len(nodes) == 1
    assert expanded is False
    assert base_retriever.queries == ["What was BXP occupancy in Q4 2025?"]
    assert retriever._last_retrieval_metadata["retrieval_strategy"] == "direct"
    assert (
        retriever._last_retrieval_metadata["routed_queries"] == base_retriever.queries
    )


def test_v1_expanded_route_forces_history_aware_variants(monkeypatch):
    retriever = VecteraRetriever(
        "client-1",
        top_k=5,
        conversation_context={
            "recent_turns": [{"role": "user", "content": "Show BXP Q4 occupancy."}]
        },
    )
    _enable_router(retriever, monkeypatch)
    base_retriever = _BaseRetriever()
    captured: dict = {}
    monkeypatch.setattr(
        retriever_module,
        "get_v1_retrieval_router",
        lambda: SimpleNamespace(
            route=lambda _question: RoutingDecision(
                RetrievalStrategy.EXPANDED,
                "Conversational follow-up.",
            )
        ),
    )

    def _variants(payload, max_rewrites=2, *, force=False):
        captured["payload"] = payload
        captured["force"] = force
        return ["What about that?", "BXP occupancy Q4 2025"]

    monkeypatch.setattr(retriever_module, "build_query_variants", _variants)
    monkeypatch.setattr(
        retriever_module.vector_store_manager,
        "get_retriever",
        lambda **_kwargs: base_retriever,
    )

    _, expanded = retriever._retrieve_with_mode("What about that?", hybrid=True)

    assert expanded is True
    assert captured["force"] is True
    assert captured["payload"]["recent_turns"][0]["content"] == "Show BXP Q4 occupancy."
    assert base_retriever.queries == ["What about that?", "BXP occupancy Q4 2025"]
    assert retriever._last_retrieval_metadata["retrieval_strategy"] == "expanded"


def test_v1_decomposed_route_fuses_sub_queries(monkeypatch):
    retriever = VecteraRetriever("client-1", top_k=5)
    _enable_router(retriever, monkeypatch)
    base_retriever = _BaseRetriever()
    monkeypatch.setattr(
        retriever_module,
        "get_v1_retrieval_router",
        lambda: SimpleNamespace(
            route=lambda _question: RoutingDecision(
                RetrievalStrategy.DECOMPOSED,
                "Multi-entity comparison.",
            )
        ),
    )
    monkeypatch.setattr(
        retriever_module,
        "decompose_query",
        lambda _question: ["BXP NOI FY2025", "VICI NOI FY2025"],
    )
    monkeypatch.setattr(
        retriever_module.vector_store_manager,
        "get_retriever",
        lambda **_kwargs: base_retriever,
    )

    nodes, expanded = retriever._retrieve_with_mode(
        "Compare BXP and VICI NOI in FY2025",
        hybrid=True,
    )

    assert expanded is False
    assert len(nodes) == 2
    assert base_retriever.queries == ["BXP NOI FY2025", "VICI NOI FY2025"]
    assert retriever._last_retrieval_metadata["retrieval_strategy"] == "decomposed"


def test_decomposition_failure_falls_back_to_expansion(monkeypatch):
    retriever = VecteraRetriever("client-1", top_k=5)
    _enable_router(retriever, monkeypatch)
    base_retriever = _BaseRetriever()
    monkeypatch.setattr(
        retriever_module,
        "get_v1_retrieval_router",
        lambda: SimpleNamespace(
            route=lambda _question: RoutingDecision(
                RetrievalStrategy.DECOMPOSED,
                "Compound question.",
            )
        ),
    )
    monkeypatch.setattr(
        retriever_module,
        "decompose_query",
        lambda _question: (_ for _ in ()).throw(RuntimeError("planner unavailable")),
    )
    monkeypatch.setattr(
        retriever_module,
        "build_query_variants",
        lambda _payload, *, force=False: ["original", "expanded variant"],
    )
    monkeypatch.setattr(
        retriever_module.vector_store_manager,
        "get_retriever",
        lambda **_kwargs: base_retriever,
    )

    _, expanded = retriever._retrieve_with_mode("original", hybrid=True)

    assert expanded is True
    assert retriever._last_retrieval_metadata["retrieval_strategy"] == "expanded"
    assert (
        "planner unavailable"
        in retriever._last_retrieval_metadata["router_fallback_reason"]
    )


def test_router_failure_uses_legacy_strategy(monkeypatch):
    retriever = VecteraRetriever("client-1", top_k=5)
    _enable_router(retriever, monkeypatch)
    base_retriever = _BaseRetriever()
    monkeypatch.setattr(
        retriever_module,
        "get_v1_retrieval_router",
        lambda: SimpleNamespace(
            route=lambda _question: (_ for _ in ()).throw(RuntimeError("selector down"))
        ),
    )
    monkeypatch.setattr(retriever_module, "should_expand_query", lambda _payload: False)
    monkeypatch.setattr(
        retriever_module.vector_store_manager,
        "get_retriever",
        lambda **_kwargs: base_retriever,
    )

    retriever._retrieve_with_mode("What is occupancy?", hybrid=True)

    assert retriever._last_retrieval_metadata["retrieval_strategy"] == "legacy"
    assert (
        "selector down" in retriever._last_retrieval_metadata["router_fallback_reason"]
    )


def test_disabled_router_uses_legacy_strategy(monkeypatch):
    retriever = VecteraRetriever("client-1", top_k=5)
    monkeypatch.setattr(settings, "ENABLE_V1_QUERY_ROUTER", False)
    retriever._v1_query_router_active = True
    monkeypatch.setattr(
        retriever_module,
        "get_v1_retrieval_router",
        lambda: (_ for _ in ()).throw(AssertionError("disabled router invoked")),
    )
    monkeypatch.setattr(retriever_module, "should_expand_query", lambda _payload: False)
    monkeypatch.setattr(
        retriever_module.vector_store_manager,
        "get_retriever",
        lambda **_kwargs: _BaseRetriever(),
    )

    retriever._retrieve_with_mode("What is occupancy?", hybrid=True)

    assert retriever._last_retrieval_metadata["retrieval_strategy"] == "legacy"


def test_retrieve_only_never_invokes_v1_router(monkeypatch):
    retriever = VecteraRetriever("client-1", top_k=5)
    monkeypatch.setattr(settings, "ENABLE_V1_QUERY_ROUTER", True)
    monkeypatch.setattr(
        retriever_module,
        "get_v1_retrieval_router",
        lambda: (_ for _ in ()).throw(AssertionError("v1 router invoked")),
    )
    monkeypatch.setattr(retriever_module, "should_expand_query", lambda _payload: False)
    monkeypatch.setattr(
        retriever_module.vector_store_manager,
        "get_retriever",
        lambda **_kwargs: _BaseRetriever(),
    )
    monkeypatch.setattr(retriever, "_rank_nodes", lambda _question, nodes: nodes)

    nodes = retriever.retrieve_only("Agentic retrieval query")

    assert len(nodes) == 1
    assert retriever._last_retrieval_metadata["retrieval_strategy"] == "legacy"
