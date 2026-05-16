import pytest
from llama_index.core.schema import NodeWithScore, TextNode
from app.components.reranker import rerank_nodes

def test_reranker_boosts_metric_matches():
    # Mock nodes: one with 'walt' metric, one without
    node1 = NodeWithScore(
        node=TextNode(text="Generic text about the market.", metadata={"metric_types": []}),
        score=0.7
    )
    node2 = NodeWithScore(
        node=TextNode(text="The WALT is 10 years.", metadata={"metric_types": ["walt"]}),
        score=0.65
    )
    
    # Query asking about WALT
    query = "What is the WALT?"
    
    # Rerank
    results = rerank_nodes([node1, node2], query=query)
    
    # node2 should be boosted above node1 because of 'walt' match
    # adjustment is +0.07, so 0.65 + 0.07 = 0.72 > 0.7
    assert results[0].node.node_id == node2.node.node_id

def test_reranker_version_consistency_with_company_ticker():
    # Two nodes from same company, different document types
    # They should NOT penalize each other as stale versions
    node1 = NodeWithScore(
        node=TextNode(
            text="Q4 report", 
            metadata={
                "company_ticker": "BXP", 
                "document_type": "quarterly report",
                "version_label": "v1"
            }
        ),
        score=0.8
    )
    node2 = NodeWithScore(
        node=TextNode(
            text="Investor Day", 
            metadata={
                "company_ticker": "BXP", 
                "document_type": "investor presentation",
                "version_label": "v1"
            }
        ),
        score=0.79
    )
    
    query = "Show me BXP data"
    results = rerank_nodes([node1, node2], query=query)
    
    # Scores should remain close, no massive version penalty (-0.08) applied
    assert abs(results[0].score - results[1].score) < 0.05

def test_reranker_version_consistency_stale_version_penalty():
    # Two nodes from same company, same document type, different versions
    node1 = NodeWithScore(
        node=TextNode(
            text="Q4 report v2", 
            metadata={
                "company_ticker": "BXP", 
                "document_type": "quarterly report",
                "version_label": "v2"
            }
        ),
        score=0.8
    )
    node2 = NodeWithScore(
        node=TextNode(
            text="Q4 report v1", 
            metadata={
                "company_ticker": "BXP", 
                "document_type": "quarterly report",
                "version_label": "v1"
            }
        ),
        score=0.79
    )
    
    query = "Show me BXP Q4 data"
    results = rerank_nodes([node1, node2], query=query)
    
    # node2 should be penalized significantly
    # 0.79 - 0.08 = 0.71
    assert results[0].node.node_id == node1.node.node_id
    assert results[1].score < 0.75
