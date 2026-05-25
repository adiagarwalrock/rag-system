from app.core.config import settings
from app.core.token_budget import ResponsesInputBudgeter


def test_response_output_token_default_supports_multi_document_answers():
    assert settings.RESPONSE_MAX_OUTPUT_TOKENS == 2500


def test_budgeted_sections_fit_within_input_budget():
    budgeter = ResponsesInputBudgeter(model="gpt-5.4-mini", ratio=0.8)

    recent_turns = [
        {"role": "user", "content": "Question " + ("x" * 300)},
        {"role": "assistant", "content": "Answer " + ("y" * 300)},
    ] * 8
    cross_lines = [f"- cross {i} " + ("z" * 200) for i in range(12)]
    evidence_lines = [f"[{i}] evidence " + ("e" * 800) for i in range(20)]
    conflict_lines = [f"- conflict {i} " + ("c" * 200) for i in range(6)]

    messages, user_context, metrics = budgeter.build_budgeted_sections(
        developer_prompt="Static developer instructions",
        question="What changed in the latest policy?",
        recent_turns=recent_turns,
        session_summary="Summary " + ("s" * 1200),
        cross_session_lines=cross_lines,
        evidence_lines=evidence_lines,
        conflict_lines=conflict_lines,
    )

    assert messages[0]["role"] == "developer"
    assert messages[-1]["role"] == "user"
    assert "CURRENT_QUERY" in user_context
    assert "SESSION_SUMMARY" in user_context
    assert "CROSS_SESSION_RELEVANT_QA" in user_context
    assert "RETRIEVAL_EVIDENCE" in user_context
    assert "CONFLICT_HINTS" in user_context
    assert metrics.total_input_tokens <= metrics.input_budget_tokens
