import pytest
from app.services.synthesizer import _split_reasoning_from_text, _scrub_cot_artifacts

def test_split_reasoning_missing_close_tag():
    # Strategy 4: <thinking> opened but NOT closed
    text = "<thinking>step 1: analyze\nstep 2: conclude"  # no </thinking>
    reasoning, answer = _split_reasoning_from_text(text)
    # It should treat the content after <thinking> as reasoning, and answer as empty if nothing before
    assert "step 1: analyze" in reasoning
    assert answer == ""

def test_split_reasoning_truncated_with_text_before():
    text = "Some intro text. <thinking>analyzing..."
    reasoning, answer = _split_reasoning_from_text(text)
    assert answer == "Some intro text."
    assert reasoning == "" # Discarded because it's an unclosed tag at the end

def test_split_reasoning_both_tags():
    text = "<thinking>reasoning</thinking><answer>42%</answer>"
    reasoning, answer = _split_reasoning_from_text(text)
    assert reasoning == "reasoning"
    assert answer == "42%"

def test_scrub_cot_artifacts():
    text = "Let me think through this.\nThe answer is 42%."
    assert _scrub_cot_artifacts(text) == "The answer is 42%."

def test_scrub_cot_artifacts_steps():
    text = "Step 1: Look at the table.\nStep 2: Calculate.\nThe total is 100."
    assert _scrub_cot_artifacts(text) == "The total is 100."

def test_split_reasoning_strategy_3():
    # <thinking> opened and closed, no <answer> — use text after </thinking>
    text = "<thinking>I will look it up.</thinking> The value is $10M."
    reasoning, answer = _split_reasoning_from_text(text)
    assert reasoning == "I will look it up."
    assert answer == "The value is $10M."
