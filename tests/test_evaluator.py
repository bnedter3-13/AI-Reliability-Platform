"""
tests/test_evaluator.py

Basic unit tests. These deliberately avoid calling the real Claude API (no network,
no cost) by testing the pure-logic pieces: JSON extraction and prompt building.
Add integration tests (marked e.g. with @pytest.mark.integration) separately once
you're ready to test against the live API using a real ANTHROPIC_API_KEY.
"""

import pytest

from app.evaluation.evaluator import _extract_json
from app.evaluation.prompts import build_judge_message
from app.root_cause.analyzer import analyze_root_cause
from app.evaluation.evaluator import EvaluationResult


def test_extract_json_plain():
    text = '{"correctness_score": 0.9, "faithfulness_score": 0.8, "hallucination_risk": 0.1, "status": "pass", "explanation": "ok"}'
    result = _extract_json(text)
    assert result["status"] == "pass"
    assert result["correctness_score"] == 0.9


def test_extract_json_with_surrounding_text():
    text = 'Here is the evaluation:\n{"correctness_score": 0.5, "faithfulness_score": 0.4, "hallucination_risk": 0.6, "status": "fail", "explanation": "bad"}\nDone.'
    result = _extract_json(text)
    assert result["status"] == "fail"


def test_extract_json_raises_on_no_json():
    with pytest.raises(ValueError):
        _extract_json("no json here at all")


def test_build_judge_message_includes_all_parts():
    message = build_judge_message(
        question="What is 2+2?",
        contexts=["2+2 equals 4."],
        answer="2+2 is 4.",
    )
    assert "What is 2+2?" in message
    assert "2+2 equals 4." in message
    assert "2+2 is 4." in message


def test_build_judge_message_handles_empty_contexts():
    message = build_judge_message(question="Q?", contexts=[], answer="A.")
    assert "no contexts retrieved" in message


def test_root_cause_no_context():
    result = EvaluationResult(
        correctness_score=0.2, faithfulness_score=0.2, hallucination_risk=0.8,
        status="fail", explanation="bad", latency_ms=100.0,
    )
    rc = analyze_root_cause(result, contexts=[])
    assert rc.cause == "no_context"


def test_root_cause_hallucination():
    result = EvaluationResult(
        correctness_score=0.5, faithfulness_score=0.5, hallucination_risk=0.9,
        status="fail", explanation="bad", latency_ms=100.0,
    )
    rc = analyze_root_cause(result, contexts=["some context"])
    assert rc.cause == "hallucination"


def test_root_cause_pass_status():
    result = EvaluationResult(
        correctness_score=0.9, faithfulness_score=0.9, hallucination_risk=0.05,
        status="pass", explanation="good", latency_ms=100.0,
    )
    rc = analyze_root_cause(result, contexts=["some context"])
    assert rc.cause == "none"
