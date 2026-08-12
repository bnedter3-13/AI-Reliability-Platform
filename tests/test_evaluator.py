"""
tests/test_evaluator.py

Basic unit tests. These deliberately avoid calling the real Claude API (no network,
no cost) by testing the pure-logic pieces: JSON extraction and prompt building.
Add integration tests (marked e.g. with @pytest.mark.integration) separately once
you're ready to test against the live API using a real ANTHROPIC_API_KEY.
"""

import pytest

import app.evaluation.evaluator as evaluator
from app.evaluation.evaluator import _extract_json, evaluate_answer
from app.evaluation.prompts import build_judge_message
from app.root_cause.analyzer import analyze_root_cause
from app.evaluation.evaluator import EvaluationResult


class _FakeContentBlock:
    def __init__(self, text, type="text"):
        self.text = text
        self.type = type


class _FakeThinkingBlock:
    """Mimics Claude's extended-thinking content block, which has no .text
    attribute - used to regression-test that evaluate_answer() finds the actual
    text block instead of assuming content[0] is always text."""

    def __init__(self, thinking):
        self.type = "thinking"
        self.thinking = thinking


class _FakeMessage:
    def __init__(self, text, leading_blocks=()):
        self.content = [*leading_blocks, _FakeContentBlock(text)]


class _FakeMessages:
    def __init__(self, text, leading_blocks=()):
        self._text = text
        self._leading_blocks = leading_blocks

    def create(self, **kwargs):
        return _FakeMessage(self._text, leading_blocks=self._leading_blocks)


class _FakeClient:
    def __init__(self, text, leading_blocks=()):
        self.messages = _FakeMessages(text, leading_blocks=leading_blocks)


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


def test_evaluate_answer_wrong_shaped_json_degrades_to_error_status(monkeypatch):
    # Valid JSON syntax, but missing "explanation" - EvaluationResult(**raw, ...)
    # used to be constructed outside the try/except, so this TypeError propagated
    # uncaught instead of degrading to status="error" like other judge failures.
    bad_json = (
        '{"correctness_score": 0.9, "faithfulness_score": 0.8, '
        '"hallucination_risk": 0.1, "status": "pass"}'
    )
    monkeypatch.setattr(evaluator, "_client", _FakeClient(bad_json))

    result = evaluate_answer(answer="A.", contexts=["ctx"], question="Q?")

    assert result.status == "error"
    assert result.correctness_score == 0.0
    assert result.faithfulness_score == 0.0
    assert result.hallucination_risk == 1.0


def test_evaluate_answer_extra_key_in_json_degrades_to_error_status(monkeypatch):
    # Valid JSON with all required keys plus one the judge shouldn't have added -
    # also raises TypeError on construction (unexpected keyword argument).
    bad_json = (
        '{"correctness_score": 0.9, "faithfulness_score": 0.8, '
        '"hallucination_risk": 0.1, "status": "pass", "explanation": "ok", '
        '"confidence": 0.99}'
    )
    monkeypatch.setattr(evaluator, "_client", _FakeClient(bad_json))

    result = evaluate_answer(answer="A.", contexts=["ctx"], question="Q?")

    assert result.status == "error"


def test_evaluate_answer_skips_leading_thinking_block(monkeypatch):
    # Claude (the judge model) sometimes emits extended thinking before its text
    # response, making content[0] a block with no .text attribute. evaluate_answer()
    # must find the actual text block rather than assuming index 0 is it.
    good_json = (
        '{"correctness_score": 0.9, "faithfulness_score": 0.8, '
        '"hallucination_risk": 0.1, "status": "pass", "explanation": "ok"}'
    )
    monkeypatch.setattr(
        evaluator, "_client",
        _FakeClient(good_json, leading_blocks=[_FakeThinkingBlock("reasoning about the answer...")]),
    )

    result = evaluate_answer(answer="A.", contexts=["ctx"], question="Q?")

    assert result.status == "pass"
    assert result.correctness_score == 0.9
    assert result.explanation == "ok"


def test_evaluate_answer_no_text_block_degrades_to_error_status(monkeypatch):
    # If every content block lacks a .text (e.g. the response were somehow all
    # thinking, or a future non-text block type), evaluate_answer() should still
    # degrade to status="error" rather than raising uncaught.
    class _NoTextMessages:
        def create(self, **kwargs):
            return type("M", (), {"content": [_FakeThinkingBlock("only thinking, no answer")]})()

    class _NoTextClient:
        def __init__(self):
            self.messages = _NoTextMessages()

    monkeypatch.setattr(evaluator, "_client", _NoTextClient())

    result = evaluate_answer(answer="A.", contexts=["ctx"], question="Q?")

    assert result.status == "error"
    assert "No text block found" in result.explanation


def test_root_cause_pass_status():
    result = EvaluationResult(
        correctness_score=0.9, faithfulness_score=0.9, hallucination_risk=0.05,
        status="pass", explanation="good", latency_ms=100.0,
    )
    rc = analyze_root_cause(result, contexts=["some context"])
    assert rc.cause == "none"
