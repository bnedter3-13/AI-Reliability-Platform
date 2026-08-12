"""
tests/test_rag_evaluator.py

Unit tests for RAG Evaluation (Component 9): Context Relevance, Context Precision,
and Context Recall. Like test_evaluator.py, these avoid calling the real Claude API
by monkeypatching the module-level `_client` with a fake object that returns
canned JSON text, so no network calls or API cost are involved.
"""

from types import SimpleNamespace

import pytest

import app.evaluation.rag_evaluator as rag_evaluator
from app.evaluation.rag_evaluator import (
    ContextPrecisionResult,
    average_context_precision,
    average_context_relevance,
    evaluate_context_precision,
    evaluate_context_recall,
    evaluate_context_relevance,
    run_full_rag_evaluation,
)


class _FakeMessages:
    """Returns each entry in `responses` in order; an Exception entry is raised
    instead of returned, so tests can simulate a mid-batch API failure."""

    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=[SimpleNamespace(text=response, type="text")])


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


# ---------------------------------------------------------------------------
# Context Relevance
# ---------------------------------------------------------------------------

def test_evaluate_context_relevance_no_client_raises(monkeypatch):
    monkeypatch.setattr(rag_evaluator, "_client", None)
    with pytest.raises(RuntimeError):
        evaluate_context_relevance("Q?", ["ctx"])


def test_evaluate_context_relevance_returns_scores(monkeypatch):
    monkeypatch.setattr(rag_evaluator, "_client", _FakeClient([
        '{"relevance_score": 0.9, "reason": "on topic"}',
        '{"relevance_score": 0.2, "reason": "off topic"}',
    ]))

    results = evaluate_context_relevance("Q?", ["ctx1", "ctx2"])

    assert [r.relevance_score for r in results] == [0.9, 0.2]
    assert results[0].reason == "on topic"


def test_evaluate_context_relevance_handles_per_context_error(monkeypatch):
    monkeypatch.setattr(rag_evaluator, "_client", _FakeClient([
        RuntimeError("boom"),
        '{"relevance_score": 0.5, "reason": "ok"}',
    ]))

    results = evaluate_context_relevance("Q?", ["bad", "good"])

    assert results[0].relevance_score == 0.0
    assert "Error" in results[0].reason
    assert results[1].relevance_score == 0.5


def test_average_context_relevance_empty():
    assert average_context_relevance([]) == 0.0


# ---------------------------------------------------------------------------
# Context Precision
# ---------------------------------------------------------------------------

def test_evaluate_context_precision_no_client_raises(monkeypatch):
    monkeypatch.setattr(rag_evaluator, "_client", None)
    with pytest.raises(RuntimeError):
        evaluate_context_precision("Q?", "ref", ["ctx"])


def test_evaluate_context_precision_returns_results(monkeypatch):
    monkeypatch.setattr(rag_evaluator, "_client", _FakeClient([
        '{"is_useful": true, "useful_score": 0.9, "reason": "used"}',
        '{"is_useful": false, "useful_score": 0.1, "reason": "unused"}',
    ]))

    results = evaluate_context_precision("Q?", "ref answer", ["ctx1", "ctx2"])

    assert results[0].is_useful is True
    assert results[1].is_useful is False


def test_evaluate_context_precision_handles_per_context_error(monkeypatch):
    monkeypatch.setattr(rag_evaluator, "_client", _FakeClient([RuntimeError("boom")]))

    results = evaluate_context_precision("Q?", "ref answer", ["ctx"])

    assert results[0].is_useful is False
    assert results[0].useful_score == 0.0
    assert "Error" in results[0].reason


def test_average_context_precision_computes_fraction():
    results = [
        ContextPrecisionResult(context="a", is_useful=True, useful_score=0.9, reason="x"),
        ContextPrecisionResult(context="b", is_useful=False, useful_score=0.1, reason="y"),
    ]
    assert average_context_precision(results) == 0.5


def test_average_context_precision_empty():
    assert average_context_precision([]) == 0.0


# ---------------------------------------------------------------------------
# Context Recall
# ---------------------------------------------------------------------------

def test_evaluate_context_recall_no_client_raises(monkeypatch):
    monkeypatch.setattr(rag_evaluator, "_client", None)
    with pytest.raises(RuntimeError):
        evaluate_context_recall("Q?", "ref", ["ctx"])


def test_evaluate_context_recall_returns_result(monkeypatch):
    monkeypatch.setattr(rag_evaluator, "_client", _FakeClient([
        '{"recall_score": 0.75, "covered_claims": 3, "total_claims": 4, "missing_info": "one detail missing"}',
    ]))

    result = evaluate_context_recall("Q?", "ref answer", ["ctx1", "ctx2"])

    assert result.recall_score == 0.75
    assert result.covered_claims == 3
    assert result.total_claims == 4
    assert result.missing_info == "one detail missing"


def test_evaluate_context_recall_handles_error(monkeypatch):
    monkeypatch.setattr(rag_evaluator, "_client", _FakeClient([RuntimeError("boom")]))

    result = evaluate_context_recall("Q?", "ref answer", ["ctx1"])

    assert result.recall_score == 0.0
    assert result.covered_claims == 0
    assert "Error" in result.missing_info


# ---------------------------------------------------------------------------
# Combined report (run_full_rag_evaluation)
# ---------------------------------------------------------------------------

def test_run_full_rag_evaluation_without_reference_answer_only_computes_relevance(monkeypatch):
    monkeypatch.setattr(rag_evaluator, "_client", _FakeClient([
        '{"relevance_score": 0.8, "reason": "ok"}',
    ]))

    report = run_full_rag_evaluation("Q?", ["ctx1"])

    assert report.context_relevance_avg == 0.8
    assert report.context_precision_avg is None
    assert report.context_recall_score is None
    assert report.recall_missing_info is None


def test_run_full_rag_evaluation_with_reference_answer_computes_all_three(monkeypatch):
    monkeypatch.setattr(rag_evaluator, "_client", _FakeClient([
        '{"relevance_score": 0.8, "reason": "ok"}',
        '{"is_useful": true, "useful_score": 0.9, "reason": "used"}',
        '{"recall_score": 0.6, "covered_claims": 2, "total_claims": 3, "missing_info": "missing X"}',
    ]))

    report = run_full_rag_evaluation("Q?", ["ctx1"], reference_answer="ref answer")

    assert report.context_relevance_avg == 0.8
    assert report.context_precision_avg == 1.0
    assert report.context_recall_score == 0.6
    assert report.recall_missing_info == "missing X"


def test_run_full_rag_evaluation_empty_contexts(monkeypatch):
    monkeypatch.setattr(rag_evaluator, "_client", _FakeClient([]))

    report = run_full_rag_evaluation("Q?", [])

    assert report.context_relevance_avg == 0.0
    assert report.context_precision_avg is None
