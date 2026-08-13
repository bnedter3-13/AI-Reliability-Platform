"""
tests/test_verification_agent.py

Unit tests for the Retrieval Verification Agent (Component 11). Uses a fake store
object (following the same fake-object pattern as test_rag_evaluator.py) so these
tests don't touch a real Chroma instance or embedding model.
"""

import pytest

from app.knowledge_base.verification_agent import RELEVANCE_THRESHOLD, verify_question
from app.knowledge_base.vector_store import QueryMatch


class _FakeStore:
    def __init__(self, matches):
        self._matches = matches

    def query(self, project_id, query_text, n_results=3):
        return self._matches


def test_no_indexed_documents_is_not_supported():
    store = _FakeStore([])
    result = verify_question("proj-a", "Any question?", store=store)

    assert result.supported is False
    assert result.best_relevance_score == 0.0
    assert result.matched_chunks == []


def test_relevance_above_threshold_is_supported():
    # distance 0.15 -> relevance 0.85, above the 0.8 threshold
    store = _FakeStore([QueryMatch(text="Refunds are allowed within 30 days.", distance=0.15)])
    result = verify_question("proj-a", "What is the refund policy?", store=store)

    assert result.supported is True
    assert result.best_relevance_score == 0.85
    assert result.threshold == RELEVANCE_THRESHOLD
    assert result.matched_chunks == ["Refunds are allowed within 30 days."]


def test_relevance_below_threshold_is_not_supported():
    # distance 0.7 -> relevance 0.3, below the 0.8 threshold
    store = _FakeStore([QueryMatch(text="Unrelated content.", distance=0.7)])
    result = verify_question("proj-a", "What is the refund policy?", store=store)

    assert result.supported is False
    assert result.best_relevance_score == pytest.approx(0.3)


def test_best_match_among_several_is_used():
    store = _FakeStore(
        [
            QueryMatch(text="Weakly related.", distance=0.5),
            QueryMatch(text="Strongly related.", distance=0.1),
        ]
    )
    result = verify_question("proj-a", "Question?", store=store)

    assert result.best_relevance_score == 0.9
    assert result.matched_chunks[0] == "Strongly related."
