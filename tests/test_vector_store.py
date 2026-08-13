"""
tests/test_vector_store.py

Unit tests for the Knowledge Base local Chroma integration (Component 11). Uses a
real Chroma PersistentClient pointed at a pytest tmp_path so tests stay isolated
from the app's real data/chroma_db/ directory and from each other.
"""

from app.knowledge_base.vector_store import KnowledgeBaseStore


def _store(tmp_path):
    return KnowledgeBaseStore(persist_dir=str(tmp_path))


def test_query_against_empty_store_returns_no_matches(tmp_path):
    store = _store(tmp_path)
    assert store.query("proj-a", "anything") == []


def test_add_and_query_round_trip(tmp_path):
    store = _store(tmp_path)
    store.add_chunks(
        "proj-a",
        "doc-1",
        [
            "The refund policy allows returns within 30 days of purchase.",
            "Bicycle maintenance requires regular chain lubrication.",
        ],
    )

    matches = store.query("proj-a", "What is the refund policy?", n_results=2)

    assert len(matches) == 2
    assert matches[0].text == "The refund policy allows returns within 30 days of purchase."
    assert matches[0].distance < matches[1].distance


def test_query_is_scoped_to_project(tmp_path):
    store = _store(tmp_path)
    store.add_chunks("proj-a", "doc-1", ["Project A content about refunds."])

    assert store.query("proj-b", "refunds") == []
