"""
tests/test_chunker.py

Unit tests for Knowledge Base chunking (Component 11).
"""

import pytest

from app.knowledge_base.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_text_shorter_than_chunk_size_returns_single_chunk():
    text = "short document"
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert chunks == [text]


def test_chunk_size_and_overlap_are_respected():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    assert all(len(c) <= 500 for c in chunks)
    # step = chunk_size - overlap = 450, so chunk boundaries start at 0, 450, 900
    assert len(chunks) == 3
    assert chunks[0] == text[0:500]
    assert chunks[1] == text[450:950]
    assert chunks[2] == text[900:1200]


def test_consecutive_chunks_overlap_by_requested_amount():
    text = "".join(str(i % 10) for i in range(1000))
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    for i in range(len(chunks) - 1):
        end_of_current = chunks[i][-50:]
        start_of_next = chunks[i + 1][:50]
        assert end_of_current == start_of_next


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)
