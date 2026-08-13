"""
app/knowledge_base/chunker.py

Knowledge Base + Retrieval Verification Agent (Component 11): splits extracted
document text into fixed-size, overlapping chunks for embedding/storage.
"""

from typing import List

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split `text` into chunks of `chunk_size` characters, each overlapping the
    previous chunk by `overlap` characters — keeps context from being cut off
    right at a chunk boundary.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks
