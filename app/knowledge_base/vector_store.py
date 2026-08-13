"""
app/knowledge_base/vector_store.py

Knowledge Base + Retrieval Verification Agent (Component 11): local Chroma DB
integration — stores chunk embeddings per project and runs semantic search over
them. Uses Chroma's bundled default embedding function (a local ONNX MiniLM-L6-v2
model), so no external embedding API/key is required.
"""

from dataclasses import dataclass
from typing import List, Optional

import chromadb

from app.config import settings

# Cosine space so distances are directly convertible to a 0-1-ish relevance score
# (relevance = 1 - distance) for the verification agent's threshold check.
_COLLECTION_METADATA = {"hnsw:space": "cosine"}


@dataclass
class QueryMatch:
    text: str
    distance: float


class KnowledgeBaseStore:
    """Wraps a Chroma PersistentClient; one collection per project_id."""

    def __init__(self, persist_dir: Optional[str] = None):
        self._client = chromadb.PersistentClient(path=persist_dir or settings.CHROMA_PERSIST_DIR)

    def _collection_name(self, project_id: str) -> str:
        return f"kb_{project_id}"

    def add_chunks(self, project_id: str, doc_id: str, chunks: List[str]) -> None:
        """Embed and store a document's chunks under the given project's collection."""
        if not chunks:
            return
        collection = self._client.get_or_create_collection(
            self._collection_name(project_id), metadata=_COLLECTION_METADATA
        )
        collection.add(
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))],
        )

    def query(self, project_id: str, query_text: str, n_results: int = 3) -> List[QueryMatch]:
        """Semantic search: return the n_results closest chunks to query_text, if any."""
        existing = {c.name for c in self._client.list_collections()}
        if self._collection_name(project_id) not in existing:
            return []

        collection = self._client.get_collection(self._collection_name(project_id))
        if collection.count() == 0:
            return []

        result = collection.query(
            query_texts=[query_text], n_results=min(n_results, collection.count())
        )
        documents = result["documents"][0]
        distances = result["distances"][0]
        return [QueryMatch(text=doc, distance=dist) for doc, dist in zip(documents, distances)]
