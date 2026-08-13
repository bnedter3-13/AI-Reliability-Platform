"""
app/knowledge_base/indexing_service.py

Knowledge Base + Retrieval Verification Agent (Component 11): the full ingestion
pipeline — extract -> chunk -> embed -> store.
"""

import uuid
from dataclasses import dataclass
from typing import Optional

from app.knowledge_base.document_loader import load_pdf_text
from app.knowledge_base.chunker import chunk_text
from app.knowledge_base.vector_store import KnowledgeBaseStore


@dataclass
class IndexingResult:
    project_id: str
    doc_id: str
    filename: Optional[str]
    num_chunks: int


def index_pdf(
    project_id: str,
    file_path: str,
    original_filename: Optional[str] = None,
    store: Optional[KnowledgeBaseStore] = None,
) -> IndexingResult:
    """Extract a PDF's text, chunk it, embed the chunks, and store them for `project_id`."""
    text = load_pdf_text(file_path)
    chunks = chunk_text(text)

    doc_id = str(uuid.uuid4())
    store = store or KnowledgeBaseStore()
    store.add_chunks(project_id, doc_id, chunks)

    return IndexingResult(
        project_id=project_id,
        doc_id=doc_id,
        filename=original_filename,
        num_chunks=len(chunks),
    )
