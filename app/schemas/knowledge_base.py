"""
app/schemas/knowledge_base.py

Pydantic models for the Knowledge Base + Retrieval Verification Agent
(Component 11) endpoints: /knowledge-base/upload and /knowledge-base/verify.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class KnowledgeBaseUploadResponse(BaseModel):
    project_id: str
    doc_id: str
    filename: Optional[str] = None
    num_chunks: int


class KnowledgeBaseVerifyRequest(BaseModel):
    project_id: str = Field(..., description="Identifier of the knowledge base to check against.")
    question: str = Field(..., description="The question to verify against indexed documents.")
    n_results: int = Field(3, description="How many top matching chunks to consider.")


class KnowledgeBaseVerifyResponse(BaseModel):
    question: str
    supported: bool
    best_relevance_score: float
    threshold: float
    matched_chunks: List[str]
