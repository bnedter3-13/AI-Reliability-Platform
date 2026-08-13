"""
app/knowledge_base/verification_agent.py

Knowledge Base + Retrieval Verification Agent (Component 11): independently checks
whether a question is actually supported by the indexed knowledge base — regardless
of what a RAG app separately claims it retrieved. Queries the vector store directly
and compares the best match's relevance against a fixed threshold.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from app.knowledge_base.vector_store import KnowledgeBaseStore

# 0.65, not 0.8: chunks mix numeric data (tables, figures) with surrounding prose,
# which dilutes embedding relevance even when the chunk clearly answers the question.
# Real supported questions were observed scoring 0.59-0.68 against 0.8.
RELEVANCE_THRESHOLD = 0.65


@dataclass
class VerificationResult:
    question: str
    supported: bool
    best_relevance_score: float
    threshold: float = RELEVANCE_THRESHOLD
    matched_chunks: List[str] = field(default_factory=list)


def verify_question(
    project_id: str,
    question: str,
    n_results: int = 3,
    store: Optional[KnowledgeBaseStore] = None,
) -> VerificationResult:
    """
    Query the project's indexed knowledge base for `question` and decide whether
    it's supported: the best-matching chunk's relevance (1 - cosine distance) must
    meet RELEVANCE_THRESHOLD.
    """
    store = store or KnowledgeBaseStore()
    matches = store.query(project_id, question, n_results=n_results)

    if not matches:
        return VerificationResult(
            question=question,
            supported=False,
            best_relevance_score=0.0,
            matched_chunks=[],
        )

    scored = [(1.0 - m.distance, m.text) for m in matches]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_relevance_score = scored[0][0]

    return VerificationResult(
        question=question,
        supported=best_relevance_score >= RELEVANCE_THRESHOLD,
        best_relevance_score=best_relevance_score,
        matched_chunks=[text for _, text in scored],
    )
