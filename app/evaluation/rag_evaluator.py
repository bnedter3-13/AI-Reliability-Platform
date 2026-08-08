"""
app/evaluation/rag_evaluator.py

Basic RAG Evaluation (Component 9): scores how relevant each retrieved context
passage is to the question. This is a starting point — for a stronger version,
add Context Precision/Recall once you have reference answers to compare against,
or swap this out for the RAGAS library.
"""

import logging
from dataclasses import dataclass
from typing import List

from app.evaluation.evaluator import _client, _extract_json  # reuse the same Claude client
from app.evaluation.prompts import CONTEXT_RELEVANCE_SYSTEM_PROMPT, build_context_relevance_message
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ContextRelevanceResult:
    context: str
    relevance_score: float
    reason: str


def evaluate_context_relevance(question: str, contexts: List[str]) -> List[ContextRelevanceResult]:
    """Score each context passage individually for relevance to the question."""
    if _client is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

    results = []
    for context in contexts:
        try:
            message = _client.messages.create(
                model=settings.MODEL_NAME,
                max_tokens=200,
                system=CONTEXT_RELEVANCE_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": build_context_relevance_message(question, context),
                }],
            )
            raw = _extract_json(message.content[0].text)
            results.append(ContextRelevanceResult(
                context=context,
                relevance_score=raw["relevance_score"],
                reason=raw["reason"],
            ))
        except Exception as exc:
            logger.error("evaluate_context_relevance() failed for one context: %s", exc)
            results.append(ContextRelevanceResult(context=context, relevance_score=0.0, reason=f"Error: {exc}"))

    return results


def average_context_relevance(results: List[ContextRelevanceResult]) -> float:
    if not results:
        return 0.0
    return round(sum(r.relevance_score for r in results) / len(results), 3)
