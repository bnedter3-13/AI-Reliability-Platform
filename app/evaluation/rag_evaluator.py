"""
app/evaluation/rag_evaluator.py

RAG Evaluation (Component 9): scores three things —
1. Context Relevance — is each retrieved passage relevant to the QUESTION?
2. Context Precision — was each retrieved passage actually USEFUL for the REFERENCE ANSWER?
3. Context Recall — did retrieval, as a whole, cover everything needed for the REFERENCE ANSWER?

Precision and Recall require a `reference_answer` (a known-correct answer) — without one,
only Context Relevance can be computed. For a stronger/faster version at scale, swap this
out for the RAGAS library, which implements the same metrics with more calibration.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from app.evaluation.evaluator import _client, _extract_json  # reuse the same Claude client
from app.evaluation.prompts import (
    CONTEXT_RELEVANCE_SYSTEM_PROMPT, build_context_relevance_message,
    CONTEXT_PRECISION_SYSTEM_PROMPT, build_context_precision_message,
    CONTEXT_RECALL_SYSTEM_PROMPT, build_context_recall_message,
)
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


# ---------------------------------------------------------------------------
# Context Precision — was each retrieved passage actually useful?
# ---------------------------------------------------------------------------

@dataclass
class ContextPrecisionResult:
    context: str
    is_useful: bool
    useful_score: float
    reason: str


def evaluate_context_precision(
    question: str, reference_answer: str, contexts: List[str]
) -> List[ContextPrecisionResult]:
    """Score each context passage for whether it was actually useful for the reference answer."""
    if _client is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

    results = []
    for context in contexts:
        try:
            message = _client.messages.create(
                model=settings.MODEL_NAME,
                max_tokens=200,
                system=CONTEXT_PRECISION_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": build_context_precision_message(question, reference_answer, context),
                }],
            )
            raw = _extract_json(message.content[0].text)
            results.append(ContextPrecisionResult(
                context=context,
                is_useful=raw["is_useful"],
                useful_score=raw["useful_score"],
                reason=raw["reason"],
            ))
        except Exception as exc:
            logger.error("evaluate_context_precision() failed for one context: %s", exc)
            results.append(ContextPrecisionResult(context=context, is_useful=False, useful_score=0.0, reason=f"Error: {exc}"))

    return results


def average_context_precision(results: List[ContextPrecisionResult]) -> float:
    """Precision = fraction of retrieved contexts that were actually useful."""
    if not results:
        return 0.0
    useful_count = sum(1 for r in results if r.is_useful)
    return round(useful_count / len(results), 3)


# ---------------------------------------------------------------------------
# Context Recall — did retrieval cover everything needed, as a whole?
# ---------------------------------------------------------------------------

@dataclass
class ContextRecallResult:
    recall_score: float
    covered_claims: int
    total_claims: int
    missing_info: str


def evaluate_context_recall(
    question: str, reference_answer: str, contexts: List[str]
) -> ContextRecallResult:
    """Score whether the full set of retrieved contexts covers everything in the reference answer."""
    if _client is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

    try:
        message = _client.messages.create(
            model=settings.MODEL_NAME,
            max_tokens=300,
            system=CONTEXT_RECALL_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": build_context_recall_message(question, reference_answer, contexts),
            }],
        )
        raw = _extract_json(message.content[0].text)
        return ContextRecallResult(
            recall_score=raw["recall_score"],
            covered_claims=raw["covered_claims"],
            total_claims=raw["total_claims"],
            missing_info=raw["missing_info"],
        )
    except Exception as exc:
        logger.error("evaluate_context_recall() failed: %s", exc)
        return ContextRecallResult(recall_score=0.0, covered_claims=0, total_claims=0, missing_info=f"Error: {exc}")


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

@dataclass
class RagEvaluationReport:
    context_relevance_avg: Optional[float]
    context_precision_avg: Optional[float]
    context_recall_score: Optional[float]
    recall_missing_info: Optional[str]

    def to_dict(self) -> dict:
        return {
            "context_relevance_avg": self.context_relevance_avg,
            "context_precision_avg": self.context_precision_avg,
            "context_recall_score": self.context_recall_score,
            "recall_missing_info": self.recall_missing_info,
        }


def run_full_rag_evaluation(
    question: str, contexts: List[str], reference_answer: Optional[str] = None
) -> RagEvaluationReport:
    """
    Convenience wrapper: always computes Context Relevance. If a reference_answer is
    provided, also computes Context Precision and Context Recall. This is the function
    to call from the API layer — it handles the "do we have a reference answer or not"
    branching so the caller doesn't have to.
    """
    relevance_results = evaluate_context_relevance(question, contexts)
    relevance_avg = average_context_relevance(relevance_results)

    if not reference_answer:
        return RagEvaluationReport(
            context_relevance_avg=relevance_avg,
            context_precision_avg=None,
            context_recall_score=None,
            recall_missing_info=None,
        )

    precision_results = evaluate_context_precision(question, reference_answer, contexts)
    precision_avg = average_context_precision(precision_results)

    recall_result = evaluate_context_recall(question, reference_answer, contexts)

    return RagEvaluationReport(
        context_relevance_avg=relevance_avg,
        context_precision_avg=precision_avg,
        context_recall_score=recall_result.recall_score,
        recall_missing_info=recall_result.missing_info or None,
    )
