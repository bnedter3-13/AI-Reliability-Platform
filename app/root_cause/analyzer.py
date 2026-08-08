"""
app/root_cause/analyzer.py

Root Cause Analysis (Component 2) + a lightweight Recommendation Engine (Component 4).

Approach: combine cheap, rule-based signals (which don't need an extra LLM call) with
the judge's own explanation text. This keeps latency and cost down — most Capstone-scale
traffic doesn't need a second full LLM call just to classify the failure type.

Extend this later by having the judge (in evaluator.py) directly return a `likely_cause`
field, or by adding a dedicated LLM call here for harder cases.
"""

from dataclasses import dataclass
from typing import List, Optional

from app.evaluation.evaluator import EvaluationResult


ROOT_CAUSES = {
    "no_context": "No relevant context was retrieved for this question.",
    "poor_retrieval": "Contexts were retrieved but appear weakly related to the question.",
    "hallucination": "The answer contains claims not supported by any retrieved context.",
    "unfaithful_answer": "The answer partially diverges from what the contexts state.",
    "low_correctness": "The answer does not adequately address the question, independent of sourcing.",
    "unknown": "Could not confidently determine a specific root cause — needs manual review.",
}

RECOMMENDATIONS = {
    "no_context": "Check the retrieval step: is the vector database populated and the query embedding correct?",
    "poor_retrieval": "Try increasing Top-K, adjusting chunk size, or switching the embedding model.",
    "hallucination": "Tighten the generation prompt to explicitly forbid unsupported claims; consider lowering temperature.",
    "unfaithful_answer": "Review the prompt for ambiguous instructions; consider adding few-shot examples of faithful answers.",
    "low_correctness": "Review whether the question itself is answerable from the available knowledge base.",
    "unknown": "Flag this case for manual review by the team.",
}


@dataclass
class RootCauseResult:
    cause: str
    explanation: str
    recommendation: str


def analyze_root_cause(
    result: EvaluationResult,
    contexts: List[str],
    context_relevance_avg: Optional[float] = None,
) -> RootCauseResult:
    """
    Classify the likely root cause of a failed/borderline evaluation using simple,
    explainable rules over the evaluation signals we already have.
    """
    if result.status == "pass":
        cause = "none"
        return RootCauseResult(
            cause=cause,
            explanation="Evaluation passed — no root cause analysis needed.",
            recommendation="No action needed.",
        )

    if not contexts:
        cause = "no_context"
    elif context_relevance_avg is not None and context_relevance_avg < 0.4:
        cause = "poor_retrieval"
    elif result.hallucination_risk >= 0.6:
        cause = "hallucination"
    elif result.faithfulness_score < 0.6:
        cause = "unfaithful_answer"
    elif result.correctness_score < 0.5:
        cause = "low_correctness"
    else:
        cause = "unknown"

    return RootCauseResult(
        cause=cause,
        explanation=ROOT_CAUSES[cause],
        recommendation=RECOMMENDATIONS[cause],
    )
