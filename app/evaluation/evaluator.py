"""
app/evaluation/evaluator.py

Core evaluation module for the AI Reliability Platform (AI Doctor).
Implements evaluate_answer() using an LLM-as-a-Judge approach with Claude.
"""

import json
import re
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional

import anthropic

from app.config import settings
from app.evaluation.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_PROMPT_VERSION, build_judge_message

logger = logging.getLogger(__name__)

# MLOps Integration (Component 10): a version identifier tagged onto every evaluation
# so performance can be compared across versions later. Bump JUDGE_PROMPT_VERSION in
# prompts.py whenever the judge prompt changes meaningfully; this composite string
# also changes automatically if the model itself changes.
EVALUATOR_VERSION = f"{settings.MODEL_NAME}:{JUDGE_PROMPT_VERSION}"

_client: Optional[anthropic.Anthropic] = (
    anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY) if settings.ANTHROPIC_API_KEY else None
)

if _client is None:
    logger.warning(
        "ANTHROPIC_API_KEY is not set. evaluate_answer() will raise at call time. "
        "Set it in your .env file (see .env.example) before running the API."
    )


@dataclass
class EvaluationResult:
    correctness_score: float
    faithfulness_score: float
    hallucination_risk: float
    status: str  # "pass" | "warning" | "fail"
    explanation: str
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_json(text: str) -> dict:
    """Judges sometimes wrap JSON in prose or code fences — pull the JSON block out safely."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text!r}")
    return json.loads(match.group(0))


def evaluate_answer(
    answer: str,
    contexts: List[str],
    reference_answer: Optional[str] = None,
    question: Optional[str] = None,
) -> EvaluationResult:
    """
    Evaluate a single RAG answer for correctness, faithfulness, and hallucination risk.

    Raises RuntimeError if no API key is configured — this is intentional so the API
    fails loudly in production rather than silently returning fake scores.
    """
    if _client is None:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured. Set it in your environment "
            "before calling evaluate_answer()."
        )

    question_text = question or "(question not provided — evaluate the answer against the contexts alone)"

    start = time.perf_counter()
    try:
        message = _client.messages.create(
            model=settings.MODEL_NAME,
            max_tokens=500,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": build_judge_message(question_text, contexts, answer),
            }],
        )
        latency_ms = (time.perf_counter() - start) * 1000
        response_text = message.content[0].text
        raw = _extract_json(response_text)

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.error("evaluate_answer() failed: %s", exc)
        # Fail-safe: treat evaluator errors as a "fail" status rather than crashing
        # the whole request, so one bad judge call doesn't take down the endpoint.
        raw = {
            "correctness_score": 0.0,
            "faithfulness_score": 0.0,
            "hallucination_risk": 1.0,
            "status": "fail",
            "explanation": f"Evaluator error, treated as fail-safe: {exc}",
        }

    return EvaluationResult(**raw, latency_ms=round(latency_ms, 1))
