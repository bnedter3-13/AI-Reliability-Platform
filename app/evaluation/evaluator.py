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
from app.evaluation.prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_PROMPT_VERSION,
    GENERATION_PROMPT_VERSION,
    COMPARATOR_VERSION,
    build_judge_message,
)

logger = logging.getLogger(__name__)

# MLOps Integration (Component 10): a version identifier tagged onto every evaluation
# so performance can be compared across versions later. Bump JUDGE_PROMPT_VERSION and/or
# GENERATION_PROMPT_VERSION in prompts.py whenever the judge prompt or the generation
# prompt (GENERATION_SYSTEM_PROMPT in app/comparison/model_comparator.py) changes
# meaningfully, and COMPARATOR_VERSION in app/comparison/model_comparator.py whenever
# compare_models()'s code-level behavior changes (e.g. retry logic) without a prompt
# wording change - each is tracked independently here so a change to one doesn't mask
# a change to the others in comparisons. This composite string also changes
# automatically if the model itself changes.
EVALUATOR_VERSION = (
    f"{settings.MODEL_NAME}:gen{GENERATION_PROMPT_VERSION}:judge{JUDGE_PROMPT_VERSION}:cmp{COMPARATOR_VERSION}"
)

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
    status: str  # "pass" | "warning" | "fail" | "error" (evaluator failure, not a judge verdict)
    explanation: str
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text!r}")

    raw = match.group(0)

    # Try a sequence of increasingly permissive parses. LLM-generated JSON commonly
    # breaks strict json.loads() in one of these specific ways:
    #  1. Perfectly valid — parses immediately.
    #  2. Apostrophes escaped as \' inside strings (not a valid JSON escape).
    #  3. Literal newlines/control characters inside string values instead of \n.
    attempts = [
        lambda s: json.loads(s),
        lambda s: json.loads(re.sub(r"\\'", "'", s)),
        lambda s: json.loads(re.sub(r"\\'", "'", s), strict=False),
        lambda s: json.loads(s, strict=False),
    ]
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            return attempt(raw)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

    raise ValueError(f"Could not parse judge response as JSON after repair attempts: {last_error}")


def _first_text_block(message) -> str:
    """content[0] isn't reliably the text block - the model sometimes emits a
    leading ThinkingBlock (extended thinking) before the TextBlock, which has
    no .text attribute. Find the first actual text block instead. Shared by
    every module in this codebase that parses a Claude message for JSON."""
    response_text = next(
        (block.text for block in message.content if getattr(block, "type", None) == "text"),
        None,
    )
    if response_text is None:
        block_types = [getattr(block, "type", type(block).__name__) for block in message.content]
        raise ValueError(f"No text block found in response (block types: {block_types!r})")
    return response_text


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
        raw = _extract_json(_first_text_block(message))
        # Constructing EvaluationResult here (inside the try) matters: _extract_json
        # only guarantees valid JSON syntax, not that it has the exact keys this
        # dataclass expects. A judge response that's valid JSON but missing/misnamed
        # a key (e.g. no "explanation") raises TypeError - that must be caught below
        # like any other judge failure, not propagate uncaught to the caller.
        return EvaluationResult(**raw, latency_ms=round(latency_ms, 1))

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.error("evaluate_answer() failed: %s", exc)
        # Fail-safe: on any evaluator error (malformed or wrong-shaped judge JSON,
        # API error, rate limit, timeout, etc.) return status="error" rather than
        # crashing the whole request. This is deliberately distinct from "fail" -
        # it's not a real judge verdict, so it can be filtered out of pass/fail
        # statistics instead of silently counted as the worst possible score.
        return EvaluationResult(
            correctness_score=0.0,
            faithfulness_score=0.0,
            hallucination_risk=1.0,
            status="error",
            explanation=f"Evaluator error (not a judge verdict): {exc}",
            latency_ms=round(latency_ms, 1),
        )
