"""
app/comparison/model_comparator.py

Model Comparison (Component 8): runs the same question through multiple models -
across Anthropic, OpenAI, and Google Gemini - evaluates each generated answer with
the same judge, and compares quality, speed, and estimated cost.

Adding another provider later: write a new _generate_with_<provider>() function
with the same signature as the existing ones, add its models to AVAILABLE_MODELS
with the matching "provider" tag, and add it to PROVIDER_GENERATORS. Nothing else
needs to change - evaluation, cost estimate, and the comparison table all key off
that dict.

IMPORTANT: model IDs below were current as of this code being written (Aug 2026).
Provider model lineups change fast:
- OpenAI: gpt-4o / gpt-4o-mini remain API-accessible legacy models; OpenAI's current
  flagship line is the GPT-5.x family (e.g. gpt-5.4) - check
  platform.openai.com/docs/models for exact API model strings before using those.
- Gemini: gemini-2.5-flash / gemini-2.5-pro are deprecated - live calls to
  gemini-2.5-flash return a 404 ("no longer available to new users"). Current
  flagships confirmed working live are gemini-3.6-flash and gemini-3.5-flash;
  check ai.google.dev/gemini-api/docs/models for the latest before assuming
  these are still current.
Swap AVAILABLE_MODELS entries below as needed - nothing else in this file depends
on the specific IDs.
"""

import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Tuple

from app.config import settings
from app.evaluation.evaluator import evaluate_answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Available models across providers
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = [
    {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5", "provider": "anthropic"},
    {"id": "claude-sonnet-5", "label": "Claude Sonnet 5", "provider": "anthropic"},
    {"id": "claude-opus-4-8", "label": "Claude Opus 4.8", "provider": "anthropic"},
    {"id": "gpt-4o-mini", "label": "GPT-4o mini", "provider": "openai"},
    {"id": "gpt-4o", "label": "GPT-4o", "provider": "openai"},
    {"id": "gemini-3.6-flash", "label": "Gemini 3.6 Flash", "provider": "gemini"},
    {"id": "gemini-3.5-flash", "label": "Gemini 3.5 Flash", "provider": "gemini"},
]
PROVIDER_BY_MODEL = {m["id"]: m["provider"] for m in AVAILABLE_MODELS}
LABEL_BY_MODEL = {m["id"]: m["label"] for m in AVAILABLE_MODELS}

# Approximate USD price per million tokens (input, output). For RELATIVE cost
# comparison between models in this tool, not for billing - prices change; check
# each provider's pricing page for current, exact numbers.
APPROX_PRICING_PER_MILLION_TOKENS = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-opus-4-8": {"input": 15.00, "output": 75.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gemini-3.6-flash": {"input": 1.50, "output": 7.50},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
}

GENERATION_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the "
    "information in the provided contexts, if any are given. Be concise."
)


@dataclass
class ModelComparisonResult:
    model_id: str
    model_label: str
    provider: str
    answer: str
    generation_latency_ms: float
    correctness_score: Optional[float]
    faithfulness_score: Optional[float]
    hallucination_risk: Optional[float]
    status: Optional[str]
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _build_generation_message(question: str, contexts: List[str]) -> str:
    if not contexts:
        return f"Question: {question}"
    context_block = "\n".join(f"- {c}" for c in contexts)
    return f"Contexts:\n{context_block}\n\nQuestion: {question}"


def _estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    pricing = APPROX_PRICING_PER_MILLION_TOKENS.get(model_id)
    if not pricing:
        return 0.0
    cost = (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]
    return round(cost, 6)


# ---------------------------------------------------------------------------
# Per-provider generation functions.
# Each returns (answer_text, latency_ms, input_tokens, output_tokens).
# Clients are created lazily inside each function (not at import time) so a
# missing API key for provider B doesn't block using provider A at all.
# ---------------------------------------------------------------------------

def _generate_with_anthropic(model_id: str, question: str, contexts: List[str]) -> Tuple[str, float, int, int]:
    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    start = time.perf_counter()
    message = client.messages.create(
        model=model_id,
        max_tokens=400,
        system=GENERATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_generation_message(question, contexts)}],
    )
    latency_ms = (time.perf_counter() - start) * 1000
    answer_text = message.content[0].text
    input_tokens = getattr(message.usage, "input_tokens", 0)
    output_tokens = getattr(message.usage, "output_tokens", 0)
    return answer_text, latency_ms, input_tokens, output_tokens


def _generate_with_openai(model_id: str, question: str, contexts: List[str]) -> Tuple[str, float, int, int]:
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=400,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": _build_generation_message(question, contexts)},
        ],
    )
    latency_ms = (time.perf_counter() - start) * 1000
    answer_text = response.choices[0].message.content
    input_tokens = getattr(response.usage, "prompt_tokens", 0)
    output_tokens = getattr(response.usage, "completion_tokens", 0)
    return answer_text, latency_ms, input_tokens, output_tokens


def _generate_with_gemini(model_id: str, question: str, contexts: List[str]) -> Tuple[str, float, int, int]:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    # Uses the current `google-genai` SDK (pip install google-genai). The older
    # `google-generativeai` package is deprecated by Google - don't use it.
    from google import genai as google_genai

    client = google_genai.Client(api_key=settings.GEMINI_API_KEY)

    start = time.perf_counter()
    response = client.models.generate_content(
        model=model_id,
        contents=_build_generation_message(question, contexts),
        config={"system_instruction": GENERATION_SYSTEM_PROMPT},
    )
    latency_ms = (time.perf_counter() - start) * 1000

    answer_text = response.text
    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
    return answer_text, latency_ms, input_tokens, output_tokens


PROVIDER_GENERATORS: dict = {
    "anthropic": _generate_with_anthropic,
    "openai": _generate_with_openai,
    "gemini": _generate_with_gemini,
}


def _generate_answer(model_id: str, question: str, contexts: List[str]) -> Tuple[str, float, int, int]:
    provider = PROVIDER_BY_MODEL.get(model_id)
    if not provider:
        raise ValueError(f"Unknown model_id: {model_id!r}. Add it to AVAILABLE_MODELS first.")
    generator: Callable = PROVIDER_GENERATORS[provider]
    return generator(model_id, question, contexts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_models(
    question: str,
    contexts: Optional[List[str]] = None,
    reference_answer: Optional[str] = None,
    model_ids: Optional[List[str]] = None,
) -> List[ModelComparisonResult]:
    """
    Run the same question through each requested model (any mix of providers),
    evaluate each answer with the standard Claude judge (evaluate_answer) so
    quality scores stay comparable across providers, and return one result per
    model. A model that fails (missing key, API error) shows up with its `error`
    field set rather than stopping the others.
    """
    contexts = contexts or []
    models_to_run = model_ids or [m["id"] for m in AVAILABLE_MODELS]

    results: List[ModelComparisonResult] = []

    for model_id in models_to_run:
        label = LABEL_BY_MODEL.get(model_id, model_id)
        provider = PROVIDER_BY_MODEL.get(model_id, "unknown")

        try:
            answer_text, gen_latency_ms, input_tokens, output_tokens = _generate_answer(
                model_id, question, contexts
            )
        except Exception as exc:
            logger.error("Model Comparison: generation failed for %s: %s", model_id, exc)
            results.append(ModelComparisonResult(
                model_id=model_id, model_label=label, provider=provider, answer="",
                generation_latency_ms=0.0, correctness_score=None, faithfulness_score=None,
                hallucination_risk=None, status=None, input_tokens=0, output_tokens=0,
                estimated_cost_usd=0.0, error=str(exc),
            ))
            continue

        try:
            eval_result = evaluate_answer(
                answer=answer_text, contexts=contexts,
                reference_answer=reference_answer, question=question,
            )
            correctness = eval_result.correctness_score
            faithfulness = eval_result.faithfulness_score
            hallucination = eval_result.hallucination_risk
            status = eval_result.status
        except Exception as exc:
            logger.warning("Model Comparison: judge evaluation failed for %s: %s", model_id, exc)
            correctness = faithfulness = hallucination = None
            status = None

        results.append(ModelComparisonResult(
            model_id=model_id,
            model_label=label,
            provider=provider,
            answer=answer_text,
            generation_latency_ms=round(gen_latency_ms, 1),
            correctness_score=correctness,
            faithfulness_score=faithfulness,
            hallucination_risk=hallucination,
            status=status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=_estimate_cost(model_id, input_tokens, output_tokens),
        ))

    return results
