"""
app/comparison/model_comparator.py

Model Comparison (Component 8): runs the same question through multiple models -
across Anthropic, OpenAI, Google Gemini, and Qwen (via OpenRouter) - evaluates each
generated answer with the same judge, and compares quality, speed, and estimated
cost.

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
from app.evaluation.evaluator import evaluate_answer, _first_text_block

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
    {"id": "qwen/qwen-2.5-72b-instruct", "label": "Qwen 2.5 72B", "provider": "qwen"},
    {"id": "qwen/qwen-plus", "label": "Qwen Plus", "provider": "qwen"},
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
    "qwen/qwen-2.5-72b-instruct": {"input": 0.36, "output": 0.40},  # per openrouter.ai/api/v1/models, Aug 2026
    "qwen/qwen-plus": {"input": 0.40, "output": 1.20},
}

GENERATION_SYSTEM_PROMPT = (
    "You are a strict, evidence-based assistant. You MUST answer the user's "
    "question using ONLY information explicitly stated in the provided "
    "contexts, if any are given. Never add facts, numbers, dates, or details "
    "that are not stated there — even if they seem true, well-known, or "
    "obviously correct. This rule has no exceptions.\n\n"
    "If the contexts are missing, or don't contain enough information to "
    "answer the question, you MUST say so explicitly and clearly (e.g. "
    "'The provided context does not contain this information'). Do NOT "
    "fall back on general knowledge, do NOT guess, and do NOT answer the "
    "question anyway 'to be helpful.' A correct refusal is always better "
    "than an unsupported answer.\n\n"
    "If asked to verify a claim, check it strictly against the contexts "
    "only — not against what sounds plausible or familiar.\n\n"
    "Be concise. Stay directly and verifiably grounded in the contexts "
    "provided at all times."
)

# Appended as a follow-up user turn when retrying generation after a judge
# "fail" verdict (see compare_models' retry-with-clarification step below).
RETRY_CORRECTIVE_MESSAGE = (
    "Your previous answer may not have been strictly grounded in the provided "
    "context. Please review it: confirm every claim is explicitly supported by "
    "the context, or clearly state if the context is insufficient to answer."
)

# Appended as a follow-up user turn for the *second* retry, i.e. when the first
# retry's answer still evaluates to "fail". More pointed than
# RETRY_CORRECTIVE_MESSAGE above: it asks the model to check each specific claim
# individually rather than repeating the same general grounding review that
# already failed to fix the answer once.
RETRY_CORRECTIVE_MESSAGE_2 = (
    "Your answer may still contain a specific detail (a number, name, date, or "
    "fact) that is not literally present in the provided context. Re-check each "
    "specific claim individually against the context word-by-word, and remove or "
    "flag anything you cannot directly verify there."
)

# MLOps Integration (Component 10): the version identifier for this module's
# code-level behavior (currently: the retry-with-clarification mechanism below)
# lives in app.evaluation.prompts as COMPARATOR_VERSION, not here - same reason
# GENERATION_PROMPT_VERSION lives there instead of in this file (see that
# constant's comment): evaluator.py needs to import it into EVALUATOR_VERSION,
# and this module already imports evaluate_answer from evaluator.py, so defining
# it here would create a circular import. Bump it whenever compare_models()'s
# behavior changes at the code level, independent of prompt wording changes.


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
    explanation: Optional[str] = None

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
#
# `previous_answer`, when set, means this is the retry-with-clarification call
# (see compare_models): the model's own prior answer plus RETRY_CORRECTIVE_MESSAGE
# are appended as a follow-up turn instead of sending the question fresh.
# ---------------------------------------------------------------------------

def _generate_with_anthropic(
    model_id: str, question: str, contexts: List[str], previous_answer: Optional[str] = None,
    corrective_message: str = RETRY_CORRECTIVE_MESSAGE,
) -> Tuple[str, float, int, int]:
    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": _build_generation_message(question, contexts)}]
    if previous_answer is not None:
        messages.append({"role": "assistant", "content": previous_answer})
        messages.append({"role": "user", "content": corrective_message})

    start = time.perf_counter()
    message = client.messages.create(
        model=model_id,
        max_tokens=400,
        system=GENERATION_SYSTEM_PROMPT,
        messages=messages,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    answer_text = _first_text_block(message)
    input_tokens = getattr(message.usage, "input_tokens", 0)
    output_tokens = getattr(message.usage, "output_tokens", 0)
    return answer_text, latency_ms, input_tokens, output_tokens


def _generate_with_openai(
    model_id: str, question: str, contexts: List[str], previous_answer: Optional[str] = None,
    corrective_message: str = RETRY_CORRECTIVE_MESSAGE,
) -> Tuple[str, float, int, int]:
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": _build_generation_message(question, contexts)},
    ]
    if previous_answer is not None:
        messages.append({"role": "assistant", "content": previous_answer})
        messages.append({"role": "user", "content": corrective_message})

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=400,
        messages=messages,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    answer_text = response.choices[0].message.content
    input_tokens = getattr(response.usage, "prompt_tokens", 0)
    output_tokens = getattr(response.usage, "completion_tokens", 0)
    return answer_text, latency_ms, input_tokens, output_tokens


def _generate_with_gemini(
    model_id: str, question: str, contexts: List[str], previous_answer: Optional[str] = None,
    corrective_message: str = RETRY_CORRECTIVE_MESSAGE,
) -> Tuple[str, float, int, int]:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    # Uses the current `google-genai` SDK (pip install google-genai). The older
    # `google-generativeai` package is deprecated by Google - don't use it.
    from google import genai as google_genai

    client = google_genai.Client(api_key=settings.GEMINI_API_KEY)

    if previous_answer is not None:
        contents = [
            {"role": "user", "parts": [{"text": _build_generation_message(question, contexts)}]},
            {"role": "model", "parts": [{"text": previous_answer}]},
            {"role": "user", "parts": [{"text": corrective_message}]},
        ]
    else:
        contents = _build_generation_message(question, contexts)

    start = time.perf_counter()
    response = client.models.generate_content(
        model=model_id,
        contents=contents,
        config={"system_instruction": GENERATION_SYSTEM_PROMPT},
    )
    latency_ms = (time.perf_counter() - start) * 1000

    answer_text = response.text
    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
    return answer_text, latency_ms, input_tokens, output_tokens


def _generate_with_qwen(
    model_id: str, question: str, contexts: List[str], previous_answer: Optional[str] = None,
    corrective_message: str = RETRY_CORRECTIVE_MESSAGE,
) -> Tuple[str, float, int, int]:
    if not settings.QWEN_API_KEY:
        raise RuntimeError("QWEN_API_KEY is not configured (expects an OpenRouter API key).")
    # Qwen is accessed via OpenRouter (openrouter.ai), which exposes an
    # OpenAI-compatible API - reuse the `openai` SDK with a different base_url
    # instead of a separate provider SDK. Model IDs use OpenRouter's
    # "qwen/<model-name>" format (e.g. "qwen/qwen-2.5-72b-instruct"). Free-tier
    # (":free") slugs come and go as OpenRouter adjusts availability - check
    # openrouter.ai/models for the current list before relying on one.
    from openai import OpenAI

    client = OpenAI(api_key=settings.QWEN_API_KEY, base_url="https://openrouter.ai/api/v1")
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": _build_generation_message(question, contexts)},
    ]
    if previous_answer is not None:
        messages.append({"role": "assistant", "content": previous_answer})
        messages.append({"role": "user", "content": corrective_message})

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=400,
        messages=messages,
        # Novita's routing for this model incorrectly rejects chat-format
        # requests ("does not support endpoint: completions") even though the
        # request is sent to /chat/completions - excluding it here so OpenRouter
        # only falls back to providers that actually work for this model.
        extra_body={"provider": {"ignore": ["Novita"]}},
    )
    latency_ms = (time.perf_counter() - start) * 1000

    error = getattr(response, "error", None)
    if error:
        raise RuntimeError(f"OpenRouter error: {error.get('message', error)}")
    if not response.choices:
        raise RuntimeError("OpenRouter returned no choices - the model may be rate-limited or unavailable.")

    answer_text = response.choices[0].message.content
    input_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
    output_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0
    return answer_text, latency_ms, input_tokens, output_tokens


PROVIDER_GENERATORS: dict = {
    "anthropic": _generate_with_anthropic,
    "openai": _generate_with_openai,
    "gemini": _generate_with_gemini,
    "qwen": _generate_with_qwen,
}


def _generate_answer(
    model_id: str, question: str, contexts: List[str], previous_answer: Optional[str] = None,
    corrective_message: str = RETRY_CORRECTIVE_MESSAGE,
) -> Tuple[str, float, int, int]:
    provider = PROVIDER_BY_MODEL.get(model_id)
    if not provider:
        raise ValueError(f"Unknown model_id: {model_id!r}. Add it to AVAILABLE_MODELS first.")
    generator: Callable = PROVIDER_GENERATORS[provider]
    return generator(model_id, question, contexts, previous_answer, corrective_message)


def _safe_evaluate(
    model_id: str, answer: str, contexts: List[str],
    reference_answer: Optional[str], question: str,
):
    """evaluate_answer(), but logs and returns None instead of raising - callers
    treat None the same as the pre-existing "judge evaluation failed" case."""
    try:
        return evaluate_answer(
            answer=answer, contexts=contexts,
            reference_answer=reference_answer, question=question,
        )
    except Exception as exc:
        logger.warning("Model Comparison: judge evaluation failed for %s: %s", model_id, exc)
        return None


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

    A judge "fail" verdict triggers a retry-with-clarification: the model is
    re-prompted with its own prior answer plus a corrective follow-up, and the
    retried answer is re-evaluated. If that still fails, one further retry is
    made with a more pointed, claim-by-claim corrective message. Whichever
    attempt is the last one made (up to three generation calls total) has its
    answer/scores/status replace all earlier attempts entirely - see
    `explanation` for how many retries were needed, if any. `generation_latency_ms`,
    `input_tokens`, `output_tokens`, and `estimated_cost_usd` always reflect the
    sum of every attempt actually made.
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

        eval_result = _safe_evaluate(model_id, answer_text, contexts, reference_answer, question)
        if eval_result is not None:
            correctness = eval_result.correctness_score
            faithfulness = eval_result.faithfulness_score
            hallucination = eval_result.hallucination_risk
            status = eval_result.status
            explanation = eval_result.explanation
        else:
            correctness = faithfulness = hallucination = status = explanation = None

        # Retry-with-clarification: a judge "fail" verdict (not "error" - that's an
        # evaluator failure, not a real verdict) gets up to two retries. The first
        # retry uses the model's own prior answer plus a general grounding-review
        # corrective message; if that still fails, a second retry uses a more
        # pointed, claim-by-claim corrective message instead. Whichever retry is
        # the last one made has its answer/scores/status *replace* every earlier
        # attempt entirely; latency, token, and cost fields accumulate every
        # attempt actually made either way.
        if status == "fail":
            try:
                retry_answer, retry_latency_ms, retry_input_tokens, retry_output_tokens = _generate_answer(
                    model_id, question, contexts, previous_answer=answer_text,
                )
            except Exception as exc:
                logger.error("Model Comparison: retry generation failed for %s: %s", model_id, exc)
            else:
                gen_latency_ms += retry_latency_ms
                input_tokens += retry_input_tokens
                output_tokens += retry_output_tokens
                answer_text = retry_answer

                retry_eval = _safe_evaluate(model_id, retry_answer, contexts, reference_answer, question)
                if retry_eval is not None:
                    correctness = retry_eval.correctness_score
                    faithfulness = retry_eval.faithfulness_score
                    hallucination = retry_eval.hallucination_risk
                    status = retry_eval.status

                    if status == "fail":
                        try:
                            (
                                retry2_answer, retry2_latency_ms,
                                retry2_input_tokens, retry2_output_tokens,
                            ) = _generate_answer(
                                model_id, question, contexts, previous_answer=retry_answer,
                                corrective_message=RETRY_CORRECTIVE_MESSAGE_2,
                            )
                        except Exception as exc:
                            logger.error(
                                "Model Comparison: second retry generation failed for %s: %s", model_id, exc
                            )
                            explanation = f"[Retry attempted twice, still failing] {retry_eval.explanation}"
                        else:
                            gen_latency_ms += retry2_latency_ms
                            input_tokens += retry2_input_tokens
                            output_tokens += retry2_output_tokens
                            answer_text = retry2_answer

                            retry2_eval = _safe_evaluate(
                                model_id, retry2_answer, contexts, reference_answer, question
                            )
                            if retry2_eval is not None:
                                correctness = retry2_eval.correctness_score
                                faithfulness = retry2_eval.faithfulness_score
                                hallucination = retry2_eval.hallucination_risk
                                status = retry2_eval.status
                                if status == "fail":
                                    explanation = f"[Retry attempted twice, still failing] {retry2_eval.explanation}"
                                elif status == "error":
                                    explanation = retry2_eval.explanation
                                else:
                                    explanation = f"[Corrected after 2 retries] {retry2_eval.explanation}"
                            else:
                                correctness = faithfulness = hallucination = status = None
                                explanation = (
                                    "[Retry attempted, evaluation failed] Could not re-score the "
                                    "retried answer - see server logs."
                                )
                    elif status == "error":
                        # Evaluator itself failed on the retry - not a real verdict,
                        # so no retry-outcome prefix is warranted.
                        explanation = retry_eval.explanation
                    else:
                        explanation = f"[Corrected after 1 retry] {retry_eval.explanation}"
                else:
                    # Retry generation succeeded (retry_answer/answer_text above are
                    # already updated) but re-evaluating it crashed. Don't blank that
                    # away silently - a retry genuinely happened, so say so, even
                    # though we have no scores to show for it.
                    correctness = faithfulness = hallucination = status = None
                    explanation = (
                        "[Retry attempted, evaluation failed] Could not re-score the "
                        "retried answer - see server logs."
                    )

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
            explanation=explanation,
        ))

    return results
