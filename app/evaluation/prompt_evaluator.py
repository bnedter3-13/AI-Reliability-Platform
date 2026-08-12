"""
app/evaluation/prompt_evaluator.py

Prompt Evaluation (Component 7): analyzes the quality of a PROMPT itself — not an
answer. Useful as a standalone developer tool: paste a system prompt or generation
prompt you're about to ship, and get feedback on clarity, completeness, and
hallucination risk before it ever reaches production traffic.

This is intentionally NOT wired into the automatic per-request evaluation pipeline
(evaluate_answer) — it's a separate, on-demand check a developer runs while writing
or revising a prompt, per the original component spec.
"""

import json
import re
import logging
from dataclasses import dataclass, asdict
from typing import Optional

from app.config import settings
from app.evaluation.evaluator import _client, _first_text_block

logger = logging.getLogger(__name__)


PROMPT_EVAL_SYSTEM_PROMPT = """You are a senior prompt engineer reviewing a PROMPT that \
will be used to instruct an LLM in production (this could be a system prompt or a \
generation prompt template). You are NOT answering the prompt — you are critiquing its \
quality as an instruction.

Evaluate it on:
- clarity_score (0.0-1.0): is the instruction unambiguous and easy for an LLM to follow?
- completeness_score (0.0-1.0): does it specify what to do with edge cases (e.g. missing \
  information, out-of-scope questions), format, and tone? Incomplete prompts leave gaps \
  the model has to guess at.
- hallucination_risk_score (0.0-1.0): how likely is this prompt, as written, to cause the \
  model to fabricate information? (e.g. no instruction to say "I don't know", no \
  constraint to stick to provided context, encourages elaboration beyond the facts)

Then give:
- issues: a list of specific problems found (empty list if none)
- suggested_rewrite: a concretely improved version of the prompt (if issues were found), \
  or empty string if the prompt is already strong

Respond with ONLY valid JSON in this exact shape, no extra text, no markdown code fences.
Inside string values, do NOT escape apostrophes with a backslash (write "don't" not "don\\'t") —
only escape double quotes and backslashes, per standard JSON rules:
{
  "clarity_score": <float>,
  "completeness_score": <float>,
  "hallucination_risk_score": <float>,
  "issues": ["<issue 1>", "<issue 2>", ...],
  "suggested_rewrite": "<improved prompt, or empty string>"
}
"""


@dataclass
class PromptEvaluationResult:
    clarity_score: float
    completeness_score: float
    hallucination_risk_score: float
    issues: list
    suggested_rewrite: str

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text!r}")

    raw = match.group(0)

    # Try a sequence of increasingly permissive parses. LLM-generated JSON
    # commonly breaks strict json.loads() in one of these specific ways:
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


def evaluate_prompt(prompt_text: str) -> PromptEvaluationResult:
    """
    Evaluate the quality of a prompt (not an answer). Raises RuntimeError if no
    API key is configured — same fail-loudly convention as evaluate_answer().
    """
    if _client is None:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured. Set it in your environment "
            "before calling evaluate_prompt()."
        )
    if not prompt_text or not prompt_text.strip():
        raise ValueError("prompt_text must not be empty.")

    message = _client.messages.create(
        model=settings.MODEL_NAME,
        max_tokens=1024,
        system=PROMPT_EVAL_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"PROMPT TO EVALUATE:\n\n{prompt_text}\n\nEvaluate this prompt now.",
        }],
    )
    raw = _extract_json(_first_text_block(message))
    return PromptEvaluationResult(
        clarity_score=_coerce_score(raw.get("clarity_score")),
        completeness_score=_coerce_score(raw.get("completeness_score")),
        hallucination_risk_score=_coerce_score(raw.get("hallucination_risk_score")),
        issues=_coerce_issues(raw.get("issues")),
        suggested_rewrite=str(raw.get("suggested_rewrite") or ""),
    )


def _coerce_score(value) -> float:
    """Defensively convert a judge-provided score to a float in [0, 1]. Never raises —
    a malformed/missing score becomes 0.0 rather than crashing the whole request."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _coerce_issues(value) -> list:
    """The judge should return a list of strings, but defensively handle a single
    string or None so a shape mismatch doesn't crash the request."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
