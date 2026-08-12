"""
app/root_cause/smart_recommendation.py

Upgrades Component 4 (Recommendation Engine) from a static cause->recommendation
dictionary to a Claude-generated, situation-specific suggestion. The static
dictionary in analyzer.py stays as the fast, free, always-on fallback — this
module is an optional enhancement layered on top for a more tailored answer.

Kept as a separate module (not merged into evaluator.py) so it can be called
independently — e.g. only for failed evaluations, only on demand, or skipped
entirely if the team wants to keep costs minimal for the Capstone demo.
"""

import json
import re
import logging
from typing import Optional

from app.config import settings
from app.evaluation.evaluator import _client, _first_text_block

logger = logging.getLogger(__name__)


RECOMMENDATION_SYSTEM_PROMPT = """You are a senior AI/RAG systems engineer reviewing a \
failed evaluation from an AI Reliability Platform. You will be given the question, the \
answer the system produced, the classified root cause, and a brief explanation. Your job \
is to write ONE specific, actionable recommendation tailored to this exact case — not a \
generic tip. Reference specific details from the question/answer when relevant (e.g. \
"the query about X likely needs..." rather than "consider improving retrieval").

Keep it to 1-2 sentences. Be concrete: name a specific parameter, technique, or check \
where possible (e.g. "increase Top-K from the current value", "add a few-shot example \
showing how to say 'I don't know'", "check if the embedding model handles numeric \
queries like this one").

Respond with ONLY valid JSON, no extra text:
{
  "recommendation": "<your specific, tailored recommendation>"
}
"""


def generate_smart_recommendation(
    question: str,
    answer: str,
    cause: str,
    explanation: str,
    fallback_recommendation: str,
) -> str:
    """
    Ask Claude for a case-specific recommendation. Falls back to the static
    recommendation (already computed by analyze_root_cause) on any failure —
    this function should never be the reason a request fails.
    """
    if _client is None:
        logger.info("Smart recommendation skipped: no API client configured.")
        return fallback_recommendation

    if cause in ("none", "unknown"):
        return fallback_recommendation

    try:
        message = _client.messages.create(
            model=settings.MODEL_NAME,
            max_tokens=200,
            system=RECOMMENDATION_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    f"QUESTION:\n{question}\n\n"
                    f"ANSWER:\n{answer}\n\n"
                    f"ROOT CAUSE: {cause}\n"
                    f"EXPLANATION: {explanation}\n\n"
                    "Give your specific recommendation now."
                ),
            }],
        )
        text = _first_text_block(message)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object in response")
        raw = json.loads(match.group(0))
        return raw.get("recommendation") or fallback_recommendation
    except Exception as exc:
        logger.warning("Smart recommendation failed, using fallback: %s", exc)
        return fallback_recommendation
