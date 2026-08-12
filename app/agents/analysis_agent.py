"""
app/agents/analysis_agent.py

AI Analysis Agent (Component 3): unlike the per-evaluation `explanation` field
(one sentence about a single answer), this agent looks ACROSS multiple recent
evaluations and reports on recurring patterns — e.g. "hallucination keeps
happening on questions about X" or "poor_retrieval spikes after 2pm" — the
kind of insight that only emerges from looking at several failures together.

It is intentionally a separate, on-demand call (not run on every request) since
it needs a batch of evaluations to find a pattern in, and costs more tokens
than a single evaluation.
"""

import json
import re
import logging
from dataclasses import dataclass
from typing import List

from sqlalchemy.orm import Session

from app.config import settings
from app.database.models import EvaluationRecord
from app.evaluation.evaluator import _client, _first_text_block  # reuse the same Claude client

logger = logging.getLogger(__name__)


ANALYSIS_SYSTEM_PROMPT = """You are a senior AI reliability analyst. You will be given a batch \
of recent evaluation records from an AI Reliability Platform, each with a question, status, \
scores, and a root cause label. Your job is to find PATTERNS across the batch — not to repeat \
what a single record already says.

Look for things like:
- A root cause that repeats disproportionately often
- A topic or type of question that keeps failing
- Whether failures cluster together or are scattered/random
- Any early signal of a systemic issue (e.g. retrieval consistently failing, not just one bad answer)

Respond with ONLY valid JSON in this exact shape, no extra text, no markdown code fences:
{
  "summary": "<2-3 sentence overview of what's going on across this batch>",
  "dominant_issue": "<the single most common root cause or theme, or 'none' if evaluations are mostly passing>",
  "pattern_detected": <true|false>,
  "pattern_description": "<1-2 sentences on the specific pattern found, or empty string if none>",
  "severity": "<low|medium|high>",
  "suggested_action": "<one concrete next step for the team>"
}
"""


@dataclass
class AnalysisReport:
    summary: str
    dominant_issue: str
    pattern_detected: bool
    pattern_description: str
    severity: str
    suggested_action: str
    sample_size: int

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "dominant_issue": self.dominant_issue,
            "pattern_detected": self.pattern_detected,
            "pattern_description": self.pattern_description,
            "severity": self.severity,
            "suggested_action": self.suggested_action,
            "sample_size": self.sample_size,
        }


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in agent response: {text!r}")
    return json.loads(match.group(0))


def _format_batch(records: List[EvaluationRecord]) -> str:
    lines = []
    for i, r in enumerate(records, start=1):
        lines.append(
            f"{i}. question=\"{r.question}\" status={r.status} "
            f"faithfulness={r.faithfulness_score:.2f} hallucination_risk={r.hallucination_risk:.2f} "
            f"root_cause={r.root_cause or 'none'}"
        )
    return "\n".join(lines)


def analyze_recent_patterns(db: Session, project_id: str = None, limit: int = 20) -> AnalysisReport:
    """
    Pull the most recent `limit` evaluations (optionally scoped to a project) and
    ask Claude to find patterns across them as a batch.
    """
    query = db.query(EvaluationRecord).order_by(EvaluationRecord.created_at.desc())
    if project_id:
        query = query.filter(EvaluationRecord.project_id == project_id)
    records = query.limit(limit).all()

    if not records:
        return AnalysisReport(
            summary="No evaluations recorded yet.",
            dominant_issue="none",
            pattern_detected=False,
            pattern_description="",
            severity="low",
            suggested_action="Run some evaluations first via /api/v1/health-checks.",
            sample_size=0,
        )

    if _client is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

    batch_text = _format_batch(records)

    message = _client.messages.create(
        model=settings.MODEL_NAME,
        max_tokens=500,
        system=ANALYSIS_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Here are the {len(records)} most recent evaluations, newest first:\n\n{batch_text}\n\nAnalyze this batch now.",
        }],
    )
    raw = _extract_json(_first_text_block(message))

    return AnalysisReport(**raw, sample_size=len(records))
